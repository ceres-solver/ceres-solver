// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/covariance_impl.h"

#include <algorithm>
#include <vector>
#include <utility>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/covariance.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/map_util.h"
#include "ceres/problem_impl.h"
#include "ceres/parameter_block.h"
#include "ceres/suitesparse.h"
#include "glog/logging.h"
#include "Eigen/SVD"

namespace ceres {
namespace internal {

typedef vector<pair<const double*, const double*> > CovarianceBlocks;

CovarianceImpl::CovarianceImpl(const Covariance::Options& options)
    : options_(options) {
  evaluate_options_.num_threads = options.num_threads;
  evaluate_options_.apply_loss_function = options.apply_loss_function;
}

CovarianceImpl::~CovarianceImpl() {
}

bool CovarianceImpl::Compute(const CovarianceBlocks& covariance_blocks,
                             ProblemImpl* problem) {
  problem_ = problem;

  parameter_block_to_row_index_.clear();
  covariance_matrix_.reset(NULL);

  return (ComputeCovarianceSparsity(covariance_blocks, problem) &&
          ComputeCovarianceValues());
}

bool CovarianceImpl::GetCovarianceBlock(const double* original_parameter_block1,
                                        const double* original_parameter_block2,
                                        double* covariance_block) const {
  if (constant_parameter_blocks_.count(original_parameter_block1) > 0||
      constant_parameter_blocks_.count(original_parameter_block1) > 0) {
    const ProblemImpl::ParameterMap& parameter_map = problem_->parameter_map();
    ParameterBlock* block1 =
        FindOrDie(parameter_map,
                  const_cast<double*>(original_parameter_block1));

    ParameterBlock* block2 =
        FindOrDie(parameter_map,
                  const_cast<double*>(original_parameter_block2));
    const int block1_size = block1->Size();
    const int block2_size = block2->Size();
    MatrixRef(covariance_block, block1_size, block2_size).setZero();
    // Set to zero
    return true;
  }

  const double* parameter_block1 = original_parameter_block1;
  const double* parameter_block2 = original_parameter_block2;
  const bool transpose = parameter_block1 > parameter_block2;
  if (transpose) {
    std::swap(parameter_block1, parameter_block2);
  }

  // Find where in the covariance matrix the block is located.
  const int row_begin =
      FindOrDie(parameter_block_to_row_index_, parameter_block1);
  const int col_begin =
      FindOrDie(parameter_block_to_row_index_, parameter_block2);
  const int* rows = covariance_matrix_->rows();
  const int* cols = covariance_matrix_->cols();
  const int row_size = rows[row_begin + 1] - rows[row_begin];
  const int* cols_begin = cols + rows[row_begin];

  // The only part that requires work is walking the compressed column
  // vector to determine where the set of columns correspnding to the
  // covariance block begin.
  int offset = 0;
  while (cols_begin[offset] != col_begin &&
         offset < row_size) {
    ++offset;
  }

  if (offset == row_size) {
    LOG(WARNING) << "Unable to find covariance block for "
                 << original_parameter_block1 << " "
                 << original_parameter_block2;
    return false;
  }

  const ProblemImpl::ParameterMap& parameter_map = problem_->parameter_map();
  ParameterBlock* block1 =
      FindOrDie(parameter_map, const_cast<double*>(parameter_block1));
  ParameterBlock* block2 =
      FindOrDie(parameter_map, const_cast<double*>(parameter_block2));
  const LocalParameterization* local_param1 = block1->local_parameterization();
  const LocalParameterization* local_param2 = block2->local_parameterization();
  const int block1_size = block1->Size();
  const int block1_local_size = block1->LocalSize();
  const int block2_size = block2->Size();
  const int block2_local_size = block2->LocalSize();

  ConstMatrixRef cov(covariance_matrix_->values() + rows[row_begin],
                     block1_size,
                     row_size);

  // Fast path when there are no local parameterizations.
  if (local_param1 == NULL && local_param2 == NULL) {
    if (transpose) {
      MatrixRef(covariance_block, block2_size, block1_size) =
          cov.block(0, offset, block1_size, block2_size).transpose();
    } else {
      MatrixRef(covariance_block, block1_size, block2_size) =
          cov.block(0, offset, block1_size, block2_size);
    }
    return true;
  }

  // If local parameterizations are used then the covariance that has
  // been computed is in the tangent space and it needs to be lifted
  // back to the ambient space.
  //
  // This is given by the formula
  //
  // C'_12 = J_1 C_12 J_2'
  //
  // Where C_12 is the local tangent space covariance for parameter
  // blocks 1 and 2. J_1 and J_2 are respectively the local to global
  // jacobians for parameter blocks 1 and 2.
  //
  // See Result 5.11 on page 142 of Hartley & Zisserman (2nd Edition)
  // for a proof.
  //
  // TODO(sameeragarwal): Add caching of local parameterization, so
  // that they are computed just once per parameter block.
  Matrix block1_jacobian(block1_size, block1_local_size);
  if (local_param1 == NULL) {
    block1_jacobian.setIdentity();
  } else {
    local_param1->ComputeJacobian(parameter_block2, block1_jacobian.data());
  }

  Matrix block2_jacobian(block2_size, block2_local_size);
  // Fast path if the user is requesting a diagonal block.
  if (parameter_block1 == parameter_block2) {
    block2_jacobian = block1_jacobian;
  } else {
    if (local_param2 == NULL) {
      block2_jacobian.setIdentity();
    } else {
      local_param2->ComputeJacobian(parameter_block2, block2_jacobian.data());
    }
  }

  if (transpose) {
    MatrixRef(covariance_block, block2_size, block1_size) =
        block2_jacobian *
        cov.block(0, offset, block1_local_size, block2_local_size).transpose() *
        block1_jacobian.transpose();
  } else {
    MatrixRef(covariance_block, block1_size, block2_size) =
        block1_jacobian *
        cov.block(0, offset, block1_local_size, block2_local_size) *
        block2_jacobian.transpose();
  }

  return true;
}

// Determine the sparsity pattern of the covariance matrix based on
// the block pairs requested by the user.
bool CovarianceImpl::ComputeCovarianceSparsity(
    const CovarianceBlocks&  original_covariance_blocks,
    ProblemImpl* problem) {

  // Determine an ordering for the parameter block, by sorting the
  // parameter blocks by their pointers.
  vector<double*> all_parameter_blocks;
  problem->GetParameterBlocks(&all_parameter_blocks);
  const ProblemImpl::ParameterMap& parameter_map = problem->parameter_map();
  constant_parameter_blocks_.clear();
  vector<double*>& active_parameter_blocks = evaluate_options_.parameter_blocks;
  active_parameter_blocks.clear();
  for (int i = 0; i < all_parameter_blocks.size(); ++i) {
    double* parameter_block = all_parameter_blocks[i];

    ParameterBlock* block = FindOrDie(parameter_map, parameter_block);
    if (block->IsConstant()) {
      constant_parameter_blocks_.insert(parameter_block);
    } else {
      active_parameter_blocks.push_back(parameter_block);
    }
  }

  sort(active_parameter_blocks.begin(), active_parameter_blocks.end());

  // Compute the number of rows.  Map each parameter block to the
  // first row corresponding to it in the covariance matrix using the
  // ordering of parameter blocks just constructed.
  int num_rows = 0;
  parameter_block_to_row_index_.clear();
  for (int i = 0; i < active_parameter_blocks.size(); ++i) {
    double* parameter_block = active_parameter_blocks[i];
    const int parameter_block_size =
        problem->ParameterBlockLocalSize(parameter_block);
    parameter_block_to_row_index_[parameter_block] = num_rows;
    num_rows += parameter_block_size;
  }

  // Compute the number of non-zeros in the covariance matrix.  Along
  // the way flip any covariance blocks which are in the lower
  // triangular part of the matrix.
  int num_nonzeros = 0;
  CovarianceBlocks covariance_blocks;
  for (int i = 0; i <  original_covariance_blocks.size(); ++i) {
    const pair<const double*, const double*>& block_pair =
        original_covariance_blocks[i];
    if (constant_parameter_blocks_.count(block_pair.first) > 0||
        constant_parameter_blocks_.count(block_pair.second) > 0) {
      continue;
    }

    int index1 = FindOrDie(parameter_block_to_row_index_, block_pair.first);
    int index2 = FindOrDie(parameter_block_to_row_index_, block_pair.second);
    const int size1 = problem->ParameterBlockLocalSize(block_pair.first);
    const int size2 = problem->ParameterBlockLocalSize(block_pair.second);
    num_nonzeros += size1 * size2;

      // Make sure we are constructing a block upper triangular matrix.
    if (index1 > index2) {
      covariance_blocks.push_back(make_pair(block_pair.second,
                                            block_pair.first));
    } else {
      covariance_blocks.push_back(block_pair);
    }
  }

  if (covariance_blocks.size() == 0) {
    VLOG(2) << "No non-zero covariance blocks found";
    covariance_matrix_.reset(NULL);
    return true;
  }

  // Sort the block pairs. As a consequence we get the covariance
  // blocks as they will occur in the CompressedRowSparseMatrix that
  // will store the covariance.
  sort(covariance_blocks.begin(), covariance_blocks.end());

  // Fill the sparsity pattern of the covariance matrix.
  covariance_matrix_.reset(
      new CompressedRowSparseMatrix(num_rows, num_rows, num_nonzeros));

  int* rows = covariance_matrix_->mutable_rows();
  int* cols = covariance_matrix_->mutable_cols();

  int i = 0;
  int cursor = 0;
  for (map<const double*, int>::const_iterator it =
           parameter_block_to_row_index_.begin();
       it != parameter_block_to_row_index_.end();
       ++it) {
    const double* row_block =  it->first;
    const int row_block_size = problem->ParameterBlockLocalSize(row_block);
    int row_begin = it->second;

    int num_col_blocks = 0;
    int num_columns = 0;
    // Iterate over the covariance blocks contained in this row block
    // and count the number of columns in this row block.
    for (int j = i; j < covariance_blocks.size(); ++j, ++num_col_blocks) {
      const pair<const double*, const double*>& block_pair =
          covariance_blocks[j];
      if (block_pair.first != row_block) {
        break;
      }
      num_columns += problem->ParameterBlockLocalSize(block_pair.second);
    }

    // Actually fill out all the compressed rows.
    for (int r = 0; r < row_block_size; ++r) {
      rows[row_begin + r] = cursor;
      for (int c = 0; c < num_col_blocks; ++c) {
        const double* col_block = covariance_blocks[i + c].second;
        const int col_block_size = problem->ParameterBlockLocalSize(col_block);
        int col_begin = FindOrDie(parameter_block_to_row_index_, col_block);
        for (int k = 0; k < col_block_size; ++k) {
          cols[cursor++] = col_begin++;
        }
      }
    }

    i+= num_col_blocks;
  }

  rows[num_rows] = cursor;
  return true;
}

bool CovarianceImpl::ComputeCovarianceValues() {
  if (options_.use_dense_algorithm) {
    return ComputeCovarianceValuesUsingEigen();
  } else {
#ifndef CERES_NO_SUITESPARSE
    return ComputeCovarianceValuesUsingSuiteSparse();
#else
    LOG(ERROR) << "Ceres compiled without SuiteSparse. "
               << "Large scale covariance computation is not possible.";
    return false
#endif
  }
}

bool CovarianceImpl::ComputeCovarianceValuesUsingSuiteSparse() {
#ifndef CERES_NO_SUITESPARSE
  if (covariance_matrix_.get() == NULL) {
    // Nothing to do, all zeros covariance matrix.
    return true;
  }

  CRSMatrix jacobian;
  problem_->Evaluate(evaluate_options_, NULL, NULL, NULL, &jacobian);

  // m is a transposed view of the Jacobian.
  cholmod_sparse m;
  m.nrow = jacobian.num_cols;
  m.ncol = jacobian.num_rows;
  m.nzmax = jacobian.values.size();
  m.nz = NULL;
  m.p = reinterpret_cast<void*>(&jacobian.rows[0]);
  m.i = reinterpret_cast<void*>(&jacobian.cols[0]);
  m.x = reinterpret_cast<void*>(&jacobian.values[0]);
  m.z = NULL;
  m.stype = 0;  // Matrix is not symmetric.
  m.itype = CHOLMOD_INT;
  m.xtype = CHOLMOD_REAL;
  m.dtype = CHOLMOD_DOUBLE;
  m.sorted = 1;
  m.packed = 1;

  cholmod_factor* factor = ss_.AnalyzeCholesky(&m);
  if (!ss_.Cholesky(&m, factor)) {
    ss_.Free(factor);
    LOG(WARNING) << "Cholesky factorization failed.";
    return false;
  }

  const int num_rows = covariance_matrix_->num_rows();
  const int* rows = covariance_matrix_->rows();
  const int* cols = covariance_matrix_->cols();
  double* values = covariance_matrix_->mutable_values();

  cholmod_dense* rhs = ss_.CreateDenseVector(NULL, num_rows, num_rows);
  double* rhs_x = reinterpret_cast<double*>(rhs->x);

  // The following loop exploits the fact that the i^th column of A^{-1}
  // is given by the solution to the linear system
  //
  // A x = e_i
  //
  // where e_i is a vector with e(i) = 1 and all other entries zero.
  //
  // Since the covariance matrix is symmetric, the i^th row and column
  // are equal.
  //
  // The ifdef separates two different version of SuiteSparse. Newer
  // versions of SuiteSparse have the cholmod_solve2 function which
  // re-uses memory across calls.
#if (SUITESPARSE_VERSION < 4002)
#pragma omp parallel for num_threads(options.num_threads) schedule(dynamic)
  for (int r = 0; r < num_rows; ++r) {
    int row_begin = rows[r];
    int row_end = rows[r + 1];
    if (row_end == row_begin) {
      continue;
    }

    rhs_x[r] = 1.0;
    cholmod_dense* solution = ss_.Solve(factor, rhs);
    double* solution_x = reinterpret_cast<double*>(solution->x);
    for (int idx = row_begin; idx < row_end; ++idx) {
      const int c = cols[idx];
      values[idx] = solution_x[c];
    }
    ss_.Free(solution);
    rhs_x[r] = 0.0;
  }
#else
  // TODO(sameeragarwal) There should be a more efficient way
  // involving the use of Bset but I am unable to make it work right
  // now.
  cholmod_dense* solution = NULL;
  cholmod_sparse* solution_set = NULL;
  cholmod_dense* y_workspace = NULL;
  cholmod_dense* e_workspace = NULL;

  for (int r = 0; r < num_rows; ++r) {
    int row_begin = rows[r];
    int row_end = rows[r + 1];
    if (row_end == row_begin) {
      continue;
    }

    rhs_x[r] = 1.0;

    cholmod_solve2(CHOLMOD_A,
                   factor,
                   rhs,
                   NULL,
                   &solution,
                   &solution_set,
                   &y_workspace,
                   &e_workspace,
                   ss_.mutable_cc());

    double* solution_x = reinterpret_cast<double*>(solution->x);
    for (int idx = row_begin; idx < row_end; ++idx) {
      const int c = cols[idx];
      values[idx] = solution_x[c];
    }
    rhs_x[r] = 0.0;
  }

  ss_.Free(solution);
  ss_.Free(solution_set);
  ss_.Free(y_workspace);
  ss_.Free(e_workspace);

#endif

  ss_.Free(rhs);
  ss_.Free(factor);
  return true;
#else
  return false;
#endif
};

bool CovarianceImpl::ComputeCovarianceValuesUsingEigen() {
  if (covariance_matrix_.get() == NULL) {
    // Nothing to do, all zeros covariance matrix.
    return true;
  }

  CRSMatrix jacobian;
  problem_->Evaluate(evaluate_options_, NULL, NULL, NULL, &jacobian);
  Matrix dense_jacobian(jacobian.num_rows, jacobian.num_cols);
  dense_jacobian.setZero();
  for (int r = 0; r < jacobian.num_rows; ++r) {
    for (int idx = jacobian.rows[r]; idx < jacobian.rows[r+1]; ++idx) {
      const int c = jacobian.cols[idx];
      dense_jacobian(r,c) = jacobian.values[idx];
    }
  }

  Eigen::JacobiSVD<Matrix> svd(dense_jacobian,
                               Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vector inverse_singular_values = svd.singularValues();

  for (int i = 0; i < inverse_singular_values.rows(); ++i) {
    if (inverse_singular_values[i] > options_.min_singular_value_threshold &&
        i < (inverse_singular_values.rows() - options_.null_space_rank)) {
      inverse_singular_values[i] =
          1./(inverse_singular_values[i] * inverse_singular_values[i]);
    } else {
      inverse_singular_values[i] = 0.0;
    }
  }

  Matrix dense_covariance =
      svd.matrixV() *
      inverse_singular_values.asDiagonal() *
      svd.matrixV().transpose();

  const int num_rows = covariance_matrix_->num_rows();
  const int* rows = covariance_matrix_->rows();
  const int* cols = covariance_matrix_->cols();
  double* values = covariance_matrix_->mutable_values();

  for (int r = 0; r < num_rows; ++r) {
    for (int idx = rows[r]; idx < rows[r+1]; ++idx) {
      const int c = cols[idx];
      values[idx] = dense_covariance(r,c);
    }
  }

  return true;
};



}  // namespace internal
}  // namespace ceres
