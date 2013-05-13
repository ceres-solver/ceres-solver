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
#include "ceres/problem.h"
#include "ceres/suitesparse.h"
#include "glog/logging.h"
#include "ceres/map_util.h"

namespace ceres {
namespace internal {

CovarianceImpl::CovarianceImpl(const Covariance::Options& options) {
  evaluate_options_.num_threads = options.num_threads;
  evaluate_options_.apply_loss_function = options.apply_loss_function;
}

CovarianceImpl::~CovarianceImpl() {
}

bool CovarianceImpl::Compute(
    const vector<pair<double*, double*> >& covariance_blocks,
    Problem* problem) {
  problem_ = problem;

  parameter_block_to_row_index_.clear();
  covariance_matrix_.reset(NULL);

  return (ComputeCovarianceSparsity(covariance_blocks, problem) &&
          ComputeCovarianceValues());
}

bool CovarianceImpl::GetCovarianceBlock(double* original_parameter_block1,
                                          double* original_parameter_block2,
                                          double* covariance_block) {
  double* parameter_block1 = original_parameter_block1;
  double* parameter_block2 = original_parameter_block2;
  const bool transpose = parameter_block1 > parameter_block2;
  if (transpose) {
    parameter_block1 = original_parameter_block2;
    parameter_block2 = original_parameter_block1;
  }

  const int row_begin = FindOrDie(parameter_block_to_row_index_, parameter_block1);
  const int col_begin = FindOrDie(parameter_block_to_row_index_, parameter_block2);
  const int row_block_size = problem_->ParameterBlockLocalSize(parameter_block1);
  const int col_block_size = problem_->ParameterBlockLocalSize(parameter_block2);

  const int* rows = covariance_matrix_->rows();
  const int* cols = covariance_matrix_->cols();
  const int row_size = rows[row_begin + 1] - rows[row_begin];
  const int* cols_begin = cols + rows[row_begin];
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

  ConstMatrixRef cov(covariance_matrix_->values() + rows[row_begin],
                     row_block_size,
                     row_size);

  if (transpose) {
    MatrixRef(covariance_block, col_block_size, row_block_size)
        = cov.block(0, offset, row_block_size, col_block_size).transpose();
  } else {
    MatrixRef(covariance_block, row_block_size, col_block_size)
        = cov.block(0, offset, row_block_size, col_block_size);
  }

  return true;
}

bool CovarianceImpl::ComputeCovarianceSparsity(
    const vector<pair<double*, double*> >& original_covariance_blocks,
    Problem* problem) {
  vector<double*>& parameter_blocks = evaluate_options_.parameter_blocks;
  parameter_blocks.clear();

  problem->GetParameterBlocks(&parameter_blocks);
  sort(parameter_blocks.begin(), parameter_blocks.end());

  parameter_block_to_row_index_.clear();

  int num_rows = 0;
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    double* parameter_block = parameter_blocks[i];
    const int parameter_block_size =
        problem->ParameterBlockLocalSize(parameter_block);

    CHECK_EQ(problem->ParameterBlockSize(parameter_block), parameter_block_size)
        << "No support yet for local parameterizations, coming soon!";
    parameter_block_to_row_index_[parameter_block] = num_rows;
    num_rows += parameter_block_size;
  }

  vector<pair<double*, double*> > covariance_blocks;
  int num_nonzeros = 0;
  for (int i = 0; i <  original_covariance_blocks.size(); ++i) {
    const pair<double*, double*>& block_pair =
        original_covariance_blocks[i];

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

  sort(covariance_blocks.begin(), covariance_blocks.end());

  covariance_matrix_.reset(
      new CompressedRowSparseMatrix(num_rows, num_rows, num_nonzeros));

  int* rows = covariance_matrix_->mutable_rows();
  int* cols = covariance_matrix_->mutable_cols();

  int i = 0;
  int cursor = 0;
  for (map<double*, int>::const_iterator it = parameter_block_to_row_index_.begin();
       it != parameter_block_to_row_index_.end();
       ++it) {

    double* row_block =  it->first;
    const int row_block_size = problem->ParameterBlockLocalSize(row_block);
    int row_begin = it->second;

    int num_col_blocks = 0;
    int num_columns = 0;
    for (int j = i; j < covariance_blocks.size(); ++j, ++num_col_blocks) {
      const pair<double*, double*>& block_pair = covariance_blocks[j];
      if (block_pair.first != row_block) {
        break;
      }

      num_columns += problem->ParameterBlockLocalSize(block_pair.second);
    }

    for (int r = 0; r < row_block_size; ++r) {
      rows[row_begin + r] = cursor;
      for (int c = 0; c < num_col_blocks; ++c) {

        double* col_block = covariance_blocks[i + c].second;
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
  CRSMatrix jacobian;
  problem_->Evaluate(evaluate_options_, NULL, NULL, NULL, &jacobian);

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
    return false;
  }

  const int num_rows = covariance_matrix_->num_rows();
  const int* rows = covariance_matrix_->rows();
  const int* cols = covariance_matrix_->cols();
  double* values = covariance_matrix_->mutable_values();

  cholmod_dense* rhs = ss_.CreateDenseVector(NULL, num_rows, num_rows);
  double* rhs_x = reinterpret_cast<double*>(rhs->x);

#if (SUITESPARSE_VERSION < 4002)
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
};



}  // namespace internal
}  // namespace ceres
