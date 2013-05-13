#include "ceres/sparse_covariance.h"

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

SparseCovariance::SparseCovariance(const Covariance::Options& options) {
}

SparseCovariance::~SparseCovariance() {
}


bool SparseCovariance::Compute(
    const vector<pair<double*, double*> >& covariance_blocks,
    Problem* problem) {
  problem_ = problem;

  parameter_block_to_row_index_.clear();
  sparse_covariance_.reset(NULL);

  return (ComputeCovarianceSparsity(covariance_blocks, problem) &&
          ComputeCovarianceValues());
}

bool SparseCovariance::GetCovarianceBlock(double* original_parameter_block1,
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

  const int* rows = sparse_covariance_->rows();
  const int* cols = sparse_covariance_->cols();
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

  ConstMatrixRef cov(sparse_covariance_->values() + rows[row_begin],
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

bool SparseCovariance::ComputeCovarianceSparsity(
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

  sparse_covariance_.reset(
      new CompressedRowSparseMatrix(num_rows, num_rows, num_nonzeros));

  int* rows = sparse_covariance_->mutable_rows();
  int* cols = sparse_covariance_->mutable_cols();

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

bool SparseCovariance::ComputeCovarianceValues() {
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

  const int num_rows = sparse_covariance_->num_rows();
  const int* rows = sparse_covariance_->rows();
  const int* cols = sparse_covariance_->cols();
  double* values = sparse_covariance_->mutable_values();

  // A first basic and expensive implementation of covariance estimation.
  cholmod_dense* rhs = ss_.CreateDenseVector(NULL, num_rows, num_rows);
  double* rhs_x = reinterpret_cast<double*>(rhs->x);
  for (int r = 0; r < sparse_covariance_->num_rows(); ++r) {
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

  ss_.Free(rhs);
  ss_.Free(factor);
  return true;
};



}  // namespace internal
}  // namespace ceres
