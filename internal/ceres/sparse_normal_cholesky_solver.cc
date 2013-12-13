// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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

#if !defined(CERES_NO_SUITESPARSE) || !defined(CERES_NO_CXSPARSE)

#include "ceres/sparse_normal_cholesky_solver.h"

#include <algorithm>
#include <cstring>
#include <ctime>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cxsparse.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_solver.h"
#include "ceres/suitesparse.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {
namespace {
// Hack for now.

struct RowColLessThan {
  RowColLessThan(const int* rows, const int* cols)
      : rows(rows), cols(cols) {
  }

  bool operator()(const int& x, const int& y) const {
    if (rows[x] == rows[y]) {
      return (cols[x] < cols[y]);
    }
    return (rows[x] < rows[y]);
  }

  const int* rows;
  const int* cols;
};

CompressedRowSparseMatrix* FooFoo(const int num_rows,
                                  const int num_cols,
                                  const vector<int>& rows,
                                  const vector<int>& cols,
                                  vector<int>* pattern) {
  EventLogger logger("compress");
  CHECK_EQ(rows.size(), cols.size());
  CHECK_GT(rows.size(), 1);
  CHECK_NOTNULL(pattern)->clear();

  vector<int> index(rows.size());
  for (int i = 0; i < rows.size(); ++i) {
    index[i] = i;
  }

  sort(index.begin(), index.end(), RowColLessThan(&rows[0], &cols[0]));
  logger.AddEvent("sort");

  int num_nonzeros = 1;
  for (int i = 1; i < index.size(); ++i) {
    if (rows[index[i-1]] != rows[index[i]] ||
        cols[index[i-1]] != cols[index[i]]) {
      ++num_nonzeros;
    }
  }

  logger.AddEvent("nnz");
  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(num_rows, num_cols, num_nonzeros);

  int* crsm_rows = matrix->mutable_rows();
  std::fill(crsm_rows, crsm_rows + num_rows + 1, 0);
  int* crsm_cols = matrix->mutable_cols();
  std::fill(crsm_cols, crsm_cols + num_nonzeros, 0);
  pattern->resize(rows.size());


  crsm_cols[0] = cols[index[0]];
  crsm_rows[1] = 1;
  (*pattern)[index[0]] = 0;
  int nnz = 0;
  for (int i = 1; i < index.size(); ++i) {
    const int prev_idx = index[i-1];
    const int cur_idx = index[i];

    if (rows[prev_idx] != rows[cur_idx] ||  cols[prev_idx] != cols[cur_idx]) {
      crsm_cols[++nnz] = cols[cur_idx];
      ++crsm_rows[rows[cur_idx] + 1];
    }

    (*pattern)[index[i]] = nnz;
  }

  for (int i = 1; i < num_rows + 1; ++i) {
    crsm_rows[i] += crsm_rows[i-1];
  }

  logger.AddEvent("fill");
  //PrintCRSM(*matrix);
  return matrix;
}


CompressedRowSparseMatrix* CreateOuterProductMatrix(
    const CompressedRowSparseMatrix& m,
    vector<int>* pattern) {
  EventLogger logger("symbolic outer");
  vector<int> rows;
  vector<int> cols;

  const vector<int>& row_blocks = m.row_blocks();
  int row_block_begin = 0;
  for (int rblock = 0; rblock < row_blocks.size(); ++rblock) {
    const int row_block_end = row_block_begin + row_blocks[rblock];
    //for (int r = row_block_begin; r < row_block_end; ++r)
    const int r = row_block_begin;
    {
      for (int idx1 = m.rows()[r]; idx1 < m.rows()[r + 1]; ++idx1) {
        for (int idx2 = m.rows()[r]; idx2 <= idx1; ++idx2) {
          rows.push_back(m.cols()[idx1]);
          cols.push_back(m.cols()[idx2]);
        }
      }
    }
    row_block_begin = row_block_end;
  }

  CHECK_EQ(row_block_begin, m.num_rows());
  logger.AddEvent("product");

  CompressedRowSparseMatrix* value =
      FooFoo(m.num_cols(), m.num_cols(), rows, cols, pattern);
  logger.AddEvent("compress");
  return value;
}

void ComputeOuterProduct(const CompressedRowSparseMatrix& m,
                         const vector<int>& pattern,
                         CompressedRowSparseMatrix* result) {
  result->SetZero();
  double* values = result->mutable_values();
  const vector<int>& row_blocks = m.row_blocks();

  int cursor = 0;
  int row_block_begin = 0;
  for (int rblock = 0; rblock < row_blocks.size(); ++rblock) {
    const int row_block_end = row_block_begin + row_blocks[rblock];
    const int saved_cursor = cursor;
    for (int r = row_block_begin; r < row_block_end; ++r) {
      cursor = saved_cursor;
      const int row_begin = m.rows()[r];
      const int row_end = m.rows()[r + 1];
      for (int idx1 = row_begin; idx1 < row_end; ++idx1) {
        const double v1 =  m.values()[idx1];
        for (int idx2 = row_begin; idx2 <= idx1; ++idx2) {
          values[pattern[cursor++]] += v1 * m.values()[idx2];
        }
      }
    }
    row_block_begin = row_block_end;
  }

  CHECK_EQ(row_block_begin, m.num_rows());
  CHECK_EQ(cursor, pattern.size());
}

} // namespace

SparseNormalCholeskySolver::SparseNormalCholeskySolver(
    const LinearSolver::Options& options)
    : factor_(NULL),
      cxsparse_factor_(NULL),
      options_(options) {
}

SparseNormalCholeskySolver::~SparseNormalCholeskySolver() {
#ifndef CERES_NO_SUITESPARSE
  if (factor_ != NULL) {
    ss_.Free(factor_);
    factor_ = NULL;
  }
#endif

#ifndef CERES_NO_CXSPARSE
  if (cxsparse_factor_ != NULL) {
    cxsparse_.Free(cxsparse_factor_);
    cxsparse_factor_ = NULL;
  }
#endif  // CERES_NO_CXSPARSE
}

LinearSolver::Summary SparseNormalCholeskySolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  switch (options_.sparse_linear_algebra_library_type) {
    case SUITE_SPARSE:
      return SolveImplUsingSuiteSparse(A, b, per_solve_options, x);
    case CX_SPARSE:
      return SolveImplUsingCXSparse(A, b, per_solve_options, x);
    default:
      LOG(FATAL) << "Unknown sparse linear algebra library : "
                 << options_.sparse_linear_algebra_library_type;
  }

  LOG(FATAL) << "Unknown sparse linear algebra library : "
             << options_.sparse_linear_algebra_library_type;
  return LinearSolver::Summary();
}

#ifndef CERES_NO_CXSPARSE
LinearSolver::Summary SparseNormalCholeskySolver::SolveImplUsingCXSparse(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  EventLogger event_logger("SparseNormalCholeskySolver::CXSparse::Solve");

  LinearSolver::Summary summary;
  summary.num_iterations = 1;
  summary.termination_type = LINEAR_SOLVER_SUCCESS;
  summary.message = "Success.";

  const int num_cols = A->num_cols();
  Vector Atb = Vector::Zero(num_cols);
  A->LeftMultiply(b, Atb.data());

  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    CompressedRowSparseMatrix D(per_solve_options.D, A->col_blocks());
    A->AppendRows(D);
  }

  VectorRef(x, num_cols).setZero();

  // Compute the normal equations. J'J delta = J'f and solve them
  // using a sparse Cholesky factorization. Notice that when compared
  // to SuiteSparse we have to explicitly compute the transpose of Jt,
  // and then the normal equations before they can be
  // factorized. CHOLMOD/SuiteSparse on the other hand can just work
  // off of Jt to compute the Cholesky factorization of the normal
  // equations.
  if (outer_product_.get() == NULL) {
    outer_product_.reset(CreateOuterProductMatrix(*A, &pattern_));
  }

  ComputeOuterProduct(*A, pattern_, outer_product_.get());
  cs_di AtA_view = cxsparse_.CreateSparseMatrixTransposeView(outer_product_.get());
  cs_di* AtA = &AtA_view;

  if (per_solve_options.D != NULL) {
    A->DeleteRows(num_cols);
  }
  event_logger.AddEvent("Setup");

  // Compute symbolic factorization if not available.
  if (cxsparse_factor_ == NULL) {
    if (options_.use_postordering) {
      cxsparse_factor_ = cxsparse_.BlockAnalyzeCholesky(AtA,
                                                        A->col_blocks(),
                                                        A->col_blocks());
    } else {
      cxsparse_factor_ = cxsparse_.AnalyzeCholeskyWithNaturalOrdering(AtA);
    }
  }
  event_logger.AddEvent("Analysis");

  if (cxsparse_factor_ == NULL) {
    summary.termination_type = LINEAR_SOLVER_FATAL_ERROR;
    summary.message =
        "CXSparse failure. Unable to find symbolic factorization.";
  } else if (cxsparse_.SolveCholesky(AtA, cxsparse_factor_, Atb.data())) {
    VectorRef(x, Atb.rows()) = Atb;
  } else {
    summary.termination_type = LINEAR_SOLVER_FAILURE;
  }
  event_logger.AddEvent("Solve");
  return summary;
}
#else
LinearSolver::Summary SparseNormalCholeskySolver::SolveImplUsingCXSparse(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  LOG(FATAL) << "No CXSparse support in Ceres.";

  // Unreachable but MSVC does not know this.
  return LinearSolver::Summary();
}
#endif

#ifndef CERES_NO_SUITESPARSE
LinearSolver::Summary SparseNormalCholeskySolver::SolveImplUsingSuiteSparse(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  EventLogger event_logger("SparseNormalCholeskySolver::SuiteSparse::Solve");
  LinearSolver::Summary summary;
  summary.termination_type = LINEAR_SOLVER_SUCCESS;
  summary.num_iterations = 1;
  summary.message = "Success.";

  const int num_cols = A->num_cols();
  Vector Atb = Vector::Zero(num_cols);
  A->LeftMultiply(b, Atb.data());

  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    CompressedRowSparseMatrix D(per_solve_options.D, A->col_blocks());
    A->AppendRows(D);
  }

  VectorRef(x, num_cols).setZero();
  cholmod_sparse lhs = ss_.CreateSparseMatrixTransposeView(A);
  event_logger.AddEvent("Setup");

  if (factor_ == NULL) {
    if (options_.use_postordering) {
      factor_ = ss_.BlockAnalyzeCholesky(&lhs,
                                         A->col_blocks(),
                                         A->row_blocks(),
                                         &summary.message);
    } else {
      factor_ = ss_.AnalyzeCholeskyWithNaturalOrdering(&lhs, &summary.message);
    }
  }
  event_logger.AddEvent("Analysis");

  if (factor_ == NULL) {
    if (per_solve_options.D != NULL) {
      A->DeleteRows(num_cols);
    }
    summary.termination_type = LINEAR_SOLVER_FATAL_ERROR;
    return summary;
  }

  summary.termination_type = ss_.Cholesky(&lhs, factor_, &summary.message);
  if (summary.termination_type != LINEAR_SOLVER_SUCCESS) {
    if (per_solve_options.D != NULL) {
      A->DeleteRows(num_cols);
    }
    return summary;
  }

  cholmod_dense* rhs = ss_.CreateDenseVector(Atb.data(), num_cols, num_cols);
  cholmod_dense* sol = ss_.Solve(factor_, rhs, &summary.message);
  event_logger.AddEvent("Solve");

  ss_.Free(rhs);
  if (per_solve_options.D != NULL) {
    A->DeleteRows(num_cols);
  }

  if (sol != NULL) {
    memcpy(x, sol->x, num_cols * sizeof(*x));
    ss_.Free(sol);
  } else {
    summary.termination_type = LINEAR_SOLVER_FAILURE;
  }

  event_logger.AddEvent("Teardown");
  return summary;
}
#else
LinearSolver::Summary SparseNormalCholeskySolver::SolveImplUsingSuiteSparse(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {
  LOG(FATAL) << "No SuiteSparse support in Ceres.";

  // Unreachable but MSVC does not know this.
  return LinearSolver::Summary();
}
#endif

}   // namespace internal
}   // namespace ceres

#endif  // !defined(CERES_NO_SUITESPARSE) || !defined(CERES_NO_CXSPARSE)
