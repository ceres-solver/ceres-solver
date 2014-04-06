// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#if !defined(CERES_NO_SUITESPARSE)

#include "ceres/dynamic_sparse_normal_cholesky_solver.h"

#include "ceres/compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

DynamicSparseNormalCholeskySolver::DynamicSparseNormalCholeskySolver(
  const LinearSolver::Options& options)
  : options_(options) {
}

DynamicSparseNormalCholeskySolver::~DynamicSparseNormalCholeskySolver() {
}

LinearSolver::Summary DynamicSparseNormalCholeskySolver::SolveImpl(
    CompressedRowSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double * x) {

  const int num_cols = A->num_cols();
  VectorRef(x, num_cols).setZero();
  A->LeftMultiply(b, x);

  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    scoped_ptr<CompressedRowSparseMatrix> regularizer;
    if (A->col_blocks().size() > 0) {
      regularizer.reset(CompressedRowSparseMatrix::CreateBlockDiagonalMatrix(
                            per_solve_options.D, A->col_blocks()));
    } else {
      regularizer.reset(new CompressedRowSparseMatrix(
                            per_solve_options.D, num_cols));
    }
    A->AppendRows(*regularizer);
  }

  LinearSolver::Summary summary;
  switch (options_.sparse_linear_algebra_library_type) {
    case SUITE_SPARSE:
      summary = SolveImplUsingSuiteSparse(A, per_solve_options, x);
      break;
    default:
      LOG(FATAL) << "Unknown sparse linear algebra library : "
                 << options_.sparse_linear_algebra_library_type;
  }

  if (per_solve_options.D != NULL) {
    A->DeleteRows(num_cols);
  }

  return summary;
}

#ifndef CERES_NO_SUITESPARSE
LinearSolver::Summary
DynamicSparseNormalCholeskySolver::SolveImplUsingSuiteSparse(
    CompressedRowSparseMatrix* A,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* rhs_and_solution) {
  EventLogger event_logger(
    "DynamicSparseNormalCholeskySolver::SuiteSparse::Solve");
  LinearSolver::Summary summary;
  summary.num_iterations = 1;
  summary.termination_type = LINEAR_SOLVER_SUCCESS;
  summary.message = "Success.";

  const int num_cols = A->num_cols();
  cholmod_sparse lhs = ss_.CreateSparseMatrixTransposeView(A);
  cholmod_dense* rhs = ss_.CreateDenseVector(rhs_and_solution,
                                             num_cols, num_cols);
  event_logger.AddEvent("Setup");

  cholmod_factor* factor = ss_.AnalyzeCholeskyWithNaturalOrdering(
    &lhs, &summary.message);
  event_logger.AddEvent("Analysis");

  if (factor == NULL) {
    summary.termination_type = LINEAR_SOLVER_FATAL_ERROR;
    return summary;
  }

  summary.termination_type = ss_.Cholesky(&lhs, factor, &summary.message);
  if (summary.termination_type != LINEAR_SOLVER_SUCCESS) {
    return summary;
  }

  cholmod_dense* solution = ss_.Solve(factor, rhs, &summary.message);
  event_logger.AddEvent("Solve");

  ss_.Free(rhs);
  ss_.Free(factor);
  if (solution != NULL) {
    memcpy(rhs_and_solution, solution->x,
           num_cols * sizeof(*rhs_and_solution));
    ss_.Free(solution);
  } else {
    summary.termination_type = LINEAR_SOLVER_FAILURE;
  }

  event_logger.AddEvent("Teardown");
  return summary;
}
#else
LinearSolver::Summary
DynamicSparseNormalCholeskySolver::SolveImplUsingSuiteSparse(
    CompressedRowSparseMatrix* A,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* rhs_and_solution) {
  LOG(FATAL) << "No SuiteSparse support in Ceres.";

  // Unreachable but MSVC does not know this.
  return LinearSolver::Summary();
}
#endif

}   // namespace internal
}   // namespace ceres

#endif // !defined(CERES_NO_SUITESPARSE)
