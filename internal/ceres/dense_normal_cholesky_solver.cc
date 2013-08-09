// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include "ceres/dense_normal_cholesky_solver.h"

#include <cstddef>

#include "Eigen/Dense"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_solver.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"

// C interface to the LAPACK Cholesky factorization and triangular solve.
extern "C" void dpotrf_(char *uplo,
                       int* N,
                       double* A,
                       int* LDA,
                       int* info);

extern "C" void dpotrs_(char* uplo,
                        int* N,
                        int* NRHS,
                        double* A,
                        int* LDA,
                        double* B,
                        int* LDB,
                        int* INFO);

extern "C" void dsyrk_(char* uplo,
                       char* trans,
                       int* N,
                       int* K,
                       double* alpha,
                       double* A,
                       int* LDA,
                       double* BETA,
                       double* C,
                       int* LDC);

namespace ceres {
namespace internal {

DenseNormalCholeskySolver::DenseNormalCholeskySolver(
    const LinearSolver::Options& options)
    : options_(options) {}

LinearSolver::Summary DenseNormalCholeskySolver::SolveImpl(
    DenseSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
   if (options_.dense_linear_algebra_library_type == EIGEN) {
     return SolveUsingEigen(A, b, per_solve_options, x);
   } else {
     return SolveUsingLAPACK(A, b, per_solve_options, x);
   }
}

LinearSolver::Summary DenseNormalCholeskySolver::SolveUsingEigen(
    DenseSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("DenseNormalCholeskySolver::Solve");

  const int num_rows = A->num_rows();
  const int num_cols = A->num_cols();

  ConstColMajorMatrixRef Aref = A->matrix();
  Matrix lhs(num_cols, num_cols);
  lhs.setZero();

  event_logger.AddEvent("Setup");
  //   lhs += A'A
  //
  // Using rankUpdate instead of GEMM, exposes the fact that its the
  // same matrix being multiplied with itself and that the product is
  // symmetric.
  lhs.selfadjointView<Eigen::Upper>().rankUpdate(Aref.transpose());

  //   rhs = A'b
  Vector rhs = Aref.transpose() * ConstVectorRef(b, num_rows);

  if (per_solve_options.D != NULL) {
    ConstVectorRef D(per_solve_options.D, num_cols);
    lhs += D.array().square().matrix().asDiagonal();
  }
  event_logger.AddEvent("Product");

  // Use dsyrk instead for the product.
  LinearSolver::Summary summary;
  summary.num_iterations = 1;
  summary.termination_type = TOLERANCE;
  VectorRef(x, num_cols) =
      lhs.selfadjointView<Eigen::Upper>().ldlt().solve(rhs);
  event_logger.AddEvent("Solve");
  return summary;
}

LinearSolver::Summary DenseNormalCholeskySolver::SolveUsingLAPACK(
    DenseSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("DenseNormalCholeskySolver::Solve");

  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    A->AppendDiagonal(per_solve_options.D);
  }

  const int num_rows = A->num_rows();
  const int num_cols = A->num_cols();

  Matrix lhs(num_cols, num_cols);

  event_logger.AddEvent("Setup");

  {
    char uplo = 'L';
    char trans = 'T';
    int N = num_cols;
    int K = num_rows;
    double alpha = 1.0;
    double beta = 0.0;
    int LDA = num_rows;
    int LDC = num_cols;

    // This is a bit hairy because of the underlying matrix size assumptions.
    // we need to make sure that we know the layout of the underlying memory.
    dsyrk_(&uplo, &trans, &N, &K, &alpha, A->mutable_values(), &LDA, &beta, lhs.data(), &LDC);
  }

  if (per_solve_options.D != NULL) {
    // Undo the modifications to the matrix A.
    A->RemoveDiagonal();
  }

  //   rhs = A'b
  VectorRef(x, num_cols) =  A->matrix().transpose() * ConstVectorRef(b, num_rows);

  if (per_solve_options.D != NULL) {
    ConstVectorRef D(per_solve_options.D, num_cols);
    lhs += D.array().square().matrix().asDiagonal();
  }
  event_logger.AddEvent("Product");

  LinearSolver::Summary summary;
  summary.num_iterations = 1;
  summary.termination_type = TOLERANCE;

  char uplo = 'L';
  int N = num_cols;
  int info = 0;
  int nrhs = 1;

  dpotrf_(&uplo, &N, lhs.data(), &N, &info);
  if (info == 0) {
    dpotrs_(&uplo, &N, &nrhs, lhs.data(), &N, x, &N, &info);
  }
  event_logger.AddEvent("Solve");

  if (info != 0) {
    LOG(INFO) << "solve failed";
  }

  return summary;
}
}   // namespace internal
}   // namespace ceres
