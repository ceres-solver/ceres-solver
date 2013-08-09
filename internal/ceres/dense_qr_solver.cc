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

#include "ceres/dense_qr_solver.h"


#include <cstddef>
#include "Eigen/Dense"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/linear_solver.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"

// C interface to the LAPACK linear least squares solver.
extern "C" void dgels_(char *uplo,
                       int* M,
                       int* N,
                       int* NRHS,
                       double* A,
                       int* LDA,
                       double* B,
                       int* LDB,
                       double* work,
                       int* lwork,
                       int* info);

namespace ceres {
namespace internal {

DenseQRSolver::DenseQRSolver(const LinearSolver::Options& options)
    : options_(options) {
  work_.resize(1);
}

LinearSolver::Summary DenseQRSolver::SolveImpl(
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
LinearSolver::Summary DenseQRSolver::SolveUsingLAPACK(
    DenseSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("DenseQRSolver::Solve");

  const int num_rows = A->num_rows();
  const int num_cols = A->num_cols();

  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    A->AppendDiagonal(per_solve_options.D);
  }

  lhs_ =  A->matrix();

  if (per_solve_options.D != NULL) {
    // Undo the modifications to the matrix A.
    A->RemoveDiagonal();
  }

  // rhs = [b;0] to account for the additional rows in the lhs.
  const int augmented_num_rows =
      num_rows + ((per_solve_options.D != NULL) ? num_cols : 0);
  if (rhs_.rows() != augmented_num_rows) {
    rhs_.resize(augmented_num_rows);
  }
  rhs_.setZero();
  rhs_.head(num_rows) = ConstVectorRef(b, num_rows);

  char TRANS = 'N';
  int M = lhs_.rows();
  int N = lhs_.cols();
  int NRHS = 1;
  int LDA = M;
  int LDB = M;
  int NB = 64;

  int LWORK = N + M * NB;
  if (work_.rows() == 1) {
    work_.resize(LWORK);
  } else {
    LWORK = work_.rows();
  }

  int INFO = 0;
  dgels_(&TRANS, &M, &N, &NRHS, lhs_.data(), &LDA, rhs_.data(), &LDB, work_.data(), &LWORK, &INFO);
  if (work_.rows() < static_cast<int>(work_[0])) {
    VLOG(2) << "Resizing work_ to : << " << work_[0];
    work_.resize(static_cast<int>(work_[0]));
  }

  LinearSolver::Summary summary;
  summary.num_iterations = 1;

  if (INFO == 0) {
    VectorRef(x, num_cols) = rhs_.head(num_cols);
    event_logger.AddEvent("Solve");
    summary.termination_type = TOLERANCE;
  } else {
    summary.termination_type = FAILURE;
  }

  event_logger.AddEvent("TearDown");
  return summary;
}

LinearSolver::Summary DenseQRSolver::SolveUsingEigen(
    DenseSparseMatrix* A,
    const double* b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    double* x) {
  EventLogger event_logger("DenseQRSolver::Solve");

  const int num_rows = A->num_rows();
  const int num_cols = A->num_cols();


  if (per_solve_options.D != NULL) {
    // Temporarily append a diagonal block to the A matrix, but undo
    // it before returning the matrix to the user.
    A->AppendDiagonal(per_solve_options.D);
  }

  // rhs = [b;0] to account for the additional rows in the lhs.
  const int augmented_num_rows =
      num_rows + ((per_solve_options.D != NULL) ? num_cols : 0);
  if (rhs_.rows() != augmented_num_rows) {
    rhs_.resize(augmented_num_rows);
    rhs_.setZero();
  }
  rhs_.head(num_rows) = ConstVectorRef(b, num_rows);
  event_logger.AddEvent("Setup");

  // Solve the system.
  //VectorRef(x, num_cols) = A->matrix().colPivHouseholderQr().solve(rhs_);
  VectorRef(x, num_cols) = A->matrix().householderQr().solve(rhs_);
  event_logger.AddEvent("Solve");

  if (per_solve_options.D != NULL) {
    // Undo the modifications to the matrix A.
    A->RemoveDiagonal();
  }

  // We always succeed, since the QR solver returns the best solution
  // it can. It is the job of the caller to determine if the solution
  // is good enough or not.
  LinearSolver::Summary summary;
  summary.num_iterations = 1;
  summary.termination_type = TOLERANCE;

  event_logger.AddEvent("TearDown");
  return summary;
}


}   // namespace internal
}   // namespace ceres
