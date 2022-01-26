// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
// http://ceres-solver.org/
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

#include "ceres/dense_cholesky.h"

#include <algorithm>
#include <memory>
#include <string>

#ifndef CERES_NO_LAPACK

// C interface to the LAPACK Cholesky factorization and triangular solve.
extern "C" void dpotrf_(
    const char* uplo, const int* n, double* a, const int* lda, int* info);

extern "C" void dpotrs_(const char* uplo,
                        const int* n,
                        const int* nrhs,
                        const double* a,
                        const int* lda,
                        double* b,
                        const int* ldb,
                        int* info);
#endif

namespace ceres {
namespace internal {

std::unique_ptr<DenseCholesky> DenseCholesky::Create(
    const LinearSolver::Options& options) {
  std::unique_ptr<DenseCholesky> dense_cholesky;

  switch (options.dense_linear_algebra_library_type) {
    case EIGEN:
      dense_cholesky = std::make_unique<EigenDenseCholesky>();
      break;

    case LAPACK:
#ifndef CERES_NO_LAPACK
      dense_cholesky = std::make_unique<LAPACKDenseCholesky>();
      break;
#else
      LOG(FATAL) << "Ceres was compiled without support for LAPACK.";
#endif

    default:
      LOG(FATAL) << "Unknown dense linear algebra library type : "
                 << DenseLinearAlgebraLibraryTypeToString(
                        options.dense_linear_algebra_library_type);
  }
  return dense_cholesky;
}

LinearSolverTerminationType DenseCholesky::FactorAndSolve(
    int num_cols,
    double* lhs,
    const double* rhs,
    double* solution,
    std::string* message) {
  LinearSolverTerminationType termination_type =
      Factorize(num_cols, lhs, message);
  if (termination_type == LINEAR_SOLVER_SUCCESS) {
    termination_type = Solve(rhs, solution, message);
  }
  return termination_type;
}

LinearSolverTerminationType EigenDenseCholesky::Factorize(
    int num_cols, double* lhs, std::string* message) {
  Eigen::Map<Eigen::MatrixXd> m(lhs, num_cols, num_cols);
  llt_ = std::make_unique<LLTType>(m);
  if (llt_->info() != Eigen::Success) {
    *message = "Eigen failure. Unable to perform dense Cholesky factorization.";
    return LINEAR_SOLVER_FAILURE;
  }

  *message = "Success.";
  return LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType EigenDenseCholesky::Solve(const double* rhs,
                                                      double* solution,
                                                      std::string* message) {
  if (llt_->info() != Eigen::Success) {
    *message = "Eigen failure. Unable to perform dense Cholesky factorization.";
    return LINEAR_SOLVER_FAILURE;
  }

  VectorRef(solution, llt_->cols()) =
      llt_->solve(ConstVectorRef(rhs, llt_->cols()));
  *message = "Success.";
  return LINEAR_SOLVER_SUCCESS;
}

#ifndef CERES_NO_LAPACK
LinearSolverTerminationType LAPACKDenseCholesky::Factorize(
    int num_cols, double* lhs, std::string* message) {
  lhs_ = lhs;
  num_cols_ = num_cols;

  const char uplo = 'L';
  int info = 0;
  dpotrf_(&uplo, &num_cols_, lhs_, &num_cols_, &info);

  if (info < 0) {
    termination_type_ = LINEAR_SOLVER_FATAL_ERROR;
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "LAPACK::dpotrf fatal error."
               << "Argument: " << -info << " is invalid.";
  } else if (info > 0) {
    termination_type_ = LINEAR_SOLVER_FAILURE;
    *message = StringPrintf(
        "LAPACK::dpotrf numerical failure. "
        "The leading minor of order %d is not positive definite.",
        info);
  } else {
    termination_type_ = LINEAR_SOLVER_SUCCESS;
    *message = "Success.";
  }
  return termination_type_;
}

LinearSolverTerminationType LAPACKDenseCholesky::Solve(const double* rhs,
                                                       double* solution,
                                                       std::string* message) {
  const char uplo = 'L';
  const int nrhs = 1;
  int info = 0;

  std::copy_n(rhs, num_cols_, solution);
  dpotrs_(
      &uplo, &num_cols_, &nrhs, lhs_, &num_cols_, solution, &num_cols_, &info);

  if (info < 0) {
    termination_type_ = LINEAR_SOLVER_FATAL_ERROR;
    LOG(FATAL) << "Congratulations, you found a bug in Ceres."
               << "Please report it."
               << "LAPACK::dpotrs fatal error."
               << "Argument: " << -info << " is invalid.";
  }

  *message = "Success";
  termination_type_ = LINEAR_SOLVER_SUCCESS;

  return termination_type_;
}

#endif  // CERES_NO_LAPACK

}  // namespace internal
}  // namespace ceres
