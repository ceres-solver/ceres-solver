// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Authors: sameeragarwal@google.com (Sameer Agarwal)
//          joydeepb@cs.utexas.edu (Joydeep Biswas)
//
// CUDA-Accelerated Conjugate Gradients based solver for positive
// semidefinite linear systems.

#include "ceres/cuda_conjugate_gradients_solver.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>

#include "ceres/internal/eigen.h"
#include "ceres/linear_operator.h"
#include "ceres/linear_solver.h"
#include "ceres/stringprintf.h"
#include "glog/logging.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_linear_operator.h"
#include "ceres/cuda_vector.h"

namespace ceres::internal {
namespace {

bool IsZeroOrInfinity(double x) { return ((x == 0.0) || std::isinf(x)); }

}  // namespace

std::unique_ptr<CudaConjugateGradientsSolver>
CudaConjugateGradientsSolver::Create(const LinearSolver::Options& options) {
  // TODO(Joydeep Biswas): Check if the options are valid.
  return std::unique_ptr<CudaConjugateGradientsSolver>(
      new CudaConjugateGradientsSolver(options));
}

bool CudaConjugateGradientsSolver::Init(
    ContextImpl* context, std::string* message) {
  if (!r_.Init(context, message) ||
      !p_.Init(context, message) ||
      !z_.Init(context, message) ||
      !tmp_.Init(context, message)) {
    return false;
  }
  context_ = context;
  return true;
}

LinearSolver::Summary CudaConjugateGradientsSolver::Solve(
    CudaLinearOperator* A_ptr,
    CudaPreconditioner* preconditioner,
    const CudaVector& b,
    const LinearSolver::PerSolveOptions& per_solve_options,
    CudaVector* x_ptr) {
  static const bool kDebug = false;
  CHECK(A_ptr != nullptr);
  CHECK(x_ptr != nullptr);
  CudaVector& x = *x_ptr;
  CudaLinearOperator& A = *A_ptr;
  CHECK_EQ(A.num_cols(), A.num_rows());
  CHECK_EQ(A.num_cols(), b.num_rows());
  x.resize(A.num_cols());
  r_.resize(A.num_rows());
  z_.resize(A.num_rows());
  LinearSolver::Summary summary;
  summary.termination_type = LinearSolverTerminationType::NO_CONVERGENCE;
  summary.message = "Maximum number of iterations reached.";
  summary.num_iterations = 0;

  if (kDebug) printf("Starting CG solver\n");
  const double norm_b = b.norm();
  if (norm_b == 0.0) {
    x.setZero();
    summary.termination_type = LinearSolverTerminationType::SUCCESS;
    summary.message = "Convergence. |b| = 0.";
    return summary;
  }

  const double tol_r = per_solve_options.r_tolerance * norm_b;

  if (kDebug) printf("r = 0\n");
  // r = 0.
  r_.setZero();
  // r = A * x.
  A.RightMultiply(x, &r_);
  if (kDebug) printf("r = A * x\n");
  // r = b - r
  //   = b - A * x.
  r_.Axpby(1.0, b, -1.0);
  if (kDebug) printf("r = b - A * x\n");

  double norm_r = r_.norm();
  if (options_.min_num_iterations == 0 && norm_r <= tol_r) {
    summary.termination_type = LinearSolverTerminationType::SUCCESS;
    summary.message =
        StringPrintf("Convergence. |r_| = %e <= %e.", norm_r, tol_r);
    if (kDebug) printf("%s\n", summary.message.c_str());
    return summary;
  }

  double rho = 1.0;

  // Initial value of the quadratic model Q = x'Ax - 2 * b'x.
  // tmp = r
  //     = b - A * x.
  tmp_.CopyFrom(r_);
  if (kDebug) printf("tmp = r\n");
  // tmp = b + tmp.
  //     = 2 * b + A * x.
  tmp_.Axpy(1.0, b);
  if (kDebug) printf("tmp = 2 * b + A * x\n");
  // Q0 = x'Ax - 2 * b'x.
  double Q0 = -1.0 * x.dot(tmp_);
  if (kDebug) printf("Q0 = x'Ax - 2 * b'x\n");

  for (summary.num_iterations = 1;; ++summary.num_iterations) {
    if (kDebug) {
      printf("Iteration %3d ||r|| = %20f\n",
             summary.num_iterations,
             norm_r);
    }
    // Apply preconditioner
    if (preconditioner != nullptr) {
      preconditioner->Apply(r_, &z_);
    } else {
      z_.CopyFrom(r_);
    }

    double last_rho = rho;
    rho = r_.dot(z_);
    if (IsZeroOrInfinity(rho)) {
      summary.termination_type = LinearSolverTerminationType::FAILURE;
      summary.message =
          StringPrintf("Numerical failure. rho = r_'z_ = %e.", rho);
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }

    if (summary.num_iterations == 1) {
      p_.CopyFrom(z_);
    } else {
      double beta = rho / last_rho;
      if (IsZeroOrInfinity(beta)) {
        summary.termination_type = LinearSolverTerminationType::FAILURE;
        summary.message = StringPrintf(
            "Numerical failure. beta = rho_n / rho_{n-1} = %e, "
            "rho_n = %e, rho_{n-1} = %e",
            beta,
            rho,
            last_rho);
        if (kDebug) printf("%s\n", summary.message.c_str());
        break;
      }
      // p = z + beta * p.
      p_.Axpby(1.0, z_, beta);
    }

    CudaVector& q = z_;
    q.setZero();
    // q = A * p.
    A.RightMultiply(p_, &q);
    const double pq = p_.dot(q);
    if ((pq <= 0) || std::isinf(pq)) {
      summary.termination_type = LinearSolverTerminationType::NO_CONVERGENCE;
      summary.message = StringPrintf(
          "Matrix is indefinite, no more progress can be made. "
          "p_'q = %e. |p_| = %e, |q| = %e",
          pq,
          p_.norm(),
          q.norm());
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }

    const double alpha = rho / pq;
    if (std::isinf(alpha)) {
      summary.termination_type = LinearSolverTerminationType::FAILURE;
      summary.message = StringPrintf(
          "Numerical failure. alpha = rho / pq = %e, rho = %e, pq = %e.",
          alpha,
          rho,
          pq);
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }

    // x = x + alpha * p.
    x.Axpy(alpha, p_);

    // Ideally we would just use the update r_ = r_ - alpha*q to keep
    // track of the residual vector. However this estimate tends to
    // drift over time due to round off errors. Thus every
    // residual_reset_period iterations, we calculate the residual as
    // r_ = b - Ax. We do not do this every iteration because this
    // requires an additional matrix vector multiply which would
    // double the complexity of the CG algorithm.
    if (summary.num_iterations % options_.residual_reset_period == 0) {
      // r = A * x.
      r_.setZero();
      A.RightMultiply(x, &r_);
      // r = b - r.
      //   = b - A * x.
      r_.Axpby(1.0, b, -1.0);
    } else {
      // r = r - alpha * q.
      r_.Axpy(-alpha, q);
    }

    // Updated quadratic model Q1 = x'Ax - 2 * b' x.
    // tmp = r
    //     = b - A * x.
    tmp_.CopyFrom(r_);
    // tmp = b + tmp.
    //     = 2 * b + A * x.
    tmp_.Axpy(1.0, b);
    // Q1 = x'Ax - 2 * b'x.
    const double Q1 = -1.0 * x.dot(tmp_);

    // For PSD matrices A, let
    //
    //   Q(x) = x'Ax - 2b'x
    //
    // be the cost of the quadratic function defined by A and b. Then,
    // the solver terminates at iteration i if
    //
    //   i * (Q(x_i) - Q(x_i-1)) / Q(x_i) < q_tolerance.
    //
    // This termination criterion is more useful when using CG to
    // solve the Newton step. This particular convergence test comes
    // from Stephen Nash's work on truncated Newton
    // methods. References:
    //
    //   1. Stephen G. Nash & Ariela Sofer, Assessing A Search
    //   Direction Within A Truncated Newton Method, Operation
    //   Research Letters 9(1990) 219-221.
    //
    //   2. Stephen G. Nash, A Survey of Truncated Newton Methods,
    //   Journal of Computational and Applied Mathematics,
    //   124(1-2), 45-59, 2000.
    //
    const double zeta = summary.num_iterations * (Q1 - Q0) / Q1;
    if (zeta < per_solve_options.q_tolerance &&
        summary.num_iterations >= options_.min_num_iterations) {
      summary.termination_type = LinearSolverTerminationType::SUCCESS;
      summary.message =
          StringPrintf("Iteration: %d Convergence: zeta = %e < %e. |r_| = %e",
                       summary.num_iterations,
                       zeta,
                       per_solve_options.q_tolerance,
                       r_.norm());
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }
    Q0 = Q1;

    // Residual based termination.
    norm_r = r_.norm();
    if (norm_r <= tol_r &&
        summary.num_iterations >= options_.min_num_iterations) {
      summary.termination_type = LinearSolverTerminationType::SUCCESS;
      summary.message =
          StringPrintf("Iteration: %d Convergence. |r_| = %e <= %e.",
                       summary.num_iterations,
                       norm_r,
                       tol_r);
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }

    if (summary.num_iterations >= options_.max_num_iterations) {
      if (kDebug) printf("%s\n", summary.message.c_str());
      break;
    }
  }

  return summary;
}

}  // namespace ceres::internal
#endif  // CERES_NO_CUDA