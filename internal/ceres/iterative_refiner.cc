// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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

#include <string>
#include "ceres/iterative_refiner.h"

#include "Eigen/Core"
#include "ceres/sparse_cholesky.h"
#include "ceres/sparse_matrix.h"

namespace ceres {
namespace internal {

IterativeRefiner::IterativeRefiner(const int num_cols,
                                   const int max_num_iterations)
    : num_cols_(num_cols),
      max_num_iterations_(max_num_iterations),
      residual_(num_cols),
      correction_(num_cols),
      lhs_x_solution_(num_cols) {}

IterativeRefiner::~IterativeRefiner() {}

IterativeRefiner::Summary IterativeRefiner::Refine(
    const SparseMatrix& lhs,
    const double* rhs_ptr,
    SparseCholesky* sparse_cholesky,
    double* solution_ptr) {
  Summary summary;

  ConstVectorRef rhs(rhs_ptr, num_cols_);
  VectorRef solution(solution_ptr, num_cols_);

  summary.lhs_max_norm = ConstVectorRef(lhs.values(), lhs.num_nonzeros())
                             .lpNorm<Eigen::Infinity>();
  summary.rhs_max_norm = rhs.lpNorm<Eigen::Infinity>();
  summary.solution_max_norm = solution.lpNorm<Eigen::Infinity>();

  // residual = rhs - lhs * solution
  lhs_x_solution_.setZero();
  lhs.RightMultiply(solution_ptr, lhs_x_solution_.data());
  residual_ = rhs - lhs_x_solution_;
  summary.residual_max_norm = residual_.lpNorm<Eigen::Infinity>();

  for (summary.num_iterations = 0;
       summary.num_iterations < max_num_iterations_;
       ++summary.num_iterations) {
    // Check the current solution for convergence.
    const double kTolerance = 5e-15;  // From Hogg & Scott.
    // residual_tolerance = (|A| |x| + |b|) * kTolerance;
    const double residual_tolerance =
        (summary.lhs_max_norm * summary.solution_max_norm +
         summary.rhs_max_norm) *
        kTolerance;
    VLOG(3) << "Refinement:"
            << " iter: " << summary.num_iterations
            << " |A|: " << summary.lhs_max_norm
            << " |b|: " << summary.rhs_max_norm
            << " |x|: " << summary.solution_max_norm
            << " |b - Ax|: " << summary.residual_max_norm
            << " tol: " << residual_tolerance;
    // |b - Ax| < (|A| |x| + |b|) * kTolerance;
    if (summary.residual_max_norm < residual_tolerance) {
      summary.converged = true;
      break;
    }

    // Solve for lhs * correction = residual
    correction_.setZero();
    std::string ignored_message;
    sparse_cholesky->Solve(
        residual_.data(), correction_.data(), &ignored_message);
    solution += correction_;
    summary.solution_max_norm = solution.lpNorm<Eigen::Infinity>();

    // residual = rhs - lhs * solution
    lhs_x_solution_.setZero();
    lhs.RightMultiply(solution_ptr, lhs_x_solution_.data());
    residual_ = rhs - lhs_x_solution_;
    summary.residual_max_norm = residual_.lpNorm<Eigen::Infinity>();
  }

  return summary;
};

}  // namespace internal
}  // namespace ceres
