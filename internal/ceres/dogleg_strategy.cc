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

#include "ceres/dogleg_strategy.h"

#include <cmath>
#include "Eigen/Dense"
#include <glog/logging.h>
#include "ceres/array_utils.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_matrix.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/types.h"
#include "ceres/polynomial_solver.h"
#include "ceres/fpclassify.h"

namespace ceres {
namespace internal {
namespace {
const double kMaxMu = 1.0;
const double kMinMu = 1e-8;
}

DoglegStrategy::DoglegStrategy(const TrustRegionStrategy::Options& options)
    : linear_solver_(options.linear_solver),
      radius_(options.initial_radius),
      max_radius_(options.max_radius),
      min_diagonal_(options.lm_min_diagonal),
      max_diagonal_(options.lm_max_diagonal),
      mu_(kMinMu),
      min_mu_(kMinMu),
      max_mu_(kMaxMu),
      mu_increase_factor_(10.0),
      increase_threshold_(0.75),
      decrease_threshold_(0.25),
      dogleg_step_norm_(0.0),
      reuse_(false),
      dogleg_type_(options.dogleg_type) {
  CHECK_NOTNULL(linear_solver_);
  CHECK_GT(min_diagonal_, 0.0);
  CHECK_LT(min_diagonal_, max_diagonal_);
  CHECK_GT(max_radius_, 0.0);
}

// If the reuse_ flag is not set, then the Cauchy point (scaled
// gradient) and the new Gauss-Newton step are computed from
// scratch. The Dogleg step is then computed as interpolation of these
// two vectors.
LinearSolver::Summary DoglegStrategy::ComputeStep(
    const TrustRegionStrategy::PerSolveOptions& per_solve_options,
    SparseMatrix* jacobian,
    const double* residuals,
    double* step) {
  CHECK_NOTNULL(jacobian);
  CHECK_NOTNULL(residuals);
  CHECK_NOTNULL(step);

  const int n = jacobian->num_cols();
  if (reuse_) {
    // Gauss-Newton and gradient vectors are always available, only a
    // new interpolant need to be computed. For the subspace case,
    // the subspace and the two-dimensional model are also still valid.
    //
    // TODO(markus) Refactor into two subclasses?
    switch(dogleg_type_) {
      case TRADITIONAL_DOGLEG:
        ComputeTraditionalDoglegStep(step);
        break;

      case SUBSPACE_DOGLEG:
        ComputeSubspaceDoglegStep(step);
        break;
    }
    LinearSolver::Summary linear_solver_summary;
    linear_solver_summary.num_iterations = 0;
    linear_solver_summary.termination_type = TOLERANCE;
    return linear_solver_summary;
  }

  reuse_ = true;
  // Check that we have the storage needed to hold the various
  // temporary vectors.
  if (diagonal_.rows() != n) {
    diagonal_.resize(n, 1);
    gradient_.resize(n, 1);
    gauss_newton_step_.resize(n, 1);
  }

  // Vector used to form the diagonal matrix that is used to
  // regularize the Gauss-Newton solve.
  jacobian->SquaredColumnNorm(diagonal_.data());
  for (int i = 0; i < n; ++i) {
    diagonal_[i] = min(max(diagonal_[i], min_diagonal_), max_diagonal_);
  }

  gradient_.setZero();
  jacobian->LeftMultiply(residuals, gradient_.data());

  // alpha * gradient is the Cauchy point.
  Vector Jg(jacobian->num_rows());
  Jg.setZero();
  jacobian->RightMultiply(gradient_.data(), Jg.data());
  alpha_ = gradient_.squaredNorm() / Jg.squaredNorm();

  LinearSolver::Summary linear_solver_summary =
      ComputeGaussNewtonStep(jacobian, residuals);

  if (linear_solver_summary.termination_type != FAILURE) {
    switch(dogleg_type_) {
      // Interpolate the Cauchy point and the Gauss-Newton step.
      case TRADITIONAL_DOGLEG:
        ComputeTraditionalDoglegStep(step);
        break;

      // Find the minimum in the subspace defined by the
      // Cauchy point and the (Gauss-)Newton step.
      case SUBSPACE_DOGLEG:
        ComputeSubspaceModel(jacobian);
        ComputeSubspaceDoglegStep(step);
        break;
    }
  }

  return linear_solver_summary;
}

void DoglegStrategy::ComputeTraditionalDoglegStep(double* dogleg) {
  CHECK_EQ(dogleg_type_, TRADITIONAL_DOGLEG);

  VectorRef dogleg_step(dogleg, gradient_.rows());

  // Case 1. The Gauss-Newton step lies inside the trust region, and
  // is therefore the optimal solution to the trust-region problem.
  const double gradient_norm = gradient_.norm();
  const double gauss_newton_norm = gauss_newton_step_.norm();
  if (gauss_newton_norm <= radius_) {
    dogleg_step = gauss_newton_step_;
    dogleg_step_norm_ = gauss_newton_norm;
    VLOG(3) << "GaussNewton step size: " << dogleg_step_norm_
            << " radius: " << radius_;
    return;
  }

  // Case 2. The Cauchy point and the Gauss-Newton steps lie outside
  // the trust region. Rescale the Cauchy point to the trust region
  // and return.
  if  (gradient_norm * alpha_ >= radius_) {
    dogleg_step = (radius_ / gradient_norm) * gradient_;
    dogleg_step_norm_ = radius_;
    VLOG(3) << "Cauchy step size: " << dogleg_step_norm_
            << " radius: " << radius_;
    return;
  }

  // Case 3. The Cauchy point is inside the trust region and the
  // Gauss-Newton step is outside. Compute the line joining the two
  // points and the point on it which intersects the trust region
  // boundary.

  // a = alpha * gradient
  // b = gauss_newton_step
  const double b_dot_a = alpha_ * gradient_.dot(gauss_newton_step_);
  const double a_squared_norm = pow(alpha_ * gradient_norm, 2.0);
  const double b_minus_a_squared_norm =
      a_squared_norm - 2 * b_dot_a + pow(gauss_newton_norm, 2);

  // c = a' (b - a)
  //   = alpha * gradient' gauss_newton_step - alpha^2 |gradient|^2
  const double c = b_dot_a - a_squared_norm;
  const double d = sqrt(c * c + b_minus_a_squared_norm *
                        (pow(radius_, 2.0) - a_squared_norm));

  double beta =
      (c <= 0)
      ? (d - c) /  b_minus_a_squared_norm
      : (radius_ * radius_ - a_squared_norm) / (d + c);
  dogleg_step = (alpha_ * (1.0 - beta)) * gradient_ + beta * gauss_newton_step_;
  dogleg_step_norm_ = dogleg_step.norm();
  VLOG(3) << "Dogleg step size: " << dogleg_step_norm_
          << " radius: " << radius_;
}

void DoglegStrategy::ComputeSubspaceDoglegStep(double* dogleg) {
  CHECK_EQ(dogleg_type_, SUBSPACE_DOGLEG);

  VectorRef dogleg_step(dogleg, gradient_.rows());

  // Find the minimum of the two-dimensional problem
  //    min. 1/2 x' B' H B x + g' B x
  //    s.t. || B x ||^2 <= r^2
  // where r is the trust region radius and B is the matrix with unit columns
  // spanning the subspace defined by the steepest descent and Newton direction.

  const double gauss_newton_norm = gauss_newton_step_.norm();

  // (0,-gauss_newton_norm) is the minimum of the unconstrained problem. If it lies
  // within the trust region, it is also the solution of the constrained
  // problem.
  if (gauss_newton_norm <= radius_) {
    dogleg_step = gauss_newton_step_;
    dogleg_step_norm_ = gauss_newton_norm;
    VLOG(3) << "GaussNewton step size: " << dogleg_step_norm_
            << " radius: " << radius_;
    return;
  }

  // If (0,-gauss_newton_norm) does not lie within the trust region, the
  // optimum lies on the boundary of the trust region. The above problem
  // therefore becomes
  //    min. 1/2 x' B' H B x + g' B x
  //    s.t. || B x ||^2 = r^2
  // Notice the equality in the constraint.
  //
  // This can be solved by forming the Lagrangian, solving for x(lambda)
  // using the gradient of the objective, and putting x(lambda) back into
  // the constraint. This results in a fourth order polynomial in lambda,
  // which can be solved using e.g. the companion matrix.
  // The result is up to four real roots, not all of which correspond to
  // feasible points. The feasible points x(lambda*) have to be tested
  // for optimality.
  EigenTypes<2,2>::Vector minimum(0.0, 0.0);
  FindMinimumOnTrustRegionBoundary(&minimum);

  // Test first order optimality at the minimum
  const EigenTypes<2,2>::Vector grad_minimum = subspace_B_ * minimum + subspace_g_;
  if (-minimum.dot(grad_minimum) < 0.99 * minimum.norm() * grad_minimum.norm()) {
    LOG(WARNING) << "First order optimality seems to be violated in subspace method!";
  }

  // Create the full step from the optimal 2d solution.
  dogleg_step = subspace_basis_ * minimum;
  dogleg_step_norm_ = dogleg_step.norm();
  VLOG(3) << "Dogleg subspace step size: " << dogleg_step_norm_
          << " radius: " << radius_;
}

void DoglegStrategy::FindMinimumOnTrustRegionBoundary(EigenTypes<2,2>::Vector* minimum) {
  CHECK_NOTNULL(minimum);

  const double detB = subspace_B_.determinant();
  const double trB = subspace_B_.trace();
  const double r2 = radius_ * radius_;
  EigenTypes<2,2>::Matrix B_adj;
  B_adj <<  subspace_B_(1,1) , -subspace_B_(0,1),
            -subspace_B_(1,0) ,  subspace_B_(0,0);

  // Build the fourth-order polynomial
  Vector polynomial(5);
  polynomial(4) = r2 * detB * detB - (B_adj * subspace_g_).squaredNorm();
  polynomial(3) = 2.0 * ( subspace_g_.transpose() * B_adj * subspace_g_ - r2 * detB * trB );
  polynomial(2) = r2 * ( trB * trB + 2.0 * detB ) - subspace_g_.squaredNorm();
  polynomial(1) = -2.0 * r2 * trB;
  polynomial(0) = r2;

  // Find the real parts lambda_i of its roots
  Vector lambda;
  if(0 != FindPolynomialRoots(polynomial, &lambda)) {
    // TODO(markus) what to do here?
  }

  // For each root lambda, compute B x(lambda) and check for feasibility.
  double minimum_value = std::numeric_limits<double>::max();
  for (int i=0; i<4; ++i) {
    const double lambda_i = lambda(i);
    const EigenTypes<2,2>::Matrix B_i = subspace_B_ - lambda_i * EigenTypes<2,2>::Matrix::Identity();
    const EigenTypes<2,2>::Vector x_i = -B_i.partialPivLu().solve(subspace_g_);

    // TODO(markus) Should this threshold be configurable?
    // Alternatively, all vectors can safely be rescaled to the trust-region
    // radius without.
    if (abs((x_i.norm() - radius_) / radius_) < 1e-6) {
      const double fi = 0.5 * x_i.transpose() * subspace_B_ * x_i + subspace_g_.dot(x_i);
      if (fi < minimum_value) {
        minimum_value = fi;
        *minimum = x_i;
      }
    }
  }
}

LinearSolver::Summary DoglegStrategy::ComputeGaussNewtonStep(
    SparseMatrix* jacobian,
    const double* residuals) {
  const int n = jacobian->num_cols();
  LinearSolver::Summary linear_solver_summary;
  linear_solver_summary.termination_type = FAILURE;

  // The Jacobian matrix is often quite poorly conditioned. Thus it is
  // necessary to add a diagonal matrix at the bottom to prevent the
  // linear solver from failing.
  //
  // We do this by computing the same diagonal matrix as the one used
  // by Levenberg-Marquardt (other choices are possible), and scaling
  // it by a small constant (independent of the trust region radius).
  //
  // If the solve fails, the multiplier to the diagonal is increased
  // up to max_mu_ by a factor of mu_increase_factor_ every time. If
  // the linear solver is still not successful, the strategy returns
  // with FAILURE.
  //
  // Next time when a new Gauss-Newton step is requested, the
  // multiplier starts out from the last successful solve.
  //
  // When a step is declared successful, the multiplier is decreased
  // by half of mu_increase_factor_.
  while (mu_ < max_mu_) {
    // Dogleg, as far as I (sameeragarwal) understand it, requires a
    // reasonably good estimate of the Gauss-Newton step. This means
    // that we need to solve the normal equations more or less
    // exactly. This is reflected in the values of the tolerances set
    // below.
    //
    // For now, this strategy should only be used with exact
    // factorization based solvers, for which these tolerances are
    // automatically satisfied.
    //
    // The right way to combine inexact solves with trust region
    // methods is to use Stiehaug's method.
    LinearSolver::PerSolveOptions solve_options;
    solve_options.q_tolerance = 0.0;
    solve_options.r_tolerance = 0.0;

    lm_diagonal_ = (diagonal_ * mu_).array().sqrt();
    solve_options.D = lm_diagonal_.data();

    InvalidateArray(n, gauss_newton_step_.data());
    linear_solver_summary = linear_solver_->Solve(jacobian,
                                                  residuals,
                                                  solve_options,
                                                  gauss_newton_step_.data());

    if (linear_solver_summary.termination_type == FAILURE ||
        !IsArrayValid(n, gauss_newton_step_.data())) {
      mu_ *= mu_increase_factor_;
      VLOG(2) << "Increasing mu " << mu_;
      linear_solver_summary.termination_type = FAILURE;
      continue;
    }
    break;
  }

  return linear_solver_summary;
}

void DoglegStrategy::StepAccepted(double step_quality) {
  CHECK_GT(step_quality, 0.0);
  if (step_quality < decrease_threshold_) {
    radius_ *= 0.5;
    return;
  }

  if (step_quality > increase_threshold_) {
    radius_ = max(radius_, 3.0 * dogleg_step_norm_);
  }

  // Reduce the regularization multiplier, in the hope that whatever
  // was causing the rank deficiency has gone away and we can return
  // to doing a pure Gauss-Newton solve.
  mu_ = max(min_mu_, 2.0 * mu_ / mu_increase_factor_ );
  reuse_ = false;
}

void DoglegStrategy::StepRejected(double step_quality) {
  radius_ *= 0.5;
  reuse_ = true;
}

void DoglegStrategy::StepIsInvalid() {
  mu_ *= mu_increase_factor_;
  reuse_ = false;
}

double DoglegStrategy::Radius() const {
  return radius_;
}

void DoglegStrategy::ComputeSubspaceModel(SparseMatrix* jacobian) {
  const double gradient_norm = gradient_.norm();
  const double gauss_newton_norm = gauss_newton_step_.norm();

  // Compute an orthogonal basis for the subspace.
  //
  // TODO(markus) Need to deal with zeros
  // due to short gradients or quasi-parallel vectors.
  subspace_basis_.resize(jacobian->num_cols(), 2);
  subspace_basis_.col(0) = gradient_ / gradient_norm;
  subspace_basis_.col(1) = gauss_newton_step_;
  subspace_basis_.col(1) -= subspace_basis_.col(0).dot(gauss_newton_step_) * subspace_basis_.col(0);
  subspace_basis_.col(1) /= subspace_basis_.col(1).norm();

  // Compute the subspace model.
  //
  // TODO(markus) The gradient_ member seems to be a
  // descent direction (i.e. neg. gradient), which is counter-intuitive.
  subspace_g_ = subspace_basis_.transpose() * -gradient_;

  Vector Jb1(jacobian->num_rows());
  Vector Jb2(jacobian->num_rows());
  Jb1.setZero();
  Jb2.setZero();
  Vector tmp;

  tmp = subspace_basis_.col(0);
  jacobian->RightMultiply(tmp.data(), Jb1.data());
  tmp = subspace_basis_.col(1);
  jacobian->RightMultiply(tmp.data(), Jb2.data());

  subspace_B_(0,0) = Jb1.dot(Jb1);
  subspace_B_(0,1) = subspace_B_(1,0) = Jb1.dot(Jb2);
  subspace_B_(1,1) = Jb2.dot(Jb2);
}

}  // namespace internal
}  // namespace ceres
