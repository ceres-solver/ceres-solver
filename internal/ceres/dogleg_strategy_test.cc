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

#include <limits>
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/dense_qr_solver.h"
#include "ceres/dogleg_strategy.h"
#include "ceres/linear_solver.h"
#include "ceres/trust_region_strategy.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gmock/mock-log.h"
#include "gtest/gtest.h"

using testing::AllOf;
using testing::AnyNumber;
using testing::HasSubstr;
using testing::ScopedMockLog;
using testing::_;

namespace ceres {
namespace internal {

const double kTolerance = 1e-15;
const double kToleranceLoose = 1e-5;
const double kEpsilon = std::numeric_limits<double>::epsilon();

// For the simple test problem
//
//   5 * x.^2 - [10, 20] * x
//
// test if the step does not exceed the trust region.
TEST(DoglegStrategy, TrustRegionObeyedTraditional) {
  // J^T J = 5 * I
  Matrix jacobian(3, 2);
  jacobian.setZero();
  jacobian(0, 0) = 3.0 / std::sqrt(5.0);
  jacobian(2, 0) = 4.0 / std::sqrt(5.0);
  jacobian(1, 1) = 5.0 / std::sqrt(5.0);

  // g = J^T r = (-10, -20)
  double residual[3] = { -2.0 * std::sqrt(5.0), -4.0 * std::sqrt(5.0), -1.0 * std::sqrt(5.0) };
  double x[2];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.dogleg_type = TRADITIONAL_DOGLEG;
  options.initial_radius = 2.0;
  options.max_radius = 2.0;
  // Spherical trust region
  options.lm_min_diagonal = 1.0;
  options.lm_max_diagonal = 1.0;

  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options.linear_solver = linear_solver.get();

  DoglegStrategy strategy(options);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso, &dsm, residual, x);
  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_LE(ConstVectorRef(x, 2).norm(),
      options.initial_radius * (1.0 + 4.0 * kEpsilon));
}

// For the simple test problem
//
//   5 * x.^2 - [10, 20] * x
//
// test if the step does not exceed the trust region.
TEST(DoglegStrategy, TrustRegionObeyedSubspace) {
  // J^T J = 5 * I
  Matrix jacobian(3, 2);
  jacobian.setZero();
  jacobian(0, 0) = 3.0 / std::sqrt(5.0);
  jacobian(2, 0) = 4.0 / std::sqrt(5.0);
  jacobian(1, 1) = 5.0 / std::sqrt(5.0);

  // g = J^T r = (-10, -20)
  double residual[3] = { -2.0 * std::sqrt(5.0), -4.0 * std::sqrt(5.0), -1.0 * std::sqrt(5.0) };
  double x[2];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.dogleg_type = SUBSPACE_DOGLEG;
  options.initial_radius = 2.0;
  options.max_radius = 2.0;
  // Spherical trust region
  options.lm_min_diagonal = 1.0;
  options.lm_max_diagonal = 1.0;

  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options.linear_solver = linear_solver.get();

  DoglegStrategy strategy(options);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso, &dsm, residual, x);
  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_LE(ConstVectorRef(x, 2).norm(),
      options.initial_radius * (1.0 + 4.0 * kEpsilon));
}

// For the simple test problem
//
//   5 * x.^2 - [10, 20] * x
//
// test if the step is in the right direction.
TEST(DoglegStrategy, CorrectStep) {
  // J^T J = 5 * I
  Matrix jacobian(3, 2);
  jacobian.setZero();
  jacobian(0, 0) = 3.0 / std::sqrt(5.0);
  jacobian(2, 0) = 4.0 / std::sqrt(5.0);
  jacobian(1, 1) = 5.0 / std::sqrt(5.0);

  // g = J^T r = (-10, -20)
  double residual[3] = { -2.0 * std::sqrt(5.0), -4.0 * std::sqrt(5.0), -1.0 * std::sqrt(5.0) };
  double x[2];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.dogleg_type = SUBSPACE_DOGLEG;
  options.initial_radius = 2.0;
  options.max_radius = 2.0;
  // Spherical trust region
  options.lm_min_diagonal = 1.0;
  options.lm_max_diagonal = 1.0;

  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options.linear_solver = linear_solver.get();

  DoglegStrategy strategy(options);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso, &dsm, residual, x);
  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_NEAR(x[0], options.initial_radius / std::sqrt(5.0) * 1.0, kToleranceLoose);
  EXPECT_NEAR(x[1], options.initial_radius / std::sqrt(5.0) * 2.0, kToleranceLoose);
}

// Test if the step is correct if gradient and Gauss-Newton point coincide.
TEST(DoglegStrategy, CorrectStepLocalOptimumAlongGradient) {
  // J^T J = R diag(2, 8) R^T (R = 45 deg rotation)
  Matrix jacobian(2, 2);
  jacobian << 1.0, -1.0,
              2.0,  2.0;

  // g = J^T r = (1, -1)
  double residual[2] = { 1.0, 0.0 };
  double x[2];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.dogleg_type = SUBSPACE_DOGLEG;
  options.initial_radius = 0.25;
  options.max_radius = 0.25;
  // Spherical trust region
  options.lm_min_diagonal = 1.0;
  options.lm_max_diagonal = 1.0;

  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options.linear_solver = linear_solver.get();

  DoglegStrategy strategy(options);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso, &dsm, residual, x);
  EXPECT_NE(summary.termination_type, FAILURE);
  EXPECT_NEAR(x[0], -options.initial_radius * std::sqrt(0.5), kToleranceLoose);
  EXPECT_NEAR(x[1], options.initial_radius * std::sqrt(0.5), kToleranceLoose);
}

// Test if the step is correct if gradient and Gauss-Newton point coincide.
TEST(DoglegStrategy, CorrectStepGlobalOptimumAlongGradient) {
  // J^T J = R diag(2,8) R^T (R = 45 deg rotation)
  Matrix jacobian(2, 2);
  jacobian << 1.0, -1.0,
              2.0,  2.0;

  // g = J^T r = (1, -1)
  double residual[2] = { 1.0, 0.0 };
  double x[2];
  DenseSparseMatrix dsm(jacobian);

  TrustRegionStrategy::Options options;
  options.dogleg_type = SUBSPACE_DOGLEG;
  options.initial_radius = 1.0;
  options.max_radius = 1.0;
  // Spherical trust region
  options.lm_min_diagonal = 1.0;
  options.lm_max_diagonal = 1.0;

  scoped_ptr<LinearSolver> linear_solver(
      new DenseQRSolver(LinearSolver::Options()));
  options.linear_solver = linear_solver.get();

  DoglegStrategy strategy(options);
  TrustRegionStrategy::PerSolveOptions pso;

  TrustRegionStrategy::Summary summary = strategy.ComputeStep(pso, &dsm, residual, x);
  EXPECT_NE(summary.termination_type, FAILURE);
  // Test if global optimum found.
  EXPECT_NEAR(x[0], -0.5, kToleranceLoose);
  EXPECT_NEAR(x[1], 0.5, kToleranceLoose);
}

}  // namespace internal
}  // namespace ceres

