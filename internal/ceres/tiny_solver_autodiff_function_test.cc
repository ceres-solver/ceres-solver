
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)

#include "ceres/tiny_solver_autodiff_function.h"

#include <algorithm>
#include <cmath>

#include "ceres/tiny_solver.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

namespace ceres {

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;

// Convex cost function with zero cost at (1, 2, 3).
struct AutoDiffTestFunctor {
  template<typename T>
  bool operator()(const T* const parameters, T* residuals) const {
    // Shift the parameters so the solution is not at the origin, to prevent
    // accidentally showing "PASS".
    const T& a = parameters[0] - T(1.0);
    const T& b = parameters[1] - T(2.0);
    const T& c = parameters[2] - T(3.0);
    residuals[0] = 1.*a*a + 2.*b*b + 4.*c*c;
    residuals[1] = 5.*a*a + 1.*b*b + 2.*c*c;
    return true;
  }
};

TEST(TinySolverAutoDiffFunction, FullSolveWithAutoDiff) {
  typedef TinySolverAutoDiffFunction<AutoDiffTestFunctor, 2, 3>
      AutoDiffTestFunction;
  AutoDiffTestFunctor autodiff_test_functor;
  AutoDiffTestFunction f(autodiff_test_functor);

  // Pick a starting point away from the minimum.
  Vec3 x(2., -3., 6);
  Vec2 residuals;
  f(x.data(), residuals.data(), NULL);
  EXPECT_NEAR(87., residuals(0), 1e-10);
  EXPECT_NEAR(48., residuals(1), 1e-10);

  TinySolver<AutoDiffTestFunction> solver;
  solver.Solve(f, &x);

  f(x.data(), residuals.data(), NULL);
  EXPECT_NEAR(0.0, residuals.norm(), 1e-10);
  EXPECT_NEAR(1.0, x(0), 1e-5);
  EXPECT_NEAR(2.0, x(1), 1e-5);
  EXPECT_NEAR(3.0, x(2), 1e-5);
}

struct SimpleAutoDiffTestFunctor {
  template<typename T>
  bool operator()(const T* const parameters, T* residuals) const {
    // Shift the parameters so the solution is not at the origin, to prevent
    // accidentally showing "PASS".
    const T& a = parameters[0] - T(1.0);
    const T& b = parameters[1] - T(2.0);
    const T& c = parameters[2] - T(3.0);
    residuals[0] = 2.*a + 0.*b + 1.*c;
    residuals[1] = 0.*a + 4.*b + 6.*c;
    return true;
  }
};

TEST(TinySolverAutoDiffFunction, SimpleFunction) {
  typedef TinySolverAutoDiffFunction<SimpleAutoDiffTestFunctor, 2, 3>
      AutoDiffTestFunction;
  SimpleAutoDiffTestFunctor autodiff_test_functor;
  AutoDiffTestFunction f(autodiff_test_functor);

  // Cost-only evaluation.
  Vec3 x(2.0, 1.0, 4.0);
  Vec2 residuals;

  // Check the case with cost-only evaluation.
  residuals.setZero();
  residuals.setConstant(-10);
  EXPECT_TRUE(f(&x(0), &residuals(0), NULL));
  EXPECT_NEAR(3.0, residuals(0), 1e-5);
  EXPECT_NEAR(2.0, residuals(1), 1e-5);

  // Check the case with cost and Jacobian evaluation.
  residuals.setConstant(-10);
  Eigen::Matrix<double, 2, 3> jacobian;
  EXPECT_TRUE(f(&x(0), &residuals(0), &jacobian(0, 0)));

  // Check cost.
  EXPECT_NEAR(3.0, residuals(0), 1e-5);
  EXPECT_NEAR(2.0, residuals(1), 1e-5);

  // Check Jacobian Row 1.
  EXPECT_NEAR(2.0, jacobian(0, 0), 1e-10);
  EXPECT_NEAR(0.0, jacobian(0, 1), 1e-10);
  EXPECT_NEAR(1.0, jacobian(0, 2), 1e-10);

  // Check Jacobian row 2.
  EXPECT_NEAR(0.0, jacobian(1, 0), 1e-10);
  EXPECT_NEAR(4.0, jacobian(1, 1), 1e-10);
  EXPECT_NEAR(6.0, jacobian(1, 2), 1e-10);
}
}  // namespace tinysolver
