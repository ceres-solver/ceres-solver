// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
//         tbennun@gmail.com (Tal Ben-Nun)

#ifdef CERES_HAS_GTEST

#include "ceres/numeric_diff_test_utils.h"

#include <algorithm>
#include <cmath>

#include "absl/strings/str_format.h"
#include "ceres/cost_function.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

using ::testing::ElementsAreArray;
using ::testing::Pointwise;

bool EasyFunctor::operator()(const double* x1,
                             const double* x2,
                             double* residuals) const {
  residuals[0] = residuals[1] = residuals[2] = 0;
  for (int i = 0; i < 5; ++i) {
    residuals[0] += x1[i] * x2[i];
    residuals[2] += x2[i] * x2[i];
  }
  residuals[1] = residuals[0] * residuals[0];
  return true;
}

void EasyFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function, NumericDiffMethodType method) const {
  // The x1[0] is made deliberately small to test the performance near zero.
  // clang-format off
  double x1[] = { 1e-64, 2.0, 3.0, 4.0, 5.0 };
  double x2[] = { 9.0, 9.0, 5.0, 5.0, 1.0 };
  double *parameters[] = { &x1[0], &x2[0] };
  // clang-format on

  double dydx1[15];  // 3 x 5, row major.
  double dydx2[15];  // 3 x 5, row major.
  double* jacobians[2] = {&dydx1[0], &dydx2[0]};

  double residuals[3] = {-1e-100, -2e-100, -3e-100};

  ASSERT_TRUE(
      cost_function.Evaluate(&parameters[0], &residuals[0], &jacobians[0]));

  double expected_residuals[3];
  EasyFunctor functor;
  functor(x1, x2, expected_residuals);
  EXPECT_THAT(residuals, ElementsAreArray(expected_residuals));

  double tolerance = 0.0;
  switch (method) {
    default:
    case CENTRAL:
      tolerance = 3e-9;
      break;

    case FORWARD:
      tolerance = 2e-5;
      break;

    case RIDDERS:
      tolerance = 1e-13;
      break;
  }

  double expected_dydx1[15];
  double expected_dydx2[15];
  for (int i = 0; i < 5; ++i) {
    // clang-format off
    expected_dydx1[5 * 0 + i] = x2[i];                     // y1
    expected_dydx2[5 * 0 + i] = x1[i];
    expected_dydx1[5 * 1 + i] = 2 * x2[i] * residuals[0];  // y2
    expected_dydx2[5 * 1 + i] = 2 * x1[i] * residuals[0];
    expected_dydx1[5 * 2 + i] = 0.0;                       // y3
    expected_dydx2[5 * 2 + i] = 2 * x2[i];
    // clang-format on
  }

  EXPECT_THAT(dydx1, Pointwise(RelativelyNear(tolerance), expected_dydx1));
  EXPECT_THAT(dydx2, Pointwise(RelativelyNear(tolerance), expected_dydx2));
}

bool TranscendentalFunctor::operator()(const double* x1,
                                       const double* x2,
                                       double* residuals) const {
  double x1x2 = 0;
  for (int i = 0; i < 5; ++i) {
    x1x2 += x1[i] * x2[i];
  }
  residuals[0] = sin(x1x2);
  residuals[1] = exp(-x1x2 / 10);
  return true;
}

void TranscendentalFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function, NumericDiffMethodType method) const {
  struct TestParameterBlocks {
    double x1[5];
    double x2[5];
  };

  // clang-format off
  std::vector<TestParameterBlocks> kTests =  {
    { { 1.0, 2.0, 3.0, 4.0, 5.0 },  // No zeros.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 0.0, 2.0, 3.0, 0.0, 5.0 },  // Some zeros x1.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 1.0, 2.0, 3.0, 1.0, 5.0 },  // Some zeros x2.
      { 0.0, 9.0, 0.0, 5.0, 0.0 },
    },
    { { 0.0, 0.0, 0.0, 0.0, 0.0 },  // All zeros x1.
      { 9.0, 9.0, 5.0, 5.0, 1.0 },
    },
    { { 1.0, 2.0, 3.0, 4.0, 5.0 },  // All zeros x2.
      { 0.0, 0.0, 0.0, 0.0, 0.0 },
    },
    { { 0.0, 0.0, 0.0, 0.0, 0.0 },  // All zeros.
      { 0.0, 0.0, 0.0, 0.0, 0.0 },
    },
  };
  // clang-format on

  for (auto& test : kTests) {
    SCOPED_TRACE(absl::StrFormat("x1 = %s, x2 = %s",
                                 ::testing::PrintToString(test.x1),
                                 ::testing::PrintToString(test.x2)));
    double* x1 = &(test.x1[0]);
    double* x2 = &(test.x2[0]);
    double* parameters[] = {x1, x2};

    double dydx1[10];
    double dydx2[10];
    double* jacobians[2] = {&dydx1[0], &dydx2[0]};

    double residuals[2];

    ASSERT_TRUE(
        cost_function.Evaluate(&parameters[0], &residuals[0], &jacobians[0]));
    double x1x2 = 0;
    for (int i = 0; i < 5; ++i) {
      x1x2 += x1[i] * x2[i];
    }

    double tolerance = 0.0;
    switch (method) {
      default:
      case CENTRAL:
        tolerance = 2e-7;
        break;

      case FORWARD:
        tolerance = 2e-5;
        break;

      case RIDDERS:
        tolerance = 3e-12;
        break;
    }

    double expected_dydx1[10];
    double expected_dydx2[10];
    for (int i = 0; i < 5; ++i) {
      // clang-format off
      expected_dydx1[5 * 0 + i] =  x2[i] * cos(x1x2);
      expected_dydx2[5 * 0 + i] =  x1[i] * cos(x1x2);
      expected_dydx1[5 * 1 + i] = -x2[i] * exp(-x1x2 / 10.) / 10.;
      expected_dydx2[5 * 1 + i] = -x1[i] * exp(-x1x2 / 10.) / 10.;
      // clang-format on
    }

    EXPECT_THAT(dydx1, Pointwise(RelativelyNear(tolerance), expected_dydx1));
    EXPECT_THAT(dydx2, Pointwise(RelativelyNear(tolerance), expected_dydx2));
  }
}

bool ExponentialFunctor::operator()(const double* x1, double* residuals) const {
  residuals[0] = exp(x1[0]);
  return true;
}

void ExponentialFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function) const {
  // Evaluating the functor at specific points for testing.
  std::vector<double> kTests = {1.0, 2.0, 3.0, 4.0, 5.0};

  // Minimal tolerance w.r.t. the cost function and the tests.
  const double kTolerance = 2e-14;

  for (double& test : kTests) {
    SCOPED_TRACE(absl::StrFormat("x = %v", test));
    double* parameters[] = {&test};
    double dydx;
    double* jacobians[1] = {&dydx};
    double residual;

    ASSERT_TRUE(
        cost_function.Evaluate(&parameters[0], &residual, &jacobians[0]));

    double expected_result = exp(test);

    // Expect residual to be close to exp(x).
    EXPECT_THAT(residual, RelativelyNear(expected_result, kTolerance));

    // Check evaluated differences. dydx should also be close to exp(x).
    EXPECT_THAT(dydx, RelativelyNear(expected_result, kTolerance));
  }
}

bool RandomizedFunctor::operator()(const double* x1, double* residuals) const {
  double random_value = uniform_distribution_(*prng_);
  residuals[0] = x1[0] * x1[0] + random_value;
  return true;
}

void RandomizedFunctor::ExpectCostFunctionEvaluationIsNearlyCorrect(
    const CostFunction& cost_function) const {
  std::vector<double> kTests = {0.0, 1.0, 3.0, 4.0, 50.0};

  const double kTolerance = 2e-4;

  for (double& test : kTests) {
    SCOPED_TRACE(absl::StrFormat("x = %v", test));
    double* parameters[] = {&test};
    double dydx;
    double* jacobians[1] = {&dydx};
    double residual;

    ASSERT_TRUE(
        cost_function.Evaluate(&parameters[0], &residual, &jacobians[0]));

    // Expect residual to be close to x^2 w.r.t. noise factor.
    EXPECT_THAT(residual, RelativelyNear(test * test, noise_factor_));

    // Check evaluated differences. (dy/dx = ~2x)
    EXPECT_THAT(dydx, RelativelyNear(2 * test, kTolerance));
  }
}

}  // namespace ceres::internal

#endif  // CERES_HAS_GTEST
