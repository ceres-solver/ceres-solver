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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/cubic_interpolation.h"

#include "ceres/jet.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(CubicInterpolator, NeedsAtleastTwoValues) {
  double x[] = {1};
  EXPECT_DEATH_IF_SUPPORTED(CubicInterpolator c(x, 0), "num_values > 1");
  EXPECT_DEATH_IF_SUPPORTED(CubicInterpolator c(x, 1), "num_values > 1");
}

static const double kTolerance = 1e-12;

class CubicInterpolatorTest : public ::testing::Test {
 public:

  void RunPolynomialInterpolationTest(const double a,
                                      const double b,
                                      const double c,
                                      const double d) {
    for (int x = 0; x < kNumSamples; ++x) {
      values_[x] = a * x * x * x + b * x * x + c * x + d;
    }

    CubicInterpolator interpolator(values_, kNumSamples);

    // Check values in the all the cells but the first and the last
    // ones. In these cells, the interpolated function values should
    // match exactly the values of the function being interpolated.
    //
    // On the boundary, we extrapolate the values of the function on
    // the basis of its first derivative, so we do not expect the
    // function values and its derivatives not to match.
    for (int j = 0; j < kNumTestSamples; ++j) {
      const double x = 1.0 + 7.0 / (kNumTestSamples - 1) * j;
      const double expected_f = a * x * x * x + b * x * x + c * x + d;
      const double expected_dfdx = 3.0 * a * x * x + 2.0 * b * x + c;
      double f, dfdx;

      EXPECT_TRUE(interpolator.Evaluate(x, &f, &dfdx));
      EXPECT_NEAR(f, expected_f, kTolerance)
          << "x: " << x
          << " actual f(x): " << expected_f
          << " estimated f(x): " << f;
      EXPECT_NEAR(dfdx, expected_dfdx, kTolerance)
          << "x: " << x
          << " actual df(x)/dx: " << expected_dfdx
          << " estimated df(x)/dx: " << dfdx;
    }
  }

  static const int kNumSamples = 10;
  static const int kNumTestSamples = 100;
  double values_[kNumSamples];
};

TEST_F(CubicInterpolatorTest, ConstantFunction) {
  RunPolynomialInterpolationTest(0.0, 0.0, 0.0, 0.5);
}

TEST_F(CubicInterpolatorTest, LinearFunction) {
  RunPolynomialInterpolationTest(0.0, 0.0, 1.0, 0.5);
}

TEST_F(CubicInterpolatorTest, QuadraticFunction) {
  RunPolynomialInterpolationTest(0.0, 0.4, 1.0, 0.5);
}

TEST(CubicInterpolator, JetEvaluation) {
  const double values[] = {1.0, 2.0, 2.0, 3.0};
  CubicInterpolator interpolator(values, 4);
  double f, dfdx;
  const double x = 2.5;
  EXPECT_TRUE(interpolator.Evaluate(x, &f, &dfdx));

  // Create a Jet with the same scalar part as x, so that the output
  // Jet will be evaluate at x.
  Jet<double, 4> input_jet;
  input_jet.a = x;
  input_jet.v(0) = 1.0;
  input_jet.v(1) = 1.1;
  input_jet.v(2) = 1.2;
  input_jet.v(3) = 1.3;

  Jet<double, 4> output_jet;
  EXPECT_TRUE(interpolator.Evaluate(input_jet, &output_jet));

  // Check that the scalar part of the Jet is f(x).
  EXPECT_EQ(output_jet.a, f);

  // Check that the derivative part of the Jet is dfdx * input_jet.v
  // by the chain rule.
  EXPECT_EQ((output_jet.v - dfdx * input_jet.v).norm(), 0.0);
}

}  // namespace internal
}  // namespace ceres
