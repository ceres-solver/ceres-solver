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

#include "ceres/loss_function.h"

#include <cstddef>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using ::testing::DoubleNear;
using ::testing::Pointwise;

namespace {

// Helper function for testing a LossFunction callback.
//
// Compares the values of rho'(s) and rho''(s) computed by the
// callback with estimates obtained by symmetric finite differencing
// of rho(s).
::testing::AssertionResult LossFunctionIsValid(const LossFunction& loss,
                                               double s) {
  if (!(s > 0)) {
    return ::testing::AssertionFailure() << "s = " << s << " is not positive";
  }

  // Evaluate rho(s), rho'(s) and rho''(s).
  double rho[3];
  loss.Evaluate(s, rho);

  // Use symmetric finite differencing to estimate rho'(s) and
  // rho''(s).
  const double kH = 1e-4;
  // Values at s + kH.
  double fwd[3];
  // Values at s - kH.
  double bwd[3];
  loss.Evaluate(s + kH, fwd);
  loss.Evaluate(s - kH, bwd);

  // First and second derivatives.
  const double fd_1 = (fwd[0] - bwd[0]) / (2 * kH);
  const double fd_2 = (fwd[0] - 2 * rho[0] + bwd[0]) / (kH * kH);

  constexpr double kTolerance = 1e-6;
  ::testing::StringMatchResultListener listener;
  if (!::testing::ExplainMatchResult(
          ::testing::Pointwise(::testing::DoubleNear(kTolerance), {fd_1, fd_2}),
          absl::MakeConstSpan(rho + 1, 2),
          &listener)) {
    return ::testing::AssertionFailure()
           << absl::StrFormat(
                  "the derivatives (rho'(s), rho''(s)) = (%v, %v) at s = %v "
                  "differ from their finite difference estimates (%v, %v) "
                  "by more than %v ",
                  rho[1],
                  rho[2],
                  s,
                  fd_1,
                  fd_2,
                  kTolerance)
           << listener.str();
  }

  return ::testing::AssertionSuccess();
}
}  // namespace

// Try two values of the scaling a = 0.7 and 1.3
// (where scaling makes sense) and of the squared norm
// s = 0.357 and 1.792
//
// Note that for the Huber loss the test exercises both code paths
//  (i.e. both small and large values of s).

TEST(LossFunction, TrivialLoss) {
  EXPECT_TRUE(LossFunctionIsValid(TrivialLoss(), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(TrivialLoss(), 1.792));
  // Check that at s = 0: rho = [0, 1, 0].
  double rho[3];
  TrivialLoss().Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, 0.0}));
}

TEST(LossFunction, HuberLoss) {
  EXPECT_TRUE(LossFunctionIsValid(HuberLoss(0.7), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(HuberLoss(0.7), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(HuberLoss(1.3), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(HuberLoss(1.3), 1.792));
  // Check that at s = 0: rho = [0, 1, 0].
  double rho[3];
  HuberLoss(0.7).Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, 0.0}));
}

TEST(LossFunction, SoftLOneLoss) {
  EXPECT_TRUE(LossFunctionIsValid(SoftLOneLoss(0.7), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(SoftLOneLoss(0.7), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(SoftLOneLoss(1.3), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(SoftLOneLoss(1.3), 1.792));
  // Check that at s = 0: rho = [0, 1, -1 / (2 * a^2)].
  double rho[3];
  SoftLOneLoss(0.7).Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, -0.5 / (0.7 * 0.7)}));
}

TEST(LossFunction, CauchyLoss) {
  EXPECT_TRUE(LossFunctionIsValid(CauchyLoss(0.7), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(CauchyLoss(0.7), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(CauchyLoss(1.3), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(CauchyLoss(1.3), 1.792));
  // Check that at s = 0: rho = [0, 1, -1 / a^2].
  double rho[3];
  CauchyLoss(0.7).Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, -1.0 / (0.7 * 0.7)}));
}

TEST(LossFunction, ArctanLoss) {
  EXPECT_TRUE(LossFunctionIsValid(ArctanLoss(0.7), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(ArctanLoss(0.7), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(ArctanLoss(1.3), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(ArctanLoss(1.3), 1.792));
  // Check that at s = 0: rho = [0, 1, 0].
  double rho[3];
  ArctanLoss(0.7).Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, 0.0}));
}

TEST(LossFunction, TolerantLoss) {
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(0.7, 0.4), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(0.7, 0.4), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(0.7, 0.4), 55.5));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(1.3, 0.1), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(1.3, 0.1), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(1.3, 0.1), 55.5));
  // Check the value at zero is actually zero.
  double rho[3];
  TolerantLoss(0.7, 0.4).Evaluate(0.0, rho);
  EXPECT_NEAR(rho[0], 0.0, 1e-6);
  // Check that loss before and after the approximation threshold are good.
  // A threshold of 36.7 is used by the implementation.
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.6));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.7));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 36.8));
  EXPECT_TRUE(LossFunctionIsValid(TolerantLoss(20.0, 1.0), 20.0 + 1000.0));
}

TEST(LossFunction, TukeyLoss) {
  EXPECT_TRUE(LossFunctionIsValid(TukeyLoss(0.7), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(TukeyLoss(0.7), 1.792));
  EXPECT_TRUE(LossFunctionIsValid(TukeyLoss(1.3), 0.357));
  EXPECT_TRUE(LossFunctionIsValid(TukeyLoss(1.3), 1.792));
  // Check that at s = 0: rho = [0, 1, -2 / a^2].
  double rho[3];
  TukeyLoss(0.7).Evaluate(0.0, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-6), {0.0, 1.0, -2.0 / (0.7 * 0.7)}));
}

TEST(LossFunction, ComposedLoss) {
  {
    HuberLoss f(0.7);
    CauchyLoss g(1.3);
    ComposedLoss c(&f, DO_NOT_TAKE_OWNERSHIP, &g, DO_NOT_TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(c, 0.357));
    EXPECT_TRUE(LossFunctionIsValid(c, 1.792));
  }
  {
    CauchyLoss f(0.7);
    HuberLoss g(1.3);
    ComposedLoss c(&f, DO_NOT_TAKE_OWNERSHIP, &g, DO_NOT_TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(c, 0.357));
    EXPECT_TRUE(LossFunctionIsValid(c, 1.792));
  }
}

TEST(LossFunction, ScaledLoss) {
  // Wrap a few loss functions, and a few scale factors. This can't combine
  // construction with the call to LossFunctionIsValid() because Apple's
  // GCC is unable to eliminate the copy of ScaledLoss, which is not copyable.
  {
    ScaledLoss scaled_loss(nullptr, 6, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 0.323));
  }
  {
    ScaledLoss scaled_loss(new TrivialLoss(), 10, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 0.357));
  }
  {
    ScaledLoss scaled_loss(new HuberLoss(0.7), 0.1, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
  {
    ScaledLoss scaled_loss(new SoftLOneLoss(1.3), 0.1, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
  {
    ScaledLoss scaled_loss(new CauchyLoss(1.3), 10, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
  {
    ScaledLoss scaled_loss(new ArctanLoss(1.3), 10, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
  {
    ScaledLoss scaled_loss(new TolerantLoss(1.3, 0.1), 10, TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
  {
    ScaledLoss scaled_loss(new ComposedLoss(new HuberLoss(0.8),
                                            TAKE_OWNERSHIP,
                                            new TolerantLoss(1.3, 0.5),
                                            TAKE_OWNERSHIP),
                           10,
                           TAKE_OWNERSHIP);
    EXPECT_TRUE(LossFunctionIsValid(scaled_loss, 1.792));
  }
}

TEST(LossFunction, LossFunctionWrapper) {
  // Initialization
  HuberLoss loss_function1(1.0);
  LossFunctionWrapper loss_function_wrapper(new HuberLoss(1.0), TAKE_OWNERSHIP);

  double s = 0.862;
  double rho_gold[3];
  double rho[3];
  loss_function1.Evaluate(s, rho_gold);
  loss_function_wrapper.Evaluate(s, rho);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-12), rho_gold));

  // Resetting
  HuberLoss loss_function2(0.5);
  loss_function_wrapper.Reset(new HuberLoss(0.5), TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function2.Evaluate(s, rho_gold);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-12), rho_gold));

  // Not taking ownership.
  HuberLoss loss_function3(0.3);
  loss_function_wrapper.Reset(&loss_function3, DO_NOT_TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function3.Evaluate(s, rho_gold);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-12), rho_gold));

  // Set to nullptr
  TrivialLoss loss_function4;
  loss_function_wrapper.Reset(nullptr, TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function4.Evaluate(s, rho_gold);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-12), rho_gold));

  // Set to nullptr, not taking ownership
  loss_function_wrapper.Reset(nullptr, DO_NOT_TAKE_OWNERSHIP);
  loss_function_wrapper.Evaluate(s, rho);
  loss_function4.Evaluate(s, rho_gold);
  EXPECT_THAT(rho, Pointwise(DoubleNear(1e-12), rho_gold));
}

}  // namespace internal
}  // namespace ceres
