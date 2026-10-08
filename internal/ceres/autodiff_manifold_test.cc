// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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

#include "ceres/autodiff_manifold.h"

#include <cmath>
#include <limits>

#include "ceres/constants.h"
#include "ceres/internal/eigen.h"
#include "ceres/manifold_test_utils.h"
#include "ceres/quaternion_manifold_test_utils.h"
#include "ceres/rotation.h"
#include "ceres/test_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

namespace {

constexpr int kNumTrials = 1000;
constexpr double kTolerance = 1e-9;

}  // namespace

struct EuclideanFunctor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    for (int i = 0; i < 3; ++i) {
      x_plus_delta[i] = x[i] + delta[i];
    }
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    for (int i = 0; i < 3; ++i) {
      y_minus_x[i] = y[i] - x[i];
    }
    return true;
  }
};

TEST(AutoDiffManifoldTest, EuclideanManifold) {
  AutoDiffManifold<EuclideanFunctor, 3, 3> manifold;
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 3);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(manifold.AmbientSize());
    const Vector y = Vector::Random(manifold.AmbientSize());
    Vector delta = Vector::Random(manifold.TangentSize());
    Vector x_plus_delta = Vector::Zero(manifold.AmbientSize());

    ASSERT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));
    EXPECT_THAT(x_plus_delta, MatrixRelativelyNear(x + delta, kTolerance));

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

struct ScaledFunctor {
  explicit ScaledFunctor(const double s) : s(s) {}

  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    for (int i = 0; i < 3; ++i) {
      x_plus_delta[i] = x[i] + s * delta[i];
    }
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    for (int i = 0; i < 3; ++i) {
      y_minus_x[i] = (y[i] - x[i]) / s;
    }
    return true;
  }

  const double s;
};

TEST(AutoDiffManifoldTest, ScaledManifold) {
  constexpr double kScale = 1.2342;
  AutoDiffManifold<ScaledFunctor, 3, 3> manifold(new ScaledFunctor(kScale));
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 3);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(manifold.AmbientSize());
    const Vector y = Vector::Random(manifold.AmbientSize());
    Vector delta = Vector::Random(manifold.TangentSize());
    Vector x_plus_delta = Vector::Zero(manifold.AmbientSize());

    ASSERT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));
    EXPECT_THAT(x_plus_delta,
                MatrixRelativelyNear(x + delta * kScale, kTolerance));

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

// Templated functor that implements the Plus and Minus operations on the
// Quaternion manifold.
struct QuaternionFunctor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    T q_delta[4];
    if (fpclassify(delta[0]) != FP_ZERO || fpclassify(delta[1]) != FP_ZERO ||
        fpclassify(delta[2]) != FP_ZERO) {
      T norm_delta = hypot(delta[0], delta[1], delta[2]);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use q_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    T minus_x[4] = {x[0], -x[1], -x[2], -x[3]};
    T ambient_y_minus_x[4];
    QuaternionProduct(y, minus_x, ambient_y_minus_x);
    if (fpclassify(ambient_y_minus_x[1]) != FP_ZERO ||
        fpclassify(ambient_y_minus_x[2]) != FP_ZERO ||
        fpclassify(ambient_y_minus_x[3]) != FP_ZERO) {
      T u_norm = hypot(
          ambient_y_minus_x[1], ambient_y_minus_x[2], ambient_y_minus_x[3]);
      T theta = atan2(u_norm, ambient_y_minus_x[0]);
      y_minus_x[0] = theta * ambient_y_minus_x[1] / u_norm;
      y_minus_x[1] = theta * ambient_y_minus_x[2] / u_norm;
      y_minus_x[2] = theta * ambient_y_minus_x[3] / u_norm;
    } else {
      // We do not use [0,0,0] here because even though the value part is
      // a constant, the derivative part is not.
      y_minus_x[0] = ambient_y_minus_x[1];
      y_minus_x[1] = ambient_y_minus_x[2];
      y_minus_x[2] = ambient_y_minus_x[3];
    }
    return true;
  }
};

using AutoDiffQuaternionManifold = AutoDiffManifold<QuaternionFunctor, 4, 3>;

INSTANTIATE_TYPED_TEST_SUITE_P(AutoDiff,
                               QuaternionManifoldPlusTest,
                               ::testing::Types<AutoDiffQuaternionManifold>);

}  // namespace ceres::internal
