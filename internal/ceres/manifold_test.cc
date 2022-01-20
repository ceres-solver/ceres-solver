// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2021 Google Inc. All rights reserved.
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

#include "ceres/manifold.h"

#include <cmath>
#include <limits>
#include <memory>

#include "Eigen/Geometry"
#include "ceres/dynamic_numeric_diff_cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/manifold_test_utils.h"
#include "ceres/numeric_diff_options.h"
#include "ceres/rotation.h"
#include "ceres/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

constexpr int kNumTrials = 1000;
constexpr double kTolerance = 1e-9;

TEST(EuclideanManifold, NormalFunctionTest) {
  EuclideanManifold manifold(3);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 3);

  Vector zero_tangent = Vector::Zero(manifold.TangentSize());
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(manifold.AmbientSize());
    const Vector y = Vector::Random(manifold.AmbientSize());
    Vector delta = Vector::Random(manifold.TangentSize());
    Vector x_plus_delta = Vector::Zero(manifold.AmbientSize());

    manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
    EXPECT_NEAR((x_plus_delta - x - delta).norm() / (x + delta).norm(),
                0.0,
                kTolerance);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(SubsetManifold, EmptyConstantParameters) {
  SubsetManifold manifold(3, {});
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(3);
    const Vector y = Vector::Random(3);
    Vector delta = Vector::Random(3);
    Vector x_plus_delta = Vector::Zero(3);

    manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
    EXPECT_NEAR((x_plus_delta - x - delta).norm() / (x + delta).norm(),
                0.0,
                kTolerance);
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(SubsetManifold, NegativeParameterIndexDeathTest) {
  EXPECT_DEATH_IF_SUPPORTED(SubsetManifold manifold(2, {-1}),
                            "greater than equal to zero");
}

TEST(SubsetManifold, GreaterThanSizeParameterIndexDeathTest) {
  EXPECT_DEATH_IF_SUPPORTED(SubsetManifold manifold(2, {2}),
                            "less than the size");
}

TEST(SubsetManifold, DuplicateParametersDeathTest) {
  EXPECT_DEATH_IF_SUPPORTED(SubsetManifold manifold(2, {1, 1}), "duplicates");
}

TEST(SubsetManifold, NormalFunctionTest) {
  const int kAmbientSize = 4;
  const int kTangentSize = 3;

  for (int i = 0; i < kAmbientSize; ++i) {
    SubsetManifold manifold_with_ith_parameter_constant(kAmbientSize, {i});
    for (int trial = 0; trial < kNumTrials; ++trial) {
      const Vector x = Vector::Random(kAmbientSize);
      Vector y = Vector::Random(kAmbientSize);
      // x and y must have the same i^th coordinate to be on the manifold.
      y[i] = x[i];
      Vector delta = Vector::Random(kTangentSize);
      Vector x_plus_delta = Vector::Zero(kAmbientSize);

      x_plus_delta.setZero();
      manifold_with_ith_parameter_constant.Plus(
          x.data(), delta.data(), x_plus_delta.data());
      int k = 0;
      for (int j = 0; j < kAmbientSize; ++j) {
        if (j == i) {
          EXPECT_EQ(x_plus_delta[j], x[j]);
        } else {
          EXPECT_EQ(x_plus_delta[j], x[j] + delta[k++]);
        }
      }

      EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(
          manifold_with_ith_parameter_constant, x, delta, y, kTolerance);
    }
  }
}

TEST(ProductManifold, Size2) {
  Manifold* manifold1 = new SubsetManifold(5, {2});
  Manifold* manifold2 = new SubsetManifold(3, {0, 1});
  ProductManifold manifold(manifold1, manifold2);

  EXPECT_EQ(manifold.AmbientSize(),
            manifold1->AmbientSize() + manifold2->AmbientSize());
  EXPECT_EQ(manifold.TangentSize(),
            manifold1->TangentSize() + manifold2->TangentSize());
}

TEST(ProductManifold, Size3) {
  Manifold* manifold1 = new SubsetManifold(5, {2});
  Manifold* manifold2 = new SubsetManifold(3, {0, 1});
  Manifold* manifold3 = new SubsetManifold(4, {1});

  ProductManifold manifold(manifold1, manifold2, manifold3);

  EXPECT_EQ(manifold.AmbientSize(),
            manifold1->AmbientSize() + manifold2->AmbientSize() +
                manifold3->AmbientSize());
  EXPECT_EQ(manifold.TangentSize(),
            manifold1->TangentSize() + manifold2->TangentSize() +
                manifold3->TangentSize());
}

TEST(ProductManifold, Size4) {
  Manifold* manifold1 = new SubsetManifold(5, {2});
  Manifold* manifold2 = new SubsetManifold(3, {0, 1});
  Manifold* manifold3 = new SubsetManifold(4, {1});
  Manifold* manifold4 = new SubsetManifold(2, {0});

  ProductManifold manifold(manifold1, manifold2, manifold3, manifold4);

  EXPECT_EQ(manifold.AmbientSize(),
            manifold1->AmbientSize() + manifold2->AmbientSize() +
                manifold3->AmbientSize() + manifold4->AmbientSize());
  EXPECT_EQ(manifold.TangentSize(),
            manifold1->TangentSize() + manifold2->TangentSize() +
                manifold3->TangentSize() + manifold4->TangentSize());
}

TEST(ProductManifold, NormalFunctionTest) {
  Manifold* manifold1 = new SubsetManifold(5, {2});
  Manifold* manifold2 = new SubsetManifold(3, {0, 1});
  Manifold* manifold3 = new SubsetManifold(4, {1});
  Manifold* manifold4 = new SubsetManifold(2, {0});

  ProductManifold manifold(manifold1, manifold2, manifold3, manifold4);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(manifold.AmbientSize());
    Vector delta = Vector::Random(manifold.TangentSize());
    Vector x_plus_delta = Vector::Zero(manifold.AmbientSize());
    Vector x_plus_delta_expected = Vector::Zero(manifold.AmbientSize());

    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

    int ambient_cursor = 0;
    int tangent_cursor = 0;

    EXPECT_TRUE(manifold1->Plus(&x[ambient_cursor],
                                &delta[tangent_cursor],
                                &x_plus_delta_expected[ambient_cursor]));
    ambient_cursor += manifold1->AmbientSize();
    tangent_cursor += manifold1->TangentSize();

    EXPECT_TRUE(manifold2->Plus(&x[ambient_cursor],
                                &delta[tangent_cursor],
                                &x_plus_delta_expected[ambient_cursor]));
    ambient_cursor += manifold2->AmbientSize();
    tangent_cursor += manifold2->TangentSize();

    EXPECT_TRUE(manifold3->Plus(&x[ambient_cursor],
                                &delta[tangent_cursor],
                                &x_plus_delta_expected[ambient_cursor]));
    ambient_cursor += manifold3->AmbientSize();
    tangent_cursor += manifold3->TangentSize();

    EXPECT_TRUE(manifold4->Plus(&x[ambient_cursor],
                                &delta[tangent_cursor],
                                &x_plus_delta_expected[ambient_cursor]));
    ambient_cursor += manifold4->AmbientSize();
    tangent_cursor += manifold4->TangentSize();

    for (int i = 0; i < x.size(); ++i) {
      EXPECT_EQ(x_plus_delta[i], x_plus_delta_expected[i]);
    }

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(
        manifold, x, delta, x_plus_delta, kTolerance);
  }
}

TEST(ProductManifold, ZeroTangentSizeAndEuclidean) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(subset_manifold, euclidean_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(3);
    Vector y = Vector::Random(3);
    y[0] = x[0];
    Vector delta = Vector::Random(2);
    Vector x_plus_delta = Vector::Zero(3);

    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

    EXPECT_EQ(x_plus_delta[0], x[0]);
    EXPECT_EQ(x_plus_delta[1], x[1] + delta[0]);
    EXPECT_EQ(x_plus_delta[2], x[2] + delta[1]);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(ProductManifold, EuclideanAndZeroTangentSize) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(euclidean_manifold, subset_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(3);
    Vector y = Vector::Random(3);
    y[2] = x[2];
    Vector delta = Vector::Random(2);
    Vector x_plus_delta = Vector::Zero(3);

    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));
    EXPECT_EQ(x_plus_delta[0], x[0] + delta[0]);
    EXPECT_EQ(x_plus_delta[1], x[1] + delta[1]);
    EXPECT_EQ(x_plus_delta[2], x[2]);
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(QuaternionManifold, PlusPiBy2) {
  QuaternionManifold manifold;
  Vector x = Vector::Zero(4);
  x[0] = 1.0;

  for (int i = 0; i < 3; ++i) {
    Vector delta = Vector::Zero(3);
    delta[i] = M_PI / 2;
    Vector x_plus_delta = Vector::Zero(4);
    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

    // Expect that the element corresponding to pi/2 is +/- 1. All other
    // elements should be zero.
    for (int j = 0; j < 4; ++j) {
      if (i == (j - 1)) {
        EXPECT_LT(std::abs(x_plus_delta[j]) - 1,
                  std::numeric_limits<double>::epsilon())
            << "\ndelta = " << delta.transpose()
            << "\nx_plus_delta = " << x_plus_delta.transpose()
            << "\n expected the " << j
            << "th element of x_plus_delta to be +/- 1.";
      } else {
        EXPECT_LT(std::abs(x_plus_delta[j]),
                  std::numeric_limits<double>::epsilon())
            << "\ndelta = " << delta.transpose()
            << "\nx_plus_delta = " << x_plus_delta.transpose()
            << "\n expected the " << j << "th element of x_plus_delta to be 0.";
      }
    }
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(
        manifold, x, delta, x_plus_delta, kTolerance);
  }
}

// Compute the expected value of QuaternionManifold::Plus via functions in
// rotation.h and compares it to the one computed by QuaternionManifold::Plus.
MATCHER_P2(QuaternionManifoldPlusIsCorrectAt, x, delta, "") {
  // This multiplication by 2 is needed because AngleAxisToQuaternion uses
  // |delta|/2 as the angle of rotation where as in the implementation of
  // QuaternionManifold for historical reasons we use |delta|.
  const Vector two_delta = delta * 2;
  Vector delta_q(4);
  AngleAxisToQuaternion(two_delta.data(), delta_q.data());

  Vector expected(4);
  QuaternionProduct(delta_q.data(), x.data(), expected.data());
  Vector actual(4);
  EXPECT_TRUE(arg.Plus(x.data(), delta.data(), actual.data()));

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const double diffnorm = n / d;
  if (diffnorm > kTolerance) {
    *result_listener << "\nx: " << x.transpose()
                     << "\ndelta: " << delta.transpose()
                     << "\nexpected: " << expected.transpose()
                     << "\nactual: " << actual.transpose()
                     << "\ndiff: " << (expected - actual).transpose()
                     << "\ndiffnorm : " << diffnorm;
    return false;
  }
  return true;
}

Vector RandomQuaternion() {
  Vector x = Vector::Random(4);
  x.normalize();
  return x;
}

TEST(QuaternionManifold, GenericDelta) {
  QuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    EXPECT_THAT(manifold, QuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(QuaternionManifold, SmallDelta) {
  QuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= 1e-6;
    EXPECT_THAT(manifold, QuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(QuaternionManifold, DeltaJustBelowPi) {
  QuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= (M_PI - 1e-6);
    EXPECT_THAT(manifold, QuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

// Compute the expected value of EigenQuaternionManifold::Plus using Eigen and
// compares it to the one computed by QuaternionManifold::Plus.
MATCHER_P2(EigenQuaternionManifoldPlusIsCorrectAt, x, delta, "") {
  // This multiplication by 2 is needed because AngleAxisToQuaternion uses
  // |delta|/2 as the angle of rotation where as in the implementation of
  // Quaternion for historical reasons we use |delta|.
  const Vector two_delta = delta * 2;
  Vector delta_q(4);
  AngleAxisToQuaternion(two_delta.data(), delta_q.data());
  Eigen::Quaterniond delta_eigen_q(
      delta_q[0], delta_q[1], delta_q[2], delta_q[3]);

  Eigen::Map<const Eigen::Quaterniond> x_eigen_q(x.data());

  Eigen::Quaterniond expected = delta_eigen_q * x_eigen_q;
  double actual[4];
  EXPECT_TRUE(arg.Plus(x.data(), delta.data(), actual));
  Eigen::Map<Eigen::Quaterniond> actual_eigen_q(actual);

  const double n = (actual_eigen_q.coeffs() - expected.coeffs()).norm();
  const double d = expected.norm();
  const double diffnorm = n / d;
  if (diffnorm > kTolerance) {
    *result_listener
        << "\nx: " << x.transpose() << "\ndelta: " << delta.transpose()
        << "\nexpected: " << expected.coeffs().transpose()
        << "\nactual: " << actual_eigen_q.coeffs().transpose() << "\ndiff: "
        << (expected.coeffs() - actual_eigen_q.coeffs()).transpose()
        << "\ndiffnorm : " << diffnorm;
    return false;
  }
  return true;
}

TEST(EigenQuaternionManifold, GenericDelta) {
  EigenQuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    EXPECT_THAT(manifold, EigenQuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(EigenQuaternionManifold, SmallDelta) {
  EigenQuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= 1e-6;
    EXPECT_THAT(manifold, EigenQuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

TEST(EigenQuaternionManifold, DeltaJustBelowPi) {
  EigenQuaternionManifold manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= (M_PI - 1e-6);
    EXPECT_THAT(manifold, EigenQuaternionManifoldPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y, kTolerance);
  }
}

}  // namespace internal
}  // namespace ceres
