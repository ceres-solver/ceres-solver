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
#include "ceres/numeric_diff_options.h"
#include "ceres/rotation.h"
#include "ceres/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// TODO(sameeragarwal): Once these helpers and matchers converge, it would be
// helpful to expose them as testing utilities which can be used by the user
// when implementing their own manifold objects.

// The tests in this file are in two parts.
//
//  1. Manifold::Plus is doing what is expected. This requires per manifold
//     logic.
//  2. The other methods of the manifold have mathematical properties that
//     make it compatible with Plus, as described in
//     https://arxiv.org/pdf/1107.1119.pdf. These tests are implemented using
//     generic matchers defined below which can all be called by the macro
//     EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y)

constexpr int kNumTrials = 1000;
constexpr double kEpsilon = 1e-9;

// Checks that the invariant Plus(x, 0) == x holds.
MATCHER_P(XPlusZeroIsXAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  Vector actual = Vector::Zero(ambient_size);
  Vector zero = Vector::Zero(tangent_size);
  EXPECT_TRUE(arg.Plus(x.data(), zero.data(), actual.data()));
  const double n = (actual - x).norm();
  const double d = x.norm();
  const double diffnorm = (d == 0.0) ? n : (n / d);
  if (diffnorm > kEpsilon) {
    *result_listener << "\nexpected (x): " << x.transpose()
                     << "\nactual: " << actual.transpose()
                     << "\ndiffnorm: " << diffnorm;
    return false;
  }
  return true;
}

// Checks that the invariant Minus(x, x) == 0 holds.
MATCHER_P(XMinusXIsZeroAt, x, "") {
  const int tangent_size = arg.TangentSize();
  Vector actual = Vector::Zero(tangent_size);
  EXPECT_TRUE(arg.Minus(x.data(), x.data(), actual.data()));
  const double diffnorm = actual.norm();
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose()  //
                     << "\nexpected: 0 0 0"
                     << "\nactual: " << actual.transpose()
                     << "\ndiffnorm: " << diffnorm;
    return false;
  }
  return true;
}

// Helper struct to curry Plus(x, .) so that it can be numerically
// differentiated.
struct PlusFunctor {
  PlusFunctor(const Manifold& manifold, const double* x)
      : manifold(manifold), x(x) {}
  bool operator()(double const* const* parameters, double* x_plus_delta) const {
    return manifold.Plus(x, parameters[0], x_plus_delta);
  }

  const Manifold& manifold;
  const double* x;
};

// Checks that the output of PlusJacobian matches the one obtained by
// numerically evaluating D_2 Plus(x,0).
MATCHER_P(HasCorrectPlusJacobianAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  NumericDiffOptions options;
  options.ridders_relative_initial_step_size = 1e-4;

  DynamicNumericDiffCostFunction<PlusFunctor, RIDDERS> cost_function(
      new PlusFunctor(arg, x.data()), TAKE_OWNERSHIP, options);
  cost_function.AddParameterBlock(tangent_size);
  cost_function.SetNumResiduals(ambient_size);

  Vector zero = Vector::Zero(tangent_size);
  double* parameters[1] = {zero.data()};

  Vector x_plus_zero = Vector::Zero(ambient_size);
  Matrix expected = Matrix::Zero(ambient_size, tangent_size);
  double* jacobians[1] = {expected.data()};

  EXPECT_TRUE(
      cost_function.Evaluate(parameters, x_plus_zero.data(), jacobians));

  Matrix actual = Matrix::Random(ambient_size, tangent_size);
  EXPECT_TRUE(arg.PlusJacobian(x.data(), actual.data()));

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const double diffnorm = (d == 0.0) ? n : n / d;
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: \n"
                     << expected << "\nactual:\n"
                     << actual << "\ndiff:\n"
                     << expected - actual << "\ndiffnorm : " << diffnorm;
    return false;
  }
  return true;
}

// Checks that the invariant Minus(Plus(x, delta), x) == delta holds.
MATCHER_P2(MinusPlusIsIdentityAt, x, delta, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();
  Vector x_plus_delta = Vector::Zero(ambient_size);
  EXPECT_TRUE(arg.Plus(x.data(), delta.data(), x_plus_delta.data()));
  Vector actual = Vector::Zero(tangent_size);
  EXPECT_TRUE(arg.Minus(x_plus_delta.data(), x.data(), actual.data()));

  const double n = (actual - delta).norm();
  const double d = delta.norm();
  const double diffnorm = (d == 0.0) ? n : (n / d);
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose()
                     << "\nexpected: " << delta.transpose()
                     << "\nactual:" << actual.transpose()
                     << "\ndiff:" << (delta - actual).transpose()
                     << "\ndiffnorm: " << diffnorm;
    return false;
  }
  return true;
}

// Checks that the invariant Plus(Minus(y, x), x) == y holds.
MATCHER_P2(PlusMinusIsIdentityAt, x, y, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  Vector y_minus_x = Vector::Zero(tangent_size);
  EXPECT_TRUE(arg.Minus(y.data(), x.data(), y_minus_x.data()));

  Vector actual = Vector::Zero(ambient_size);
  EXPECT_TRUE(arg.Plus(x.data(), y_minus_x.data(), actual.data()));

  const double n = (actual - y).norm();
  const double d = y.norm();
  const double diffnorm = (d == 0.0) ? n : (n / d);
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose()
                     << "\nexpected: " << y.transpose()
                     << "\nactual:" << actual.transpose()
                     << "\ndiff:" << (y - actual).transpose()
                     << "\ndiffnorm: " << diffnorm;
    return false;
  }
  return true;
}

// Helper struct to curry Minus(., x) so that it can be numerically
// differentiated.
struct MinusFunctor {
  MinusFunctor(const Manifold& manifold, const double* x)
      : manifold(manifold), x(x) {}
  bool operator()(double const* const* parameters, double* y_minus_x) const {
    return manifold.Minus(parameters[0], x, y_minus_x);
  }

  const Manifold& manifold;
  const double* x;
};

// Checks that the output of MinusJacobian matches the one obtained by
// numerically evaluating D_1 Minus(x,x).
MATCHER_P(HasCorrectMinusJacobianAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  Vector y = x;
  Vector y_minus_x = Vector::Zero(tangent_size);

  NumericDiffOptions options;
  options.ridders_relative_initial_step_size = 1e-4;
  DynamicNumericDiffCostFunction<MinusFunctor, RIDDERS> cost_function(
      new MinusFunctor(arg, x.data()), TAKE_OWNERSHIP, options);
  cost_function.AddParameterBlock(ambient_size);
  cost_function.SetNumResiduals(tangent_size);

  double* parameters[1] = {y.data()};

  Matrix expected = Matrix::Zero(tangent_size, ambient_size);
  double* jacobians[1] = {expected.data()};

  EXPECT_TRUE(cost_function.Evaluate(parameters, y_minus_x.data(), jacobians));

  Matrix actual = Matrix::Random(tangent_size, ambient_size);
  EXPECT_TRUE(arg.MinusJacobian(x.data(), actual.data()));

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const double diffnorm = (d == 0.0) ? n : (n / d);
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: \n"
                     << expected << "\nactual:\n"
                     << actual << "\ndiff:\n"
                     << expected - actual << "\ndiffnorm: " << diffnorm;
    return false;
  }
  return true;
}

// Checks that D_delta Minus(Plus(x, delta), x) at delta = 0 is an identity
// matrix.
MATCHER_P(MinusPlusJacobianIsIdentityAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  Matrix plus_jacobian(ambient_size, tangent_size);
  EXPECT_TRUE(arg.PlusJacobian(x.data(), plus_jacobian.data()));
  Matrix minus_jacobian(tangent_size, ambient_size);
  EXPECT_TRUE(arg.MinusJacobian(x.data(), minus_jacobian.data()));

  const Matrix actual = minus_jacobian * plus_jacobian;
  const Matrix expected = Matrix::Identity(tangent_size, tangent_size);

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const double diffnorm = n / d;
  if (diffnorm > kEpsilon) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: \n"
                     << expected << "\nactual:\n"
                     << actual << "\ndiff:\n"
                     << expected - actual << "\ndiffnorm: " << diffnorm;

    return false;
  }
  return true;
}

// Verify that the output of RightMultiplyByPlusJacobian is ambient_matrix *
// plus_jacobian.
MATCHER_P(HasCorrectRightMultiplyByPlusJacobianAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  constexpr int kMinNumRows = 0;
  constexpr int kMaxNumRows = 3;
  for (int num_rows = kMinNumRows; num_rows <= kMaxNumRows; ++num_rows) {
    Matrix plus_jacobian = Matrix::Random(ambient_size, tangent_size);
    EXPECT_TRUE(arg.PlusJacobian(x.data(), plus_jacobian.data()));

    Matrix ambient_matrix = Matrix::Random(num_rows, ambient_size);
    Matrix expected = ambient_matrix * plus_jacobian;

    Matrix actual = Matrix::Random(num_rows, tangent_size);
    EXPECT_TRUE(arg.RightMultiplyByPlusJacobian(
        x.data(), num_rows, ambient_matrix.data(), actual.data()));
    const double n = (actual - expected).norm();
    const double d = expected.norm();
    const double diffnorm = (d == 0.0) ? n : (n / d);
    if (diffnorm > kEpsilon) {
      *result_listener << "\nx: " << x.transpose() << "\nambient_matrix : \n"
                       << ambient_matrix << "\nplus_jacobian : \n"
                       << plus_jacobian << "\nexpected: \n"
                       << expected << "\nactual:\n"
                       << actual << "\ndiff:\n"
                       << expected - actual << "\ndiffnorm : " << diffnorm;
      return false;
    }
  }
  return true;
}

#define EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y) \
  Vector zero_tangent = Vector::Zero(manifold.TangentSize());       \
  EXPECT_THAT(manifold, XPlusZeroIsXAt(x));                         \
  EXPECT_THAT(manifold, XMinusXIsZeroAt(x));                        \
  EXPECT_THAT(manifold, MinusPlusIsIdentityAt(x, delta));           \
  EXPECT_THAT(manifold, MinusPlusIsIdentityAt(x, zero_tangent));    \
  EXPECT_THAT(manifold, PlusMinusIsIdentityAt(x, x));               \
  EXPECT_THAT(manifold, PlusMinusIsIdentityAt(x, y));               \
  EXPECT_THAT(manifold, HasCorrectPlusJacobianAt(x));               \
  EXPECT_THAT(manifold, HasCorrectMinusJacobianAt(x));              \
  EXPECT_THAT(manifold, MinusPlusJacobianIsIdentityAt(x));          \
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobianAt(x));

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
    EXPECT_NEAR(
        (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
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
    EXPECT_NEAR(
        (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
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
          manifold_with_ith_parameter_constant, x, delta, y);
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

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, x_plus_delta);
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

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
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
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

TEST(Quaternion, PlusPiBy2) {
  Quaternion manifold;
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
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, x_plus_delta);
  }
}

// Compute the expected value of Quaternion::Plus via functions in rotation.h
// and compares it to the one computed by Quaternion::Plus.
MATCHER_P2(QuaternionPlusIsCorrectAt, x, delta, "") {
  // This multiplication by 2 is needed because AngleAxisToQuaternion uses
  // |delta|/2 as the angle of rotation where as in the implementation of
  // Quaternion for historical reasons we use |delta|.
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
  if (diffnorm > kEpsilon) {
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

TEST(Quaternion, GenericDelta) {
  Quaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    EXPECT_THAT(manifold, QuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

TEST(Quaternion, SmallDelta) {
  Quaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= 1e-6;
    EXPECT_THAT(manifold, QuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

TEST(Quaternion, DeltaJustBelowPi) {
  Quaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= (M_PI - 1e-6);
    EXPECT_THAT(manifold, QuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

// Compute the expected value of EigenQuaternion::Plus using Eigen and compares
// it to the one computed by Quaternion::Plus.
MATCHER_P2(EigenQuaternionPlusIsCorrectAt, x, delta, "") {
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
  if (diffnorm > kEpsilon) {
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

TEST(EigenQuaternion, GenericDelta) {
  EigenQuaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    EXPECT_THAT(manifold, EigenQuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

TEST(EigenQuaternion, SmallDelta) {
  EigenQuaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= 1e-6;
    EXPECT_THAT(manifold, EigenQuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

TEST(EigenQuaternion, DeltaJustBelowPi) {
  EigenQuaternion manifold;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    Vector delta = Vector::Random(3);
    delta.normalize();
    delta *= (M_PI - 1e-6);
    EXPECT_THAT(manifold, EigenQuaternionPlusIsCorrectAt(x, delta));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta, y);
  }
}

}  // namespace internal
}  // namespace ceres
