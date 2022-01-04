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
#include "ceres/random.h"
#include "ceres/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// TODO(sameeragarwal): Once these helpers and matchers converge, it would be
// helpful to expose them as testing utilities which can be used by the user
// when implementing their own manifold objects.

constexpr int kNumTrials = 100;
constexpr double kEpsilon = 1e-10;

// Checks that the invariant Plus(x, 0) == x holds.
MATCHER_P(XPlusZeroIsXAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  Vector actual = Vector::Zero(ambient_size);
  Vector zero = Vector::Zero(tangent_size);
  arg.Plus(x.data(), zero.data(), actual.data());
  const double n = (actual - x).norm();
  const double d = x.norm();
  const bool result = (d == 0.0) ? (n == 0.0) : (n <= kEpsilon * d);
  if (!result) {
    *result_listener << "\nx: " << x.transpose()
                     << "\nactual: " << actual.transpose();
    return false;
  }
  return true;
}

// Checks that the invariant Minus(x, x) == 0 holds.
MATCHER_P(XMinusXIsZeroAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();
  Vector actual = Vector::Zero(tangent_size);
  arg.Minus(x.data(), x.data(), actual.data());
  const bool result = actual.norm() <= kEpsilon;
  if (!result) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: 0 0 0"
                     << "\nactual: " << actual.transpose();
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

  CHECK(cost_function.Evaluate(parameters, x_plus_zero.data(), jacobians));

  Matrix actual = Matrix::Random(ambient_size, tangent_size);
  arg.PlusJacobian(x.data(), actual.data());

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const bool result = (d == 0.0) ? (n == 0.0) : (n <= kEpsilon * d);
  if (!result) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: \n"
                     << expected << "\nactual:\n"
                     << actual << "\ndiff:\n"
                     << expected - actual << "\n norm: " << n / d;

    return false;
  }

  return true;
}

// Checks that the invariant Minus(Plus(x, delta), x) == delta holds.
MATCHER_P2(HasCorrectMinusAt, x, delta, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();
  Vector x_plus_delta = Vector::Zero(ambient_size);
  arg.Plus(x.data(), delta.data(), x_plus_delta.data());
  Vector actual = Vector::Zero(tangent_size);
  arg.Minus(x_plus_delta.data(), x.data(), actual.data());

  const double n = (actual - delta).norm();
  const double d = delta.norm();
  const bool result = (d == 0.0) ? (n <= kEpsilon) : (n <= kEpsilon * d);
  if (!result) {
    *result_listener << "\nx: " << x.transpose()
                     << "\nexpected: " << delta.transpose()
                     << "\nactual:" << actual.transpose()
                     << "\ndiff:" << (delta - actual).transpose();
  }
  return result;
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

  CHECK(cost_function.Evaluate(parameters, y_minus_x.data(), jacobians));

  Matrix actual = Matrix::Random(tangent_size, ambient_size);
  arg.MinusJacobian(x.data(), actual.data());

  const double n = (actual - expected).norm();
  const double d = expected.norm();
  const bool result = (d == 0.0) ? (n == 0.0) : (n <= kEpsilon * d);
  if (!result) {
    *result_listener << "\nx: " << x.transpose() << "\nexpected: \n"
                     << expected << "\nactual:\n"
                     << actual << "\ndiff:\n"
                     << expected - actual << "\n norm: " << n / d;

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
    arg.PlusJacobian(x.data(), plus_jacobian.data());

    Matrix ambient_matrix = Matrix::Random(num_rows, ambient_size);
    Matrix expected = ambient_matrix * plus_jacobian;

    Matrix actual = Matrix::Random(num_rows, tangent_size);
    arg.RightMultiplyByPlusJacobian(
        x.data(), num_rows, ambient_matrix.data(), actual.data());
    const double n = (actual - expected).norm();
    const double d = expected.norm();
    const bool result = (d == 0.0) ? (n == 0.0) : (n <= kEpsilon * d);
    if (!result) {
      *result_listener << "\nx: " << x.transpose() << "\nambient_matrix : \n"
                       << ambient_matrix << "\nplus_jacobian : \n"
                       << plus_jacobian << "\nexpected: \n"
                       << expected << "\nactual:\n"
                       << actual << "\ndiff:\n"
                       << expected - actual;

      return false;
    }
  }
  return true;
}

#define EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta) \
  Vector zero_tangent = Vector::Zero(manifold.TangentSize());    \
  EXPECT_THAT(manifold, XPlusZeroIsXAt(x));                      \
  EXPECT_THAT(manifold, XMinusXIsZeroAt(x));                     \
  EXPECT_THAT(manifold, HasCorrectMinusAt(x, delta));            \
  EXPECT_THAT(manifold, HasCorrectMinusAt(x, zero_tangent));     \
  EXPECT_THAT(manifold, HasCorrectPlusJacobianAt(x));            \
  EXPECT_THAT(manifold, HasCorrectMinusJacobianAt(x));           \
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobianAt(x));

TEST(EuclideanManifold, NormalFunctionTest) {
  EuclideanManifold manifold(3);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 3);

  Vector zero_tangent = Vector::Zero(manifold.TangentSize());
  for (int trial = 0; trial < kNumTrials; ++trial) {
    const Vector x = Vector::Random(manifold.AmbientSize());
    Vector delta = Vector::Random(manifold.TangentSize());
    Vector x_plus_delta = Vector::Zero(manifold.AmbientSize());

    manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
    EXPECT_NEAR(
        (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
  }
}

TEST(SubsetManifold, EmptyConstantParameters) {
  SubsetManifold manifold(3, {});
  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector x = Vector::Random(3);
    Vector delta = Vector::Random(3);
    Vector x_plus_delta = Vector::Zero(3);

    manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
    EXPECT_NEAR(
        (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
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
    SubsetManifold manifold(kAmbientSize, {i});

    for (int trial = 0; trial < kNumTrials; ++trial) {
      const Vector x = Vector::Random(kAmbientSize);
      Vector delta = Vector::Random(kTangentSize);
      Vector x_plus_delta = Vector::Zero(kAmbientSize);

      x_plus_delta.setZero();
      manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
      int k = 0;
      for (int j = 0; j < kAmbientSize; ++j) {
        if (j == i) {
          EXPECT_EQ(x_plus_delta[j], x[j]);
        } else {
          EXPECT_EQ(x_plus_delta[j], x[j] + delta[k++]);
        }
      }

      EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
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
    Vector x = Vector::Random(manifold.AmbientSize());
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

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
  }
}

TEST(ProductManifold, ZeroTangentSizeAndEuclidean) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(subset_manifold, euclidean_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector x = Vector::Random(3);
    Vector delta = Vector::Random(2);
    Vector x_plus_delta = Vector::Zero(3);

    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

    EXPECT_EQ(x_plus_delta[0], x[0]);
    EXPECT_EQ(x_plus_delta[1], x[1] + delta[0]);
    EXPECT_EQ(x_plus_delta[2], x[2] + delta[1]);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
  }
}

TEST(ProductManifold, EuclideanAndZeroTangentSize) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(euclidean_manifold, subset_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector x = Vector::Random(3);
    Vector delta = Vector::Random(2);
    Vector x_plus_delta = Vector::Zero(3);

    EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

    EXPECT_EQ(x_plus_delta[0], x[0] + delta[0]);
    EXPECT_EQ(x_plus_delta[1], x[1] + delta[1]);
    EXPECT_EQ(x_plus_delta[2], x[2]);

    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold, x, delta);
  }
}

}  // namespace internal
}  // namespace ceres
