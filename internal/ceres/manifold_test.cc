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

constexpr int kNumTrials = 10;
constexpr double kEpsilon = 1e-10;

// Helper struct to curry Plus(x, .) so that it can be numerically
// differentiated.
struct PlusFunctor {
  PlusFunctor(const Manifold& manifold, double* x) : manifold(manifold), x(x) {}
  bool operator()(double const* const* parameters, double* x_plus_delta) const {
    return manifold.Plus(x, parameters[0], x_plus_delta);
  }

  const Manifold& manifold;
  const double* x;
};

// Checks that the output of PlusJacobian matches the one obtained by
// numerically evaluating D_2 Plus(x,0).
MATCHER(HasCorrectPlusJacobian, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector x = Vector::Random(ambient_size);

    NumericDiffOptions options;
    DynamicNumericDiffCostFunction<PlusFunctor, RIDDERS> cost_function(
        new PlusFunctor(arg, x.data()));
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
                       << expected - actual;

      return false;
    }
  }
  return true;
}

// Checks that the invariant Minus(Plus(x, delta), x) == delta holds.
MATCHER_P(HasCorrectMinusAt, x, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();
  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector expected = Vector::Random(tangent_size);

    Vector x_plus_expected = Vector::Zero(ambient_size);
    arg.Plus(x.data(), expected.data(), x_plus_expected.data());

    Vector actual = Vector::Zero(tangent_size);
    arg.Minus(x_plus_expected.data(), x.data(), actual.data());

    const double n = (actual - expected).norm();
    const double d = expected.norm();
    const bool result = (d == 0.0) ? (n == 0.0) : (n <= kEpsilon * d);
    if (!result) {
      *result_listener << "\nx: " << x.transpose()
                       << "\nexpected: " << expected.transpose()
                       << "\nactual:" << actual.transpose()
                       << "\ndiff:" << (expected - actual).transpose();

      return false;
    }
  }
  return true;
}

// Helper struct to curry Minus(., x) so that it can be numerically
// differentiated.
struct MinusFunctor {
  MinusFunctor(const Manifold& manifold, double* x)
      : manifold(manifold), x(x) {}
  bool operator()(double const* const* parameters, double* y_minus_x) const {
    return manifold.Minus(parameters[0], x, y_minus_x);
  }

  const Manifold& manifold;
  const double* x;
};

// Checks that the output of MinusJacobian matches the one obtained by
// numerically evaluating D_1 Minus(x,x).
MATCHER(HasCorrectMinusJacobian, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  for (int trial = 0; trial < kNumTrials; ++trial) {
    Vector y = Vector::Random(ambient_size);
    Vector x = y;
    Vector y_minus_x = Vector::Zero(tangent_size);

    NumericDiffOptions options;
    DynamicNumericDiffCostFunction<MinusFunctor, RIDDERS> cost_function(
        new MinusFunctor(arg, x.data()));
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
                       << expected - actual;

      return false;
    }
  }
  return true;
}

// Verify that the output of RightMultiplyByPlusJacobian is ambient_matrix *
// plus_jacobian.
MATCHER(HasCorrectRightMultiplyByPlusJacobian, "") {
  const int ambient_size = arg.AmbientSize();
  const int tangent_size = arg.TangentSize();

  constexpr int kMinNumRows = 0;
  constexpr int kMaxNumRows = 3;
  for (int num_rows = kMinNumRows; num_rows <= kMaxNumRows; ++num_rows) {
    Vector x = Vector::Random(ambient_size);
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

TEST(EuclideanManifold, NormalFunctionTest) {
  EuclideanManifold manifold(3);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 3);

  Vector x = Vector::Random(3);
  Vector delta = Vector::Random(3);
  Vector x_plus_delta = Vector::Zero(3);

  manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
  EXPECT_NEAR(
      (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);
  EXPECT_THAT(manifold, HasCorrectMinusAt(x));
  EXPECT_THAT(manifold, HasCorrectPlusJacobian());
  EXPECT_THAT(manifold, HasCorrectMinusJacobian());
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
}

TEST(SubsetManifold, EmptyConstantParameters) {
  SubsetManifold manifold(3, {});

  Vector x = Vector::Random(3);
  Vector delta = Vector::Random(3);
  Vector x_plus_delta = Vector::Zero(3);

  manifold.Plus(x.data(), delta.data(), x_plus_delta.data());
  EXPECT_NEAR(
      (x_plus_delta - x - delta).norm() / (x + delta).norm(), 0.0, kEpsilon);
  EXPECT_THAT(manifold, HasCorrectMinusAt(x));
  EXPECT_THAT(manifold, HasCorrectPlusJacobian());
  EXPECT_THAT(manifold, HasCorrectMinusJacobian());
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
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
  Vector x = Vector::Random(kAmbientSize);
  Vector delta = Vector::Random(kTangentSize);
  Vector x_plus_delta = Vector::Zero(kAmbientSize);

  for (int i = 0; i < kAmbientSize; ++i) {
    SubsetManifold manifold(kAmbientSize, {i});
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
    EXPECT_THAT(manifold, HasCorrectMinusAt(x));
    EXPECT_THAT(manifold, HasCorrectPlusJacobian());
    EXPECT_THAT(manifold, HasCorrectMinusJacobian());
    EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
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

  EXPECT_THAT(manifold, HasCorrectMinusAt(x));
  EXPECT_THAT(manifold, HasCorrectPlusJacobian());
  EXPECT_THAT(manifold, HasCorrectMinusJacobian());
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
}

TEST(ProductManifold, ZeroTangentSizeAndEuclidean) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(subset_manifold, euclidean_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  Vector x = Vector::Random(3);
  Vector delta = Vector::Random(2);
  Vector x_plus_delta = Vector::Zero(3);

  EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

  EXPECT_EQ(x_plus_delta[0], x[0]);
  EXPECT_EQ(x_plus_delta[1], x[1] + delta[0]);
  EXPECT_EQ(x_plus_delta[2], x[2] + delta[1]);

  EXPECT_THAT(manifold, HasCorrectMinusAt(x));
  EXPECT_THAT(manifold, HasCorrectPlusJacobian());
  EXPECT_THAT(manifold, HasCorrectMinusJacobian());
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
}

TEST(ProductManifold, EuclideanAndZeroTangentSize) {
  Manifold* subset_manifold = new SubsetManifold(1, {0});
  Manifold* euclidean_manifold = new EuclideanManifold(2);
  ProductManifold manifold(euclidean_manifold, subset_manifold);
  EXPECT_EQ(manifold.AmbientSize(), 3);
  EXPECT_EQ(manifold.TangentSize(), 2);

  Vector x = Vector::Random(3);
  Vector delta = Vector::Random(2);
  Vector x_plus_delta = Vector::Zero(3);

  EXPECT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));

  EXPECT_EQ(x_plus_delta[0], x[0] + delta[0]);
  EXPECT_EQ(x_plus_delta[1], x[1] + delta[1]);
  EXPECT_EQ(x_plus_delta[2], x[2]);

  EXPECT_THAT(manifold, HasCorrectMinusAt(x));
  EXPECT_THAT(manifold, HasCorrectPlusJacobian());
  EXPECT_THAT(manifold, HasCorrectMinusJacobian());
  EXPECT_THAT(manifold, HasCorrectRightMultiplyByPlusJacobian());
}

}  // namespace internal
}  // namespace ceres
