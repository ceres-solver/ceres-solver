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

#include "ceres/test_util.h"

#include <cfenv>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ceres/internal/eigen.h"
#include "ceres/jet.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {
namespace {

using ::testing::DescribeMatcher;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::ExplainMatchResult;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Pointwise;
using ::testing::StringMatchResultListener;

constexpr double kTolerance = 1e-10;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <typename Value, typename Matcher>
std::string Explain(const Matcher& matcher, const Value& value) {
  StringMatchResultListener listener;
  ExplainMatchResult(matcher, value, &listener);
  return listener.str();
}

TEST(FloatEnvironmentScope, RestoresRaisedExceptionsAndRoundingMode) {
  std::feclearexcept(FE_ALL_EXCEPT);
  const int rounding_mode = std::fegetround();
  {
    const FloatEnvironmentScope float_environment_scope;
    std::feraiseexcept(FE_INVALID | FE_DIVBYZERO);
    std::fesetround(FE_TOWARDZERO);
  }
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
  EXPECT_EQ(std::fegetround(), rounding_mode);
}

TEST(RelativeDifference, IsRelativeToTheLargerMagnitude) {
  EXPECT_DOUBLE_EQ(RelativeDifference(4.0, 5.0), 0.2);
  EXPECT_DOUBLE_EQ(RelativeDifference(5.0, 4.0), 0.2);
  EXPECT_DOUBLE_EQ(RelativeDifference(-4.0, -5.0), 0.2);
}

TEST(RelativeDifference, IsAbsoluteIfEitherValueIsZero) {
  EXPECT_DOUBLE_EQ(RelativeDifference(0.0, 1e-3), 1e-3);
  EXPECT_DOUBLE_EQ(RelativeDifference(-1e-3, 0.0), 1e-3);
}

TEST(RelativeDifference, IsZeroForEqualNonFiniteValues) {
  EXPECT_EQ(RelativeDifference(kInfinity, kInfinity), 0.0);
  EXPECT_EQ(RelativeDifference(-kInfinity, -kInfinity), 0.0);
  EXPECT_EQ(RelativeDifference(kNaN, kNaN), 0.0);
}

TEST(RelativeDifference, IsInfiniteForDifferentInfinities) {
  EXPECT_EQ(RelativeDifference(kInfinity, -kInfinity), kInfinity);
  EXPECT_EQ(RelativeDifference(kInfinity, 1.0), kInfinity);
  EXPECT_EQ(RelativeDifference(1.0, -kInfinity), kInfinity);
}

TEST(RelativeDifference, IsNaNIfExactlyOneValueIsNaN) {
  EXPECT_THAT(RelativeDifference(kNaN, 1.0), testing::IsNan());
  EXPECT_THAT(RelativeDifference(1.0, kNaN), testing::IsNan());
}

TEST(RelativeDifference, DoesNotRaiseFloatingPointExceptions) {
  std::feclearexcept(FE_ALL_EXCEPT);
  RelativeDifference(kInfinity, -kInfinity);
  RelativeDifference(kInfinity, 1.0);
  RelativeDifference(kNaN, 1.0);
  RelativeDifference(std::numeric_limits<double>::max(),
                     -std::numeric_limits<double>::max());
  EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
}

TEST(RelativelyNear, MatchesValuesWithinTheTolerance) {
  EXPECT_THAT(1.0 + 0.5 * kTolerance, RelativelyNear(1.0, kTolerance));
  EXPECT_THAT(1.0 + 2.0 * kTolerance, Not(RelativelyNear(1.0, kTolerance)));
}

TEST(RelativelyNear, MatchesEqualNonFiniteValues) {
  EXPECT_THAT(kInfinity, RelativelyNear(kInfinity, kTolerance));
  EXPECT_THAT(kNaN, RelativelyNear(kNaN, kTolerance));
  EXPECT_THAT(-kInfinity, Not(RelativelyNear(kInfinity, kTolerance)));
  EXPECT_THAT(kNaN, Not(RelativelyNear(1.0, kTolerance)));
}

TEST(RelativelyNear, DescribesTheExpectedValueAndTheTolerance) {
  const auto matcher = RelativelyNear(4.0, 0.5);
  EXPECT_EQ(DescribeMatcher<double>(matcher),
            "is within a relative difference of 0.5 of 4");
  EXPECT_EQ(DescribeMatcher<double>(matcher, /*negation=*/true),
            "isn't within a relative difference of 0.5 of 4");
}

TEST(RelativelyNear, ExplainsTheRelativeDifference) {
  EXPECT_EQ(Explain(RelativelyNear(4.0, 0.1), 5.0),
            "which is a relative difference of 0.2");
}

TEST(RelativelyNear, ComparesPairsPointwise) {
  const std::vector<double> expected{1.0, -2.0, kInfinity};
  const std::vector<double> close{1.0 + 0.5 * kTolerance, -2.0, kInfinity};
  const std::vector<double> far{1.0, -2.0 + 1e-3, kInfinity};
  EXPECT_THAT(close, Pointwise(RelativelyNear(kTolerance), expected));
  EXPECT_THAT(far, Not(Pointwise(RelativelyNear(kTolerance), expected)));
}

TEST(RelativelyNear, ExplainsThePointwiseMismatch) {
  const auto matcher = Pointwise(RelativelyNear(0.1), std::vector{4.0});
  EXPECT_THAT(Explain(matcher, std::vector{5.0}),
              HasSubstr("which is a relative difference of 0.2"));
}

TEST(MatrixNear, MatchesMatricesWithinTheTolerance) {
  const Matrix expected = Matrix::Identity(2, 3);
  Matrix actual = expected;
  actual(1, 2) = 0.5 * kTolerance;
  EXPECT_THAT(actual, MatrixNear(expected, kTolerance));
  actual(1, 2) = 2.0 * kTolerance;
  EXPECT_THAT(actual, Not(MatrixNear(expected, kTolerance)));
}

TEST(MatrixNear, AcceptsEigenExpressions) {
  const Vector x = Vector::LinSpaced(3, 1.0, 3.0);
  EXPECT_THAT(2.0 * x, MatrixNear(x + x, kTolerance));
  EXPECT_THAT(x.head(2), MatrixNear(Eigen::Vector2d(1.0, 2.0), kTolerance));
}

TEST(MatrixNear, RejectsDifferentDimensions) {
  const Matrix expected = Matrix::Zero(2, 3);
  const Matrix actual = Matrix::Zero(3, 2);
  EXPECT_THAT(actual, Not(MatrixNear(expected, kTolerance)));
  EXPECT_EQ(Explain(MatrixNear(expected, kTolerance), actual),
            "which has 3 rows and 2 columns instead of 2 rows and 3 columns");
}

TEST(MatrixNear, ExplainsTheDistanceAndTheLargestDifference) {
  const Eigen::Vector2d expected(0.0, 0.0);
  const Eigen::Vector2d actual(3.0, -4.0);
  EXPECT_EQ(Explain(MatrixNear(expected, kTolerance), actual),
            "which is at a distance of 5 with the largest difference of -4 "
            "at (1, 0)");
}

TEST(MatrixNear, ExplainsEqualMatrices) {
  const Eigen::Vector2d expected(1.0, 2.0);
  EXPECT_EQ(Explain(MatrixNear(expected, kTolerance), expected),
            "which is at a distance of 0");
}

TEST(MatrixNear, DescribesLargeMatricesByTheirDimensions) {
  const auto matcher = MatrixNear(Matrix::Zero(20, 30), 0.5);
  EXPECT_EQ(DescribeMatcher<Matrix>(matcher),
            "is within a distance of 0.5 of the 20x30 expected matrix");
}

TEST(MatrixNear, DescribesTheToleranceAndTheExpectedMatrix) {
  const auto matcher = MatrixNear(Eigen::Vector2d(1.0, 2.0), 0.5);
  EXPECT_EQ(DescribeMatcher<Vector>(matcher),
            "is within a distance of 0.5 of\n1\n2");
  EXPECT_EQ(DescribeMatcher<Vector>(matcher, /*negation=*/true),
            "isn't within a distance of 0.5 of\n1\n2");
}

TEST(MatrixRelativelyNear, ScalesTheToleranceByTheLargerNorm) {
  const Eigen::Vector2d expected(300.0, 400.0);
  EXPECT_THAT(Eigen::Vector2d(300.0, 404.0),
              MatrixRelativelyNear(expected, 1e-2));
  EXPECT_THAT(Eigen::Vector2d(300.0, 406.0),
              Not(MatrixRelativelyNear(expected, 1e-2)));
}

TEST(MatrixRelativelyNear, IsSymmetric) {
  const Eigen::Vector2d smaller(0.0, 1.0);
  const Eigen::Vector2d larger(0.0, 2.0);
  EXPECT_THAT(larger, MatrixRelativelyNear(smaller, 0.5));
  EXPECT_THAT(smaller, MatrixRelativelyNear(larger, 0.5));
}

TEST(MatrixRelativelyNear, ExplainsTheRelativeDistance) {
  const Eigen::Vector2d expected(300.0, 395.0);
  const Eigen::Vector2d actual(300.0, 400.0);
  EXPECT_EQ(Explain(MatrixRelativelyNear(expected, 1e-3), actual),
            "which is at a relative distance of 0.01 with the largest "
            "difference of 5 at (1, 0)");
}

TEST(MatrixRelativelyNear, MatchesZeroMatrices) {
  const Matrix zero = Matrix::Zero(2, 3);
  EXPECT_THAT(zero, MatrixRelativelyNear(zero, kTolerance));
  EXPECT_THAT(Matrix::Ones(2, 3), Not(MatrixRelativelyNear(zero, kTolerance)));
}

TEST(MatrixRelativelyNear, IsAbsoluteIfEitherMatrixIsZero) {
  const Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  const Eigen::Vector2d roundoff(0.5 * kTolerance, 0.0);
  EXPECT_THAT(roundoff, MatrixRelativelyNear(zero, kTolerance));
  EXPECT_THAT(zero, MatrixRelativelyNear(roundoff, kTolerance));
  EXPECT_EQ(Explain(MatrixRelativelyNear(zero, kTolerance),
                    Eigen::Vector2d(0.0, 0.5)),
            "which is at a relative distance of 0.5 with the largest "
            "difference of 0.5 at (1, 0)");
}

TEST(MatrixRelativelyNear, DescribesTheToleranceAndTheExpectedMatrix) {
  const auto matcher = MatrixRelativelyNear(Eigen::Vector2d(1.0, 2.0), 0.5);
  EXPECT_EQ(DescribeMatcher<Vector>(matcher),
            "is within a relative distance of 0.5 of\n1\n2");
}

TEST(MatrixPointwise, MatchesCoefficientsSatisfyingTheTupleMatcher) {
  const Matrix expected = Matrix::Identity(2, 3);
  Matrix actual = expected;
  actual(1, 2) = 0.5 * kTolerance;
  EXPECT_THAT(actual, MatrixPointwise(DoubleNear(kTolerance), expected));
  actual(1, 2) = 2.0 * kTolerance;
  EXPECT_THAT(actual, Not(MatrixPointwise(DoubleNear(kTolerance), expected)));
}

TEST(MatrixPointwise, AcceptsEigenExpressionsAndTupleMatchers) {
  const Vector x = Vector::LinSpaced(3, 1.0, 3.0);
  EXPECT_THAT(2.0 * x, MatrixPointwise(Eq(), x + x));
  EXPECT_THAT(x.head(2), MatrixPointwise(Eq(), Eigen::Vector2d(1.0, 2.0)));
  EXPECT_THAT(Eigen::Vector2i(1, 2),
              MatrixPointwise(Eq(), Eigen::Vector2d(1.0, 2.0)));
  EXPECT_THAT(1.1 * x, MatrixPointwise(RelativelyNear(0.1), x));
  EXPECT_THAT(1.2 * x, Not(MatrixPointwise(RelativelyNear(0.1), x)));
}

TEST(MatrixPointwise, RejectsDifferentDimensions) {
  const Matrix expected = Matrix::Zero(2, 3);
  const Matrix actual = Matrix::Zero(3, 2);
  EXPECT_THAT(actual, Not(MatrixPointwise(Eq(), expected)));
  EXPECT_EQ(Explain(MatrixPointwise(Eq(), expected), actual),
            "which has 3 rows and 2 columns instead of 2 rows and 3 columns");
}

TEST(MatrixPointwise, ExplainsTheFirstMismatchByRowAndColumn) {
  Matrix expected(2, 2);
  expected << 1.0, 2.0, 4.0, 8.0;
  Matrix actual = expected;
  actual(1, 0) = 5.0;
  actual(1, 1) = 10.0;
  EXPECT_EQ(Explain(MatrixPointwise(RelativelyNear(0.1), expected), actual),
            "where the value pair (5, 4) at (1, 0) doesn't match, which is a "
            "relative difference of 0.2");
}

TEST(MatrixPointwise, DescribesTheTupleMatcherAndTheExpectedMatrix) {
  const auto matcher =
      MatrixPointwise(RelativelyNear(0.5), Eigen::Vector2d(1.0, 2.0));
  EXPECT_EQ(DescribeMatcher<Vector>(matcher),
            "is a 2x1 matrix whose coefficients and the corresponding "
            "coefficients of\n1\n2\nare within a relative difference of 0.5");
  EXPECT_EQ(DescribeMatcher<Vector>(matcher, /*negation=*/true),
            "isn't a 2x1 matrix or has a coefficient that and the "
            "corresponding coefficient of\n1\n2\naren't within a relative "
            "difference of 0.5");
}

TEST(MatrixPointwise, DescribesLargeMatricesByTheirDimensions) {
  const auto matcher = MatrixPointwise(Eq(), Matrix::Zero(20, 30));
  EXPECT_EQ(DescribeMatcher<Matrix>(matcher),
            "is a 20x30 matrix whose coefficients and the corresponding "
            "coefficients of the expected matrix are an equal pair");
}

TEST(MatrixEq, MatchesEqualMatrices) {
  const Vector x = Vector::LinSpaced(3, 1.0, 3.0);
  EXPECT_THAT(2.0 * x, MatrixEq(x + x));
  EXPECT_THAT(x, Not(MatrixEq(x + Vector::Unit(3, 1))));
  EXPECT_THAT(x.head(2), Not(MatrixEq(x)));
}

TEST(MatrixEq, ExplainsTheFirstMismatchByRowAndColumn) {
  EXPECT_EQ(
      Explain(MatrixEq(Eigen::Vector2d(1.0, 2.0)), Eigen::Vector2d(1.0, 3.0)),
      "where the value pair (3, 2) at (1, 0) doesn't match");
}

using J2 = Jet<double, 2>;
using J3 = Jet<double, 3>;

TEST(JetIsClose, MatchesJetsWhosePartsAreRelativelyNear) {
  const J2 expected(2.0, Eigen::Vector2d(4.0, 0.0));
  EXPECT_THAT(J2(2.0 * (1.0 + 0.5 * kTolerance), Eigen::Vector2d(4.0, 0.0)),
              JetIsClose(expected, kTolerance));
  EXPECT_THAT(J2(2.0 * (1.0 + 2.0 * kTolerance), Eigen::Vector2d(4.0, 0.0)),
              Not(JetIsClose(expected, kTolerance)));
  EXPECT_THAT(J2(2.0, Eigen::Vector2d(4.0, 2.0 * kTolerance)),
              Not(JetIsClose(expected, kTolerance)));
}

TEST(JetIsClose, AcceptsJetsOfAnyDimension) {
  const J3 expected(1.0, Eigen::Vector3d(1.0, 2.0, 3.0));
  EXPECT_THAT(expected, JetIsClose(expected, 0.0));
  EXPECT_THAT(J3(1.0, Eigen::Vector3d(1.0, 2.0, 4.0)),
              Not(JetIsClose(expected, kTolerance)));
}

TEST(JetIsClose, ExplainsTheMismatchingDualPartCoefficient) {
  const J2 expected(1.0, Eigen::Vector2d(4.0, 8.0));
  const J2 actual(1.0, Eigen::Vector2d(5.0, 8.0));
  const std::string explanation =
      Explain(JetIsClose(expected, kTolerance), actual);
  EXPECT_THAT(explanation, HasSubstr("whose field `v`"));
  EXPECT_THAT(explanation,
              HasSubstr("where the value pair (5, 4) at (0, 0) doesn't match, "
                        "which is a relative difference of 0.2"));
}

TEST(JetIsClose, ComparesPairsPointwise) {
  const std::vector<J2> expected{J2(1.0, 0), J2(2.0, 1)};
  EXPECT_THAT(expected, Pointwise(JetIsClose(kTolerance), expected));
  const std::vector<J2> far{J2(1.0, 0), J2(2.0, 0)};
  const auto matcher = Pointwise(JetIsClose(kTolerance), expected);
  EXPECT_THAT(far, Not(matcher));
  EXPECT_THAT(Explain(matcher, far),
              HasSubstr("which is a relative difference of 1"));
}

TEST(kHalfSqrt2, IsADoubleConstantWithAVariableTemplate) {
  static_assert(std::is_same_v<decltype(kHalfSqrt2), const double>);
  static_assert(std::is_same_v<decltype(kHalfSqrt2_v<float>), const float>);
  static_assert(kHalfSqrt2 == kHalfSqrt2_v<double>);
  EXPECT_DOUBLE_EQ(2.0 * kHalfSqrt2 * kHalfSqrt2, 1.0);
}

}  // namespace
}  // namespace ceres::internal
