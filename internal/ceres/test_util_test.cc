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

#include <algorithm>
#include <cfenv>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "ceres/block_structure.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/triplet_sparse_matrix.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {
namespace {

using ::testing::DescribeMatcher;
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

TEST(MatrixRelativelyNear, ScalesTheToleranceByTheExpectedNorm) {
  const Eigen::Vector2d expected(300.0, 400.0);
  EXPECT_THAT(Eigen::Vector2d(300.0, 404.0),
              MatrixRelativelyNear(expected, 1e-2));
  EXPECT_THAT(Eigen::Vector2d(300.0, 406.0),
              Not(MatrixRelativelyNear(expected, 1e-2)));
}

TEST(MatrixRelativelyNear, ExplainsTheRelativeDistance) {
  const Eigen::Vector2d expected(300.0, 400.0);
  const Eigen::Vector2d actual(300.0, 405.0);
  EXPECT_EQ(Explain(MatrixRelativelyNear(expected, 1e-3), actual),
            "which is at a relative distance of 0.01 with the largest "
            "difference of 5 at (1, 0)");
}

TEST(MatrixRelativelyNear, MatchesZeroMatrices) {
  const Matrix zero = Matrix::Zero(2, 3);
  EXPECT_THAT(zero, MatrixRelativelyNear(zero, kTolerance));
  EXPECT_THAT(Matrix::Ones(2, 3), Not(MatrixRelativelyNear(zero, kTolerance)));
}

TEST(MatrixRelativelyNear, DescribesTheToleranceAndTheExpectedMatrix) {
  const auto matcher = MatrixRelativelyNear(Eigen::Vector2d(1.0, 2.0), 0.5);
  EXPECT_EQ(DescribeMatcher<Vector>(matcher),
            "is within a relative distance of 0.5 of\n1\n2");
}

TEST(RightMultiplyByIdentity, ReconstructsTheDenseMatrix) {
  TripletSparseMatrix m(2, 3, 2);
  m.mutable_rows()[0] = 0;
  m.mutable_cols()[0] = 2;
  m.mutable_values()[0] = 4.0;
  m.mutable_rows()[1] = 1;
  m.mutable_cols()[1] = 0;
  m.mutable_values()[1] = 5.0;
  m.set_num_nonzeros(2);

  Matrix expected(2, 3);
  expected << 0.0, 0.0, 4.0, 5.0, 0.0, 0.0;
  EXPECT_THAT(RightMultiplyByIdentity(m), MatrixNear(expected, 0.0));
}

// Creates the symmetric positive definite matrix
//
//   [ 4 2 ]
//   [ 2 3 ]
//
// storing only the given triangular part.
std::unique_ptr<CompressedRowSparseMatrix> CreateSymmetricMatrix(
    CompressedRowSparseMatrix::StorageType storage_type) {
  auto m = std::make_unique<CompressedRowSparseMatrix>(2, 2, 3);
  m->set_storage_type(storage_type);
  const bool upper =
      storage_type == CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR;
  const int rows[] = {0, upper ? 2 : 1, 3};
  const int upper_cols[] = {0, 1, 1};
  const int lower_cols[] = {0, 0, 1};
  const double values[] = {4.0, 2.0, 3.0};
  std::copy(std::begin(rows), std::end(rows), m->mutable_rows());
  std::copy(std::begin(upper ? upper_cols : lower_cols),
            std::end(upper ? upper_cols : lower_cols),
            m->mutable_cols());
  std::copy(std::begin(values), std::end(values), m->mutable_values());
  return m;
}

TEST(SolveUsingDenseCholesky, SolvesSystemsStoredInEitherTriangle) {
  const Eigen::Vector2d expected(1.0, -2.0);
  const Eigen::Vector2d rhs(4.0 * 1.0 + 2.0 * -2.0, 2.0 * 1.0 + 3.0 * -2.0);
  for (const auto storage_type :
       {CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR,
        CompressedRowSparseMatrix::StorageType::LOWER_TRIANGULAR}) {
    const auto lhs = CreateSymmetricMatrix(storage_type);
    Vector solution;
    ASSERT_TRUE(SolveUsingDenseCholesky(*lhs, rhs, &solution));
    EXPECT_THAT(solution, MatrixNear(expected, kTolerance));
  }
}

TEST(SolveUsingDenseCholesky, FailsForIndefiniteMatrices) {
  auto lhs = CreateSymmetricMatrix(
      CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR);
  lhs->mutable_values()[0] = -4.0;
  Vector solution;
  EXPECT_FALSE(SolveUsingDenseCholesky(*lhs, Vector::Ones(2), &solution));
}

TEST(PrintTo, PrintsTheSizeAndThePositionOfABlock) {
  EXPECT_EQ(::testing::PrintToString(Block(2, 3)), "Block(size=2, position=3)");
}

// Creates the matrix
//
//   [ 1 0 2 ]
//   [ 0 3 0 ]
std::unique_ptr<CompressedRowSparseMatrix> CreateCompressedRowSparseMatrix() {
  auto m = std::make_unique<CompressedRowSparseMatrix>(2, 3, 3);
  const int rows[] = {0, 2, 3};
  const int cols[] = {0, 2, 1};
  const double values[] = {1.0, 2.0, 3.0};
  std::copy(std::begin(rows), std::end(rows), m->mutable_rows());
  std::copy(std::begin(cols), std::end(cols), m->mutable_cols());
  std::copy(std::begin(values), std::end(values), m->mutable_values());
  return m;
}

TEST(CompressedRowsAre, MatchesTheCompressedRowArrays) {
  const auto m = CreateCompressedRowSparseMatrix();
  EXPECT_THAT(*m,
              CompressedRowsAre(std::vector{0, 2, 3},
                                std::vector{0, 2, 1},
                                std::vector{1.0, 2.0, 3.0}));
}

TEST(CompressedRowsAre, ExplainsMismatchingColumnIndices) {
  const auto m = CreateCompressedRowSparseMatrix();
  const auto matcher = CompressedRowsAre(
      std::vector{0, 2, 3}, std::vector{0, 1, 1}, std::vector{1.0, 2.0, 3.0});
  EXPECT_THAT(*m, Not(matcher));
  EXPECT_THAT(Explain(matcher, *m), HasSubstr("column indices"));
}

TEST(CompressedRowsAre, RejectsAdditionalRows) {
  const auto m = CreateCompressedRowSparseMatrix();
  EXPECT_THAT(*m,
              Not(CompressedRowsAre(std::vector{0, 2, 3, 3},
                                    std::vector{0, 2, 1},
                                    std::vector{1.0, 2.0, 3.0})));
}

}  // namespace
}  // namespace ceres::internal
