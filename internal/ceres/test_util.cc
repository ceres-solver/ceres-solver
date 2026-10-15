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
// Author: keir@google.com (Keir Mierle)
//
// Utility functions useful for testing.

#ifdef CERES_HAS_GTEST

#include "ceres/test_util.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include "Eigen/Cholesky"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "ceres/file.h"
#include "ceres/is_close.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"


// This macro is used to inject additional path information specific
// to the build system.

#ifndef CERES_TEST_SRCDIR_SUFFIX
#define CERES_TEST_SRCDIR_SUFFIX ""
#endif

namespace ceres {
namespace internal {

bool ExpectClose(double x, double y, double max_abs_relative_difference) {
  if (std::isinf(x) && std::isinf(y)) {
    EXPECT_EQ(std::signbit(x), std::signbit(y));
    return true;
  }

  if (std::isnan(x) && std::isnan(y)) {
    return true;
  }

  double absolute_difference = fabs(x - y);
  double relative_difference;
  if (std::fpclassify(x) == FP_ZERO || std::fpclassify(y) == FP_ZERO) {
    // If x or y is exactly zero, then relative difference doesn't have any
    // meaning. Take the absolute difference instead.
    relative_difference = absolute_difference;
  } else {
    relative_difference =
        absolute_difference / std::max(std::fabs(x), std::fabs(y));
  }
  if (relative_difference > max_abs_relative_difference) {
    VLOG(1) << absl::StrFormat("x=%17g y=%17g abs=%17g rel=%17g",
                               x,
                               y,
                               absolute_difference,
                               relative_difference);
  }

  EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
  return relative_difference <= max_abs_relative_difference;
}

void ExpectArraysCloseUptoScale(int n,
                                const double* p,
                                const double* q,
                                double tol) {
  CHECK_GT(n, 0);
  CHECK(p);
  CHECK(q);

  double p_max = 0;
  double q_max = 0;
  int p_i = 0;
  int q_i = 0;

  for (int i = 0; i < n; ++i) {
    if (std::abs(p[i]) > p_max) {
      p_max = std::abs(p[i]);
      p_i = i;
    }
    if (std::abs(q[i]) > q_max) {
      q_max = std::abs(q[i]);
      q_i = i;
    }
  }

  // If both arrays are all zeros, they are equal up to scale, but
  // for testing purposes, that's more likely to be an error than
  // a desired result.
  CHECK_NE(p_max, 0.0);
  CHECK_NE(q_max, 0.0);

  for (int i = 0; i < n; ++i) {
    double p_norm = p[i] / p[p_i];
    double q_norm = q[i] / q[q_i];

    EXPECT_NEAR(p_norm, q_norm, tol) << "i=" << i;
  }
}

void ExpectArraysClose(int n, const double* p, const double* q, double tol) {
  CHECK_GT(n, 0);
  CHECK(p);
  CHECK(q);

  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(ExpectClose(p[i], q[i], tol)) << "p[" << i << "]" << p[i] << " "
                                              << "q[" << i << "]" << q[i] << " "
                                              << "tol: " << tol;
  }
}

double RelativeDifference(double x, double y) {
  if (std::isnan(x) || std::isnan(y)) {
    return std::isnan(x) && std::isnan(y)
               ? 0.0
               : std::numeric_limits<double>::quiet_NaN();
  }

  if (std::isinf(x) || std::isinf(y)) {
    return x == y ? 0.0 : std::numeric_limits<double>::infinity();
  }

  const FloatEnvironmentScope float_environment_scope;
  double relative_difference;
  IsClose(x, y, 0.0, &relative_difference, nullptr);
  return relative_difference;
}

namespace {

// Explains the dimensions of actual if they differ from those of expected.
bool MatchDimensions(const Matrix& actual,
                     const Matrix& expected,
                     ::testing::MatchResultListener* listener) {
  if (actual.rows() == expected.rows() && actual.cols() == expected.cols()) {
    return true;
  }

  *listener << absl::StrFormat(
      "which has %d rows and %d columns instead of %d rows and %d columns",
      actual.rows(),
      actual.cols(),
      expected.rows(),
      expected.cols());
  return false;
}

// Printing large matrices obscures the failure message.
constexpr Eigen::Index kMaxDescribedCoefficients = 64;

bool IsDescribedByCoefficients(const Matrix& expected) {
  return expected.size() <= kMaxDescribedCoefficients;
}

// Prints the coefficients of the expected matrix on separate lines.
std::string PrintCoefficients(const Matrix& expected) {
  std::ostringstream stream;
  stream << "\n" << expected;
  return stream.str();
}

}  // namespace

MatrixNearMatcher::MatrixNearMatcher(Matrix expected,
                                     double tolerance,
                                     bool relative)
    : expected_(std::move(expected)),
      tolerance_(tolerance),
      relative_(relative) {}

bool MatrixNearMatcher::MatchAndExplainMatrix(
    const Matrix& actual, ::testing::MatchResultListener* listener) const {
  if (!MatchDimensions(actual, expected_, listener)) {
    return false;
  }

  const Matrix difference = actual - expected_;
  double distance = difference.norm();
  const double actual_norm = actual.norm();
  const double expected_norm = expected_.norm();
  // Relative to the zero matrix, the distance is absolute.
  if (relative_ && std::fpclassify(actual_norm) != FP_ZERO &&
      std::fpclassify(expected_norm) != FP_ZERO) {
    distance /= std::max(actual_norm, expected_norm);
  }

  *listener << absl::StrFormat("which is at a %sdistance of %s",
                               relative_ ? "relative " : "",
                               ::testing::PrintToString(distance));
  if (difference.size() > 0 && difference.cwiseAbs().maxCoeff() > 0) {
    Eigen::Index row;
    Eigen::Index col;
    difference.cwiseAbs().maxCoeff(&row, &col);
    *listener << absl::StrFormat(
        " with the largest difference of %s at (%d, %d)",
        ::testing::PrintToString(difference(row, col)),
        row,
        col);
  }

  return std::islessequal(distance, tolerance_);
}

void MatrixNearMatcher::DescribeTo(std::ostream* os) const {
  Describe(os, /*negation=*/false);
}

void MatrixNearMatcher::DescribeNegationTo(std::ostream* os) const {
  Describe(os, /*negation=*/true);
}

void MatrixNearMatcher::Describe(std::ostream* os, bool negation) const {
  *os << absl::StrFormat("%s within a %sdistance of %s of%s",
                         negation ? "isn't" : "is",
                         relative_ ? "relative " : "",
                         ::testing::PrintToString(tolerance_),
                         IsDescribedByCoefficients(expected_)
                             ? PrintCoefficients(expected_)
                             : absl::StrFormat(" the %dx%d expected matrix",
                                               expected_.rows(),
                                               expected_.cols()));
}

MatrixPointwiseMatcher::MatrixPointwiseMatcher(TupleMatcher tuple_matcher,
                                               Matrix expected)
    : tuple_matcher_(std::move(tuple_matcher)),
      expected_(std::move(expected)) {}

bool MatrixPointwiseMatcher::MatchAndExplainMatrix(
    const Matrix& actual, ::testing::MatchResultListener* listener) const {
  if (!MatchDimensions(actual, expected_, listener)) {
    return false;
  }

  for (Eigen::Index row = 0; row < actual.rows(); ++row) {
    for (Eigen::Index col = 0; col < actual.cols(); ++col) {
      ::testing::StringMatchResultListener inner_listener;
      if (!tuple_matcher_.MatchAndExplain(
              std::make_tuple(actual(row, col), expected_(row, col)),
              &inner_listener)) {
        *listener << absl::StrFormat(
            "where the value pair (%s, %s) at (%d, %d) doesn't match",
            ::testing::PrintToString(actual(row, col)),
            ::testing::PrintToString(expected_(row, col)),
            row,
            col);
        if (!inner_listener.str().empty()) {
          *listener << ", " << inner_listener.str();
        }
        return false;
      }
    }
  }

  return true;
}

void MatrixPointwiseMatcher::DescribeTo(std::ostream* os) const {
  *os << absl::StrFormat(
      "is a %dx%d matrix whose coefficients and the corresponding "
      "coefficients of%s",
      expected_.rows(),
      expected_.cols(),
      IsDescribedByCoefficients(expected_)
          ? PrintCoefficients(expected_) + "\n"
          : std::string(" the expected matrix "));
  tuple_matcher_.DescribeTo(os);
}

void MatrixPointwiseMatcher::DescribeNegationTo(std::ostream* os) const {
  *os << absl::StrFormat(
      "isn't a %dx%d matrix or has a coefficient that and the corresponding "
      "coefficient of%s",
      expected_.rows(),
      expected_.cols(),
      IsDescribedByCoefficients(expected_)
          ? PrintCoefficients(expected_) + "\n"
          : std::string(" the expected matrix "));
  tuple_matcher_.DescribeNegationTo(os);
}

Matrix RightMultiplyByIdentity(const SparseMatrix& m) {
  Matrix dense(m.num_rows(), m.num_cols());
  for (int i = 0; i < m.num_cols(); ++i) {
    const Vector x = Vector::Unit(m.num_cols(), i);
    Vector y = Vector::Zero(m.num_rows());
    m.RightMultiplyAndAccumulate(x.data(), y.data());
    dense.col(i) = y;
  }
  return dense;
}

bool SolveUsingDenseCholesky(const CompressedRowSparseMatrix& lhs,
                             const Vector& rhs,
                             Vector* solution) {
  Matrix dense_triangular_lhs;
  lhs.ToDenseMatrix(&dense_triangular_lhs);
  const Matrix dense_lhs =
      lhs.storage_type() ==
              CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR
          ? Matrix(dense_triangular_lhs.selfadjointView<Eigen::Upper>())
          : Matrix(dense_triangular_lhs.selfadjointView<Eigen::Lower>());
  const Eigen::LLT<Matrix> llt(dense_lhs);
  if (llt.info() != Eigen::Success) {
    return false;
  }
  *solution = llt.solve(rhs);
  return llt.info() == Eigen::Success;
}

std::string TestFileAbsolutePath(const std::string& filename) {
#ifdef CERES_TEST_DATA_DIR
  // The absolute data directory does not depend on the location of the
  // build directory or on how the test is run.
  return JoinPath(CERES_TEST_DATA_DIR, filename);
#else
  return JoinPath(::testing::SrcDir() + CERES_TEST_SRCDIR_SUFFIX, filename);
#endif
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_HAS_GTEST
