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

#include <cmath>
#include <limits>
#include <ostream>
#include <utility>

#include "Eigen/Cholesky"
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

MatrixNearMatcher::MatrixNearMatcher(Matrix expected,
                                     double tolerance,
                                     bool relative)
    : expected_(std::move(expected)),
      tolerance_(tolerance),
      relative_(relative) {}

bool MatrixNearMatcher::MatchAndExplainMatrix(
    const Matrix& actual, ::testing::MatchResultListener* listener) const {
  if (actual.rows() != expected_.rows() || actual.cols() != expected_.cols()) {
    *listener << absl::StrFormat(
        "which has %d rows and %d columns instead of %d rows and %d columns",
        actual.rows(),
        actual.cols(),
        expected_.rows(),
        expected_.cols());
    return false;
  }

  const Matrix difference = actual - expected_;
  double distance = difference.norm();
  if (relative_ && distance > 0) {
    // Any difference from the zero matrix is infinitely large relative to it.
    distance /= expected_.norm();
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
  // Printing large matrices obscures the failure message.
  constexpr Eigen::Index kMaxDescribedCoefficients = 64;
  *os << absl::StrFormat("%s within a %sdistance of %s of",
                         negation ? "isn't" : "is",
                         relative_ ? "relative " : "",
                         ::testing::PrintToString(tolerance_));
  if (expected_.size() > kMaxDescribedCoefficients) {
    *os << absl::StrFormat(
        " the %dx%d expected matrix", expected_.rows(), expected_.cols());
  } else {
    *os << "\n" << expected_;
  }
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
