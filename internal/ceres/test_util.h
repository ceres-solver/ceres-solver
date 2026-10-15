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

#ifdef CERES_HAS_GTEST

#ifndef CERES_INTERNAL_TEST_UTIL_H_
#define CERES_INTERNAL_TEST_UTIL_H_

#include <cfenv>
#include <cmath>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ceres/block_structure.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "ceres/jet.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/sparse_matrix.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Expects that x and y have a relative difference of no more than
// max_abs_relative_difference. If either x or y is zero, then the relative
// difference is interpreted as an absolute difference.
// If x and y have the same non-finite value (inf or nan) we treat them as being
// close. In such a case no error is thrown and true is returned.
CERES_NO_EXPORT bool ExpectClose(double x,
                                 double y,
                                 double max_abs_relative_difference);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] - q[i]| / max(|p[i]|, |q[i]|) < tolerance
CERES_NO_EXPORT void ExpectArraysClose(int n,
                                       const double* p,
                                       const double* q,
                                       double tolerance);

// Expects that for all i = 1,.., n - 1
//
//   |p[i] / max_norm_p - q[i] / max_norm_q| < tolerance
//
// where max_norm_p and max_norm_q are the max norms of the arrays p
// and q respectively.
CERES_NO_EXPORT void ExpectArraysCloseUptoScale(int n,
                                                const double* p,
                                                const double* q,
                                                double tolerance);

// √2/2, i.e., the sine and cosine of π/4. Halving is exact in binary
// floating-point arithmetic, so the value is correctly rounded.
//
// The constant is computed using mpmath by the following shell command:
//
// python3 - <<EOF
// from mpmath import mp
// mp.dps = 64
// print(mp.sqrt(2) / 2)
// EOF
template <typename T>
inline constexpr T kHalfSqrt2_v(
    0.707106781186547524400844362104849039284835937688474036588339869L);
inline constexpr double kHalfSqrt2 = kHalfSqrt2_v<double>;

// Stores the floating-point environment including the raised floating-point
// exceptions and the rounding mode, and restores it on destruction. Use it to
// keep computations from leaking floating-point exceptions to the caller.
class CERES_NO_EXPORT FloatEnvironmentScope {
 public:
  FloatEnvironmentScope() { std::fegetenv(&environment_); }
  ~FloatEnvironmentScope() { std::fesetenv(&environment_); }

  FloatEnvironmentScope(const FloatEnvironmentScope&) = delete;
  FloatEnvironmentScope& operator=(const FloatEnvironmentScope&) = delete;

 private:
  std::fenv_t environment_;
};

// Returns the absolute difference between x and y divided by the larger of
// their magnitudes. If either x or y is zero, the relative difference is the
// absolute difference. Two equal infinities and two NaNs have a relative
// difference of zero. Otherwise, an infinity yields an infinite and a single
// NaN a NaN relative difference. The floating-point environment of the caller
// is left unchanged.
CERES_NO_EXPORT double RelativeDifference(double x, double y);

// Matches a double whose RelativeDifference to expected does not exceed
// tolerance.
MATCHER_P2(RelativelyNear,
           expected,
           tolerance,
           absl::StrFormat("%s within a relative difference of %s of %s",
                           negation ? "isn't" : "is",
                           ::testing::PrintToString(tolerance),
                           ::testing::PrintToString(expected))) {
  const double relative_difference = RelativeDifference(arg, expected);
  *result_listener << "which is a relative difference of "
                   << relative_difference;
  return std::islessequal(relative_difference, tolerance);
}

// Matches a 2-tuple of doubles whose RelativeDifference does not exceed
// tolerance. Use as
//
//   EXPECT_THAT(actual, Pointwise(RelativelyNear(tolerance), expected));
MATCHER_P(RelativelyNear,
          tolerance,
          absl::StrFormat("%s within a relative difference of %s",
                          negation ? "aren't" : "are",
                          ::testing::PrintToString(tolerance))) {
  const double relative_difference =
      RelativeDifference(std::get<0>(arg), std::get<1>(arg));
  *result_listener << "which is a relative difference of "
                   << relative_difference;
  return std::islessequal(relative_difference, tolerance);
}

// Polymorphic matcher comparing an Eigen matrix or vector to the expected one
// by the Frobenius norm of their difference. Use the MatrixNear and
// MatrixRelativelyNear factory functions to create it.
class CERES_NO_EXPORT MatrixNearMatcher {
 public:
  MatrixNearMatcher(Matrix expected, double tolerance, bool relative);

  template <typename Derived>
  bool MatchAndExplain(const Eigen::DenseBase<Derived>& actual,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainMatrix(actual.template cast<double>(), listener);
  }

  void DescribeTo(std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 private:
  bool MatchAndExplainMatrix(const Matrix& actual,
                             ::testing::MatchResultListener* listener) const;
  void Describe(std::ostream* os, bool negation) const;

  Matrix expected_;
  double tolerance_;
  bool relative_;
};

// Matches a matrix of the same dimensions as expected with
//
//   ||actual - expected||_F <= tolerance.
template <typename Derived>
::testing::PolymorphicMatcher<MatrixNearMatcher> MatrixNear(
    const Eigen::DenseBase<Derived>& expected, double tolerance) {
  return ::testing::MakePolymorphicMatcher(MatrixNearMatcher(
      expected.template cast<double>(), tolerance, /*relative=*/false));
}

// Matches a matrix of the same dimensions as expected with
//
//   ||actual - expected||_F <= tolerance * max(||actual||_F, ||expected||_F).
//
// If either matrix is zero, the distance is absolute as in MatrixNear.
template <typename Derived>
::testing::PolymorphicMatcher<MatrixNearMatcher> MatrixRelativelyNear(
    const Eigen::DenseBase<Derived>& expected, double tolerance) {
  return ::testing::MakePolymorphicMatcher(MatrixNearMatcher(
      expected.template cast<double>(), tolerance, /*relative=*/true));
}

// Polymorphic matcher comparing each coefficient of an Eigen matrix or vector
// to the corresponding coefficient of the expected one using a matcher of
// 2-tuples. Use the MatrixPointwise factory function to create it.
class CERES_NO_EXPORT MatrixPointwiseMatcher {
 public:
  using TupleMatcher = ::testing::Matcher<std::tuple<double, double>>;

  MatrixPointwiseMatcher(TupleMatcher tuple_matcher, Matrix expected);

  template <typename Derived>
  bool MatchAndExplain(const Eigen::DenseBase<Derived>& actual,
                       ::testing::MatchResultListener* listener) const {
    return MatchAndExplainMatrix(actual.template cast<double>(), listener);
  }

  void DescribeTo(std::ostream* os) const;
  void DescribeNegationTo(std::ostream* os) const;

 private:
  bool MatchAndExplainMatrix(const Matrix& actual,
                             ::testing::MatchResultListener* listener) const;

  TupleMatcher tuple_matcher_;
  Matrix expected_;
};

// Matches a matrix of the same dimensions as expected whose coefficients and
// the corresponding expected coefficients satisfy the 2-tuple matcher. Unlike
// ::testing::Pointwise, it reports the row and the column of the first
// mismatch. Use as
//
//   EXPECT_THAT(actual, MatrixPointwise(DoubleNear(tolerance), expected));
//   EXPECT_THAT(actual, MatrixPointwise(RelativelyNear(tolerance), expected));
template <typename TupleMatcher, typename Derived>
::testing::PolymorphicMatcher<MatrixPointwiseMatcher> MatrixPointwise(
    const TupleMatcher& tuple_matcher,
    const Eigen::DenseBase<Derived>& expected) {
  return ::testing::MakePolymorphicMatcher(MatrixPointwiseMatcher(
      ::testing::MatcherCast<std::tuple<double, double>>(tuple_matcher),
      expected.template cast<double>()));
}

// Matches a matrix of the same dimensions as expected whose coefficients are
// equal to the corresponding expected coefficients.
template <typename Derived>
::testing::PolymorphicMatcher<MatrixPointwiseMatcher> MatrixEq(
    const Eigen::DenseBase<Derived>& expected) {
  return MatrixPointwise(::testing::Eq(), expected);
}

// Matches a Jet whose scalar part and each coefficient of whose dual part are
// RelativelyNear the corresponding parts of expected.
template <typename T, int N>
auto JetIsClose(const Jet<T, N>& expected, double tolerance) {
  return ::testing::AllOf(
      ::testing::Field(
          "a", &Jet<T, N>::a, RelativelyNear(expected.a, tolerance)),
      ::testing::Field("v",
                       &Jet<T, N>::v,
                       MatrixPointwise(RelativelyNear(tolerance), expected.v)));
}

// Matches a 2-tuple of Jets satisfying JetIsClose. Use as
//
//   EXPECT_THAT(actual, Pointwise(JetIsClose(tolerance), expected));
MATCHER_P(JetIsClose,
          tolerance,
          absl::StrFormat("%s within a relative difference of %s",
                          negation ? "aren't" : "are",
                          ::testing::PrintToString(tolerance))) {
  return ::testing::ExplainMatchResult(JetIsClose(std::get<1>(arg), tolerance),
                                       std::get<0>(arg),
                                       result_listener);
}

// Returns the dense matrix whose columns are the products of m and the
// columns of the identity matrix computed by RightMultiplyAndAccumulate.
CERES_NO_EXPORT Matrix RightMultiplyByIdentity(const SparseMatrix& m);

// Solves the linear system whose symmetric matrix is given by the triangular
// part stored in lhs using a dense Cholesky factorization. Returns false if
// the factorization fails.
CERES_NO_EXPORT bool SolveUsingDenseCholesky(
    const CompressedRowSparseMatrix& lhs, const Vector& rhs, Vector* solution);

// Prints a Block in assertion failure messages.
inline void PrintTo(const Block& block, std::ostream* os) {
  *os << absl::StrFormat(
      "Block(size=%d, position=%d)", block.size, block.position);
}

// Matches a CompressedRowSparseMatrix whose row offsets, column indices and
// values are equal to the given containers.
template <typename Rows, typename Cols, typename Values>
auto CompressedRowsAre(const Rows& rows,
                       const Cols& cols,
                       const Values& values) {
  using ::testing::ElementsAreArray;
  using ::testing::ResultOf;
  return ::testing::AllOf(
      ResultOf(
          "row offsets",
          [](const CompressedRowSparseMatrix& m) {
            return absl::MakeConstSpan(m.rows(), m.num_rows() + 1);
          },
          ElementsAreArray(rows)),
      ResultOf(
          "column indices",
          [](const CompressedRowSparseMatrix& m) {
            return absl::MakeConstSpan(m.cols(), m.num_nonzeros());
          },
          ElementsAreArray(cols)),
      ResultOf(
          "values",
          [](const CompressedRowSparseMatrix& m) {
            return absl::MakeConstSpan(m.values(), m.num_nonzeros());
          },
          ElementsAreArray(values)));
}

// Construct a fully qualified path for the test file depending on the
// local build/testing environment.
CERES_NO_EXPORT std::string TestFileAbsolutePath(const std::string& filename);

// A templated test fixture, that is used for testing Ceres end to end
// by computing a solution to the problem for a given solver
// configuration and comparing it to a reference solver configuration.
//
// It is assumed that the SystemTestProblem has an Solver::Options
// struct that contains the reference Solver configuration.
template <typename SystemTestProblem>
class CERES_NO_EXPORT SystemTest : public ::testing::Test {
 protected:
  void SetUp() final {
    SystemTestProblem system_test_problem;
    SolveAndEvaluateFinalResiduals(
        *system_test_problem.mutable_solver_options(),
        system_test_problem.mutable_problem(),
        &expected_final_residuals_);
  }

  void RunSolverForConfigAndExpectResidualsMatch(const Solver::Options& options,
                                                 Problem* problem) {
    std::vector<double> final_residuals;
    SolveAndEvaluateFinalResiduals(options, problem, &final_residuals);

    // We compare solutions by comparing their residual vectors. We do
    // not compare parameter vectors because it is much more brittle
    // and error prone to do so, since the same problem can have
    // nearly the same residuals at two completely different positions
    // in parameter space.
    CHECK_EQ(expected_final_residuals_.size(), final_residuals.size());
    for (int i = 0; i < final_residuals.size(); ++i) {
      EXPECT_NEAR(final_residuals[i],
                  expected_final_residuals_[i],
                  SystemTestProblem::kResidualTolerance)
          << "Not close enough residual:" << i;
    }
  }

  void SolveAndEvaluateFinalResiduals(const Solver::Options& options,
                                      Problem* problem,
                                      std::vector<double>* final_residuals) {
    Solver::Summary summary;
    Solve(options, problem, &summary);
    CHECK_NE(summary.termination_type, ceres::FAILURE);
    problem->Evaluate(
        Problem::EvaluateOptions(), nullptr, final_residuals, nullptr, nullptr);
  }

  std::vector<double> expected_final_residuals_;
};


}  // namespace internal
}  // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_TEST_UTIL_H_

#endif // CERES_HAS_GTEST
