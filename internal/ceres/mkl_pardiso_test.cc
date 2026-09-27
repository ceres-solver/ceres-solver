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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#include "ceres/mkl_pardiso.h"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/config.h"
#include "ceres/mkl_pardiso_diagnostics.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

#ifndef CERES_NO_MKL
namespace {

CompressedRowSparseMatrix CreateUpperTriangularMatrix(
    const int num_rows,
    const std::vector<int>& rows,
    const std::vector<int>& cols,
    const std::vector<double>& values) {
  CompressedRowSparseMatrix matrix(
      num_rows, num_rows, static_cast<int>(values.size()));
  std::copy(rows.begin(), rows.end(), matrix.mutable_rows());
  std::copy(cols.begin(), cols.end(), matrix.mutable_cols());
  std::copy(values.begin(), values.end(), matrix.mutable_values());
  matrix.set_storage_type(
      CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR);
  return matrix;
}

// A = [4 1; 1 3]
CompressedRowSparseMatrix CreateUpperPositiveDefiniteMatrix() {
  return CreateUpperTriangularMatrix(2, {0, 2, 3}, {0, 1, 1}, {4.0, 1.0, 3.0});
}

// Factorizes matrix and expects that solving it for rhs yields expected.
void ExpectSolution(MklSparseCholesky& solver,
                    CompressedRowSparseMatrix* matrix,
                    const std::vector<double>& rhs,
                    const std::vector<double>& expected) {
  constexpr double kTolerance = 1e-12;
  std::string message;
  ASSERT_EQ(solver.Factorize(matrix, &message),
            LinearSolverTerminationType::SUCCESS)
      << message;
  std::vector<double> solution(rhs.size());
  ASSERT_EQ(solver.Solve(rhs.data(), solution.data(), &message),
            LinearSolverTerminationType::SUCCESS)
      << message;
  EXPECT_THAT(
      solution,
      ::testing::Pointwise(::testing::DoubleNear(kTolerance), expected));
}

// Switching to 64-bit integers is only a remedy if they are not used yet.
template <typename Integer>
struct SuggestsIlp64Interface;

template <>
struct SuggestsIlp64Interface<int> : std::true_type {};

template <>
struct SuggestsIlp64Interface<MKL_INT64> : std::false_type {};

}  // namespace

TEST(MklPardiso, SolvesWithTwoLevelFactorization) {
  auto matrix = CreateUpperPositiveDefiniteMatrix();
  auto solver = MklSparseCholesky::Create(OrderingType::NESDIS, 2, true);
  ExpectSolution(*solver, &matrix, {1.0, 2.0}, {1.0 / 11.0, 7.0 / 11.0});
}

// The solver keeps the structure of the first factorization, as the
// iterations of a solve do, and only copies the new values.
TEST(MklPardiso, RefactorizesMatrixWithSortedRows) {
  auto matrix = CreateUpperPositiveDefiniteMatrix();
  auto solver = MklSparseCholesky::Create(OrderingType::AMD, 1, false);
  ASSERT_NO_FATAL_FAILURE(
      ExpectSolution(*solver, &matrix, {1.0, 2.0}, {1.0 / 11.0, 7.0 / 11.0}));

  // A = [6 2; 2 5]
  const std::vector<double> values{6.0, 2.0, 5.0};
  std::copy(values.begin(), values.end(), matrix.mutable_values());
  ExpectSolution(*solver, &matrix, {1.0, 2.0}, {1.0 / 26.0, 5.0 / 13.0});
}

// Ceres does not guarantee sorted columns within a row, which PARDISO
// requires. Block upper triangular storage also holds entries below the
// diagonal, here (1, 0) between the entries of the second row. Such rows take
// a separate path when the values are copied again.
TEST(MklPardiso, RefactorizesMatrixWithUnsortedRows) {
  // A = [4 1 0; 1 3 1; 0 1 2]
  auto matrix = CreateUpperTriangularMatrix(
      3, {0, 2, 5, 6}, {1, 0, 2, 0, 1, 2}, {1.0, 4.0, 1.0, 1.0, 3.0, 2.0});
  auto solver = MklSparseCholesky::Create(OrderingType::AMD, 1, false);
  ASSERT_NO_FATAL_FAILURE(ExpectSolution(
      *solver, &matrix, {1.0, 2.0, 3.0}, {2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0}));

  // A = [5 2 0; 2 6 1; 0 1 4]
  const std::vector<double> values{2.0, 5.0, 1.0, 2.0, 6.0, 4.0};
  std::copy(values.begin(), values.end(), matrix.mutable_values());
  ExpectSolution(*solver,
                 &matrix,
                 {1.0, 2.0, 3.0},
                 {13.0 / 99.0, 17.0 / 99.0, 70.0 / 99.0});
}

TEST(MklPardiso, RejectsChangedNonzeroCountAfterAnalysis) {
  auto matrix = CreateUpperPositiveDefiniteMatrix();
  auto solver = MklSparseCholesky::Create(OrderingType::AMD, 1, false);
  std::string message;
  ASSERT_EQ(solver->Factorize(&matrix, &message),
            LinearSolverTerminationType::SUCCESS)
      << message;

  auto diagonal = CreateUpperTriangularMatrix(2, {0, 1, 2}, {0, 1}, {4.0, 3.0});
  EXPECT_EQ(solver->Factorize(&diagonal, &message),
            LinearSolverTerminationType::FATAL_ERROR);
  EXPECT_THAT(message, ::testing::HasSubstr("structure changed"));
}

TEST(MklPardiso, ReportsIndefiniteMatrixAsNumericalFailure) {
  auto matrix = CreateUpperPositiveDefiniteMatrix();
  matrix.mutable_values()[0] = -4.0;

  auto solver = MklSparseCholesky::Create(OrderingType::AMD, 1, false);
  std::string message;
  EXPECT_EQ(solver->Factorize(&matrix, &message),
            LinearSolverTerminationType::FAILURE);
}

// The count PARDISO reported for the factor of the Schur complement of
// final/problem-13682-4456117 from the BAL dataset with 32-bit integers.
constexpr PardisoIndex kNegativeFactorNonzeros = -1763537418;

TEST(MklPardiso, ChecksReportedFactorNonzeroCount) {
  std::string message;
  EXPECT_TRUE(CheckPardisoFactorNonzeros(0, &message)) << message;
  EXPECT_FALSE(CheckPardisoFactorNonzeros(kNegativeFactorNonzeros, &message));
  EXPECT_THAT(message,
              ::testing::AllOf(
                  ::testing::HasSubstr(std::to_string(kNegativeFactorNonzeros)),
                  ::testing::HasSubstr(std::to_string(
                      std::numeric_limits<PardisoIndex>::max()))));
  EXPECT_THAT(message,
              ::testing::Conditional(
                  SuggestsIlp64Interface<PardisoIndex>::value,
                  ::testing::HasSubstr("intel_ilp64"),
                  ::testing::Not(::testing::HasSubstr("intel_ilp64"))));
}

#endif  // CERES_NO_MKL

}  // namespace ceres::internal
