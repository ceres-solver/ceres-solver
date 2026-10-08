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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/compressed_row_sparse_matrix.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "Eigen/SparseCore"
#include "absl/strings/str_format.h"
#include "ceres/casts.h"
#include "ceres/context_impl.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/test_util.h"
#include "ceres/triplet_sparse_matrix.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

constexpr int kMinNumBlocks = 1;
constexpr int kMaxNumBlocks = 10;
constexpr int kNumTrials = 10;

class CompressedRowSparseMatrixTest : public ::testing::Test {
 protected:
  void SetUp() final {
    auto problem = CreateLinearLeastSquaresProblemFromId(1);

    ASSERT_NE(problem, nullptr);

    tsm.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));
    crsm = CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm);

    num_rows = tsm->num_rows();
    num_cols = tsm->num_cols();

    std::vector<Block>* row_blocks = crsm->mutable_row_blocks();
    row_blocks->resize(num_rows);
    for (int i = 0; i < row_blocks->size(); ++i) {
      (*row_blocks)[i] = Block(1, i);
    }
    std::vector<Block>* col_blocks = crsm->mutable_col_blocks();
    col_blocks->resize(num_cols);
    for (int i = 0; i < col_blocks->size(); ++i) {
      (*col_blocks)[i] = Block(1, i);
    }
  }

  int num_rows;
  int num_cols;

  std::unique_ptr<TripletSparseMatrix> tsm;
  std::unique_ptr<CompressedRowSparseMatrix> crsm;
};

TEST_F(CompressedRowSparseMatrixTest, Scale) {
  Vector scale(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    scale(i) = i + 1;
  }

  tsm->ScaleColumns(scale.data());
  crsm->ScaleColumns(scale.data());
  EXPECT_THAT(RightMultiplyByIdentity(*crsm),
              MatrixNear(RightMultiplyByIdentity(*tsm), 0.0));
}

TEST_F(CompressedRowSparseMatrixTest, DeleteRows) {
  // Clear the row and column blocks as these are purely scalar tests.
  crsm->mutable_row_blocks()->clear();
  crsm->mutable_col_blocks()->clear();

  for (int i = 0; i < num_rows; ++i) {
    tsm->Resize(num_rows - i, num_cols);
    crsm->DeleteRows(crsm->num_rows() - tsm->num_rows());
    EXPECT_THAT(RightMultiplyByIdentity(*crsm),
                MatrixNear(RightMultiplyByIdentity(*tsm), 0.0));
  }
}

TEST_F(CompressedRowSparseMatrixTest, AppendRows) {
  // Clear the row and column blocks as these are purely scalar tests.
  crsm->mutable_row_blocks()->clear();
  crsm->mutable_col_blocks()->clear();

  for (int i = 0; i < num_rows; ++i) {
    TripletSparseMatrix tsm_appendage(*tsm);
    tsm_appendage.Resize(i, num_cols);

    tsm->AppendRows(tsm_appendage);
    auto crsm_appendage =
        CompressedRowSparseMatrix::FromTripletSparseMatrix(tsm_appendage);

    crsm->AppendRows(*crsm_appendage);
    EXPECT_THAT(RightMultiplyByIdentity(*crsm),
                MatrixNear(RightMultiplyByIdentity(*tsm), 0.0));
  }
}

TEST_F(CompressedRowSparseMatrixTest, AppendAndDeleteBlockDiagonalMatrix) {
  int num_diagonal_rows = crsm->num_cols();

  auto diagonal = std::make_unique<double[]>(num_diagonal_rows);
  for (int i = 0; i < num_diagonal_rows; ++i) {
    diagonal[i] = i;
  }

  std::vector<Block> row_and_column_blocks;
  row_and_column_blocks.emplace_back(1, 0);
  row_and_column_blocks.emplace_back(2, 1);
  row_and_column_blocks.emplace_back(2, 3);

  const std::vector<Block> pre_row_blocks = crsm->row_blocks();
  const std::vector<Block> pre_col_blocks = crsm->col_blocks();

  auto appendage = CompressedRowSparseMatrix::CreateBlockDiagonalMatrix(
      diagonal.get(), row_and_column_blocks);

  crsm->AppendRows(*appendage);

  const std::vector<Block> post_row_blocks = crsm->row_blocks();
  const std::vector<Block> post_col_blocks = crsm->col_blocks();

  std::vector<Block> expected_row_blocks = pre_row_blocks;
  expected_row_blocks.insert(expected_row_blocks.end(),
                             row_and_column_blocks.begin(),
                             row_and_column_blocks.end());

  std::vector<Block> expected_col_blocks = pre_col_blocks;

  EXPECT_EQ(expected_row_blocks, crsm->row_blocks());
  EXPECT_EQ(expected_col_blocks, crsm->col_blocks());

  crsm->DeleteRows(num_diagonal_rows);
  EXPECT_EQ(crsm->row_blocks(), pre_row_blocks);
  EXPECT_EQ(crsm->col_blocks(), pre_col_blocks);
}

TEST_F(CompressedRowSparseMatrixTest, ToDenseMatrix) {
  Matrix tsm_dense;
  Matrix crsm_dense;

  tsm->ToDenseMatrix(&tsm_dense);
  crsm->ToDenseMatrix(&crsm_dense);

  EXPECT_THAT(tsm_dense, MatrixNear(crsm_dense, 0.0));
}

TEST_F(CompressedRowSparseMatrixTest, ToCRSMatrix) {
  CRSMatrix crs_matrix;
  crsm->ToCRSMatrix(&crs_matrix);
  EXPECT_EQ(crsm->num_rows(), crs_matrix.num_rows);
  EXPECT_EQ(crsm->num_cols(), crs_matrix.num_cols);
  EXPECT_THAT(
      *crsm,
      CompressedRowsAre(crs_matrix.rows, crs_matrix.cols, crs_matrix.values));
}

TEST(CompressedRowSparseMatrix, CreateBlockDiagonalMatrix) {
  std::vector<Block> blocks;
  blocks.emplace_back(1, 0);
  blocks.emplace_back(2, 1);
  blocks.emplace_back(2, 3);

  Vector diagonal(5);
  for (int i = 0; i < 5; ++i) {
    diagonal(i) = i + 1;
  }

  auto matrix = CompressedRowSparseMatrix::CreateBlockDiagonalMatrix(
      diagonal.data(), blocks);

  EXPECT_EQ(matrix->num_rows(), 5);
  EXPECT_EQ(matrix->num_cols(), 5);
  EXPECT_EQ(matrix->num_nonzeros(), 9);
  EXPECT_EQ(blocks, matrix->row_blocks());
  EXPECT_EQ(blocks, matrix->col_blocks());

  Vector x(5);
  Vector y(5);

  x.setOnes();
  y.setZero();
  matrix->RightMultiplyAndAccumulate(x.data(), y.data());
  EXPECT_THAT(y, MatrixNear(diagonal, 0.0));

  y.setZero();
  matrix->LeftMultiplyAndAccumulate(x.data(), y.data());
  EXPECT_THAT(y, MatrixNear(diagonal, 0.0));

  Matrix dense;
  matrix->ToDenseMatrix(&dense);
  EXPECT_THAT(dense.diagonal(), MatrixNear(diagonal, 0.0));
}

TEST(CompressedRowSparseMatrix, Transpose) {
  //  0  1  0  2  3  0
  //  4  5  6  0  0  7
  //  8  9  0 10 11  0
  // 12  0 13 14 15  0
  //  0 16 17  0  0  0

  // Block structure:
  //  A  A  A  A  B  B
  //  A  A  A  A  B  B
  //  A  A  A  A  B  B
  //  C  C  C  C  D  D
  //  C  C  C  C  D  D
  //  C  C  C  C  D  D

  CompressedRowSparseMatrix matrix(5, 6, 30);
  int* rows = matrix.mutable_rows();
  int* cols = matrix.mutable_cols();
  double* values = matrix.mutable_values();
  matrix.mutable_row_blocks()->emplace_back(3, 0);
  matrix.mutable_row_blocks()->emplace_back(3, 3);
  matrix.mutable_col_blocks()->emplace_back(4, 0);
  matrix.mutable_col_blocks()->emplace_back(2, 4);

  rows[0] = 0;
  cols[0] = 1;
  cols[1] = 3;
  cols[2] = 4;

  rows[1] = 3;
  cols[3] = 0;
  cols[4] = 1;
  cols[5] = 2;
  cols[6] = 5;

  rows[2] = 7;
  cols[7] = 0;
  cols[8] = 1;
  cols[9] = 3;
  cols[10] = 4;

  rows[3] = 11;
  cols[11] = 0;
  cols[12] = 2;
  cols[13] = 3;
  cols[14] = 4;

  rows[4] = 15;
  cols[15] = 1;
  cols[16] = 2;
  rows[5] = 17;

  std::iota(values, values + 17, 1);

  auto transpose = matrix.Transpose();

  EXPECT_EQ(transpose->row_blocks(), matrix.col_blocks());
  EXPECT_EQ(transpose->col_blocks(), matrix.row_blocks());

  Matrix dense_matrix;
  matrix.ToDenseMatrix(&dense_matrix);

  Matrix dense_transpose;
  transpose->ToDenseMatrix(&dense_transpose);
  EXPECT_THAT(dense_transpose, MatrixNear(dense_matrix.transpose(), 1e-14));
}

TEST(CompressedRowSparseMatrix, FromTripletSparseMatrix) {
  std::mt19937 prng;
  TripletSparseMatrix::RandomMatrixOptions options;
  options.num_rows = 5;
  options.num_cols = 7;
  options.density = 0.5;

  for (int i = 0; i < kNumTrials; ++i) {
    SCOPED_TRACE(absl::StrFormat("trial %d", i));
    auto tsm = TripletSparseMatrix::CreateRandomMatrix(options, prng);
    auto crsm = CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm);

    Matrix expected;
    tsm->ToDenseMatrix(&expected);
    Matrix actual;
    crsm->ToDenseMatrix(&actual);
    EXPECT_THAT(
        actual,
        MatrixRelativelyNear(expected, std::numeric_limits<double>::epsilon()));
  }
}

TEST(CompressedRowSparseMatrix, FromTripletSparseMatrixTransposed) {
  std::mt19937 prng;
  TripletSparseMatrix::RandomMatrixOptions options;
  options.num_rows = 5;
  options.num_cols = 7;
  options.density = 0.5;

  for (int i = 0; i < kNumTrials; ++i) {
    SCOPED_TRACE(absl::StrFormat("trial %d", i));
    auto tsm = TripletSparseMatrix::CreateRandomMatrix(options, prng);
    auto crsm =
        CompressedRowSparseMatrix::FromTripletSparseMatrixTransposed(*tsm);

    Matrix tmp;
    tsm->ToDenseMatrix(&tmp);
    Matrix expected = tmp.transpose();
    Matrix actual;
    crsm->ToDenseMatrix(&actual);
    EXPECT_THAT(
        actual,
        MatrixRelativelyNear(expected, std::numeric_limits<double>::epsilon()));
  }
}

using StorageType = CompressedRowSparseMatrix::StorageType;

static std::string ParamInfoToString(
    const testing::TestParamInfo<StorageType>& info) {
  switch (info.param) {
    case StorageType::UPPER_TRIANGULAR:
      return "UPPER";
    case StorageType::LOWER_TRIANGULAR:
      return "LOWER";
    case StorageType::UNSYMMETRIC:
      return "UNSYMMETRIC";
  }
  return "UNKNOWN";
}

// Creates a random block matrix with twice as many row blocks as column
// blocks.
static std::unique_ptr<CompressedRowSparseMatrix> CreateRandomMatrix(
    int num_blocks, StorageType storage_type, std::mt19937& prng) {
  constexpr int kMinBlockSize = 1;
  constexpr int kMaxBlockSize = 5;
  std::uniform_real_distribution<double> uniform(0.5, 1.0);
  CompressedRowSparseMatrix::RandomMatrixOptions options;
  options.num_col_blocks = num_blocks;
  options.min_col_block_size = kMinBlockSize;
  options.max_col_block_size = kMaxBlockSize;
  options.num_row_blocks = 2 * num_blocks;
  options.min_row_block_size = kMinBlockSize;
  options.max_row_block_size = kMaxBlockSize;
  options.block_density = uniform(prng);
  options.storage_type = storage_type;
  return CompressedRowSparseMatrix::CreateRandomMatrix(options, prng);
}

// Returns the dense matrix represented by the matrix with the given storage
// type.
static Matrix ToFullDenseMatrix(const CompressedRowSparseMatrix& matrix) {
  Matrix dense;
  matrix.ToDenseMatrix(&dense);
  switch (matrix.storage_type()) {
    case StorageType::UPPER_TRIANGULAR:
      return dense.selfadjointView<Eigen::Upper>();
    case StorageType::LOWER_TRIANGULAR:
      return dense.selfadjointView<Eigen::Lower>();
    case StorageType::UNSYMMETRIC:
      return dense;
  }
  return dense;
}

class RightMultiplyAndAccumulateTest
    : public ::testing::TestWithParam<StorageType> {};

TEST_P(RightMultiplyAndAccumulateTest, MatchesDenseProduct) {
  constexpr double kTolerance = std::numeric_limits<double>::epsilon() * 10;
  std::mt19937 prng;
  for (int num_blocks = kMinNumBlocks; num_blocks < kMaxNumBlocks;
       ++num_blocks) {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      SCOPED_TRACE(absl::StrFormat("%d blocks, trial %d", num_blocks, trial));
      auto matrix = CreateRandomMatrix(num_blocks, GetParam(), prng);
      const Vector x = Vector::Random(matrix->num_cols());

      Vector actual_y = Vector::Zero(matrix->num_rows());
      matrix->RightMultiplyAndAccumulate(x.data(), actual_y.data());

      const Matrix dense = ToFullDenseMatrix(*matrix);
      ASSERT_THAT(actual_y, MatrixRelativelyNear(dense * x, kTolerance))
          << "matrix:\n"
          << dense << "\nx: " << x.transpose();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CompressedRowSparseMatrix,
                         RightMultiplyAndAccumulateTest,
                         ::testing::Values(StorageType::LOWER_TRIANGULAR,
                                           StorageType::UPPER_TRIANGULAR,
                                           StorageType::UNSYMMETRIC),
                         ParamInfoToString);

class LeftMultiplyAndAccumulateTest
    : public ::testing::TestWithParam<StorageType> {};

TEST_P(LeftMultiplyAndAccumulateTest, MatchesDenseProduct) {
  constexpr double kTolerance = std::numeric_limits<double>::epsilon() * 10;
  std::mt19937 prng;
  for (int num_blocks = kMinNumBlocks; num_blocks < kMaxNumBlocks;
       ++num_blocks) {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      SCOPED_TRACE(absl::StrFormat("%d blocks, trial %d", num_blocks, trial));
      auto matrix = CreateRandomMatrix(num_blocks, GetParam(), prng);
      const Vector x = Vector::Random(matrix->num_rows());

      Vector actual_y = Vector::Zero(matrix->num_cols());
      matrix->LeftMultiplyAndAccumulate(x.data(), actual_y.data());

      const Matrix dense = ToFullDenseMatrix(*matrix);
      ASSERT_THAT(actual_y,
                  MatrixRelativelyNear(dense.transpose() * x, kTolerance))
          << "matrix:\n"
          << dense << "\nx: " << x.transpose();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CompressedRowSparseMatrix,
                         LeftMultiplyAndAccumulateTest,
                         ::testing::Values(StorageType::LOWER_TRIANGULAR,
                                           StorageType::UPPER_TRIANGULAR,
                                           StorageType::UNSYMMETRIC),
                         ParamInfoToString);

class SquaredColumnNormTest : public ::testing::TestWithParam<StorageType> {};

TEST_P(SquaredColumnNormTest, MatchesDenseSquaredColumnNorm) {
  constexpr double kTolerance = std::numeric_limits<double>::epsilon() * 10;
  std::mt19937 prng;
  for (int num_blocks = kMinNumBlocks; num_blocks < kMaxNumBlocks;
       ++num_blocks) {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      SCOPED_TRACE(absl::StrFormat("%d blocks, trial %d", num_blocks, trial));
      auto matrix = CreateRandomMatrix(num_blocks, GetParam(), prng);

      Vector actual = Vector::Zero(matrix->num_cols());
      matrix->SquaredColumnNorm(actual.data());

      const Matrix dense = ToFullDenseMatrix(*matrix);
      const Vector expected = dense.colwise().squaredNorm();
      ASSERT_THAT(actual, MatrixRelativelyNear(expected, kTolerance))
          << "matrix:\n"
          << dense;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(CompressedRowSparseMatrix,
                         SquaredColumnNormTest,
                         ::testing::Values(StorageType::LOWER_TRIANGULAR,
                                           StorageType::UPPER_TRIANGULAR,
                                           StorageType::UNSYMMETRIC),
                         ParamInfoToString);

const int kMaxNumThreads = 8;
class CompressedRowSparseMatrixParallelTest
    : public ::testing::TestWithParam<int> {
  void SetUp() final { context_.EnsureMinimumThreads(kMaxNumThreads); }

 protected:
  ContextImpl context_;
};

TEST_P(CompressedRowSparseMatrixParallelTest,
       RightMultiplyAndAccumulateUnsymmetric) {
  constexpr double kTolerance = std::numeric_limits<double>::epsilon() * 10;
  const int kNumThreads = GetParam();
  std::mt19937 prng;
  for (int num_blocks = kMinNumBlocks; num_blocks < kMaxNumBlocks;
       ++num_blocks) {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      SCOPED_TRACE(absl::StrFormat("%d blocks, trial %d", num_blocks, trial));
      auto matrix =
          CreateRandomMatrix(num_blocks, StorageType::UNSYMMETRIC, prng);
      const Vector x = Vector::Random(matrix->num_cols());

      Vector actual_y = Vector::Zero(matrix->num_rows());
      matrix->RightMultiplyAndAccumulate(
          x.data(), actual_y.data(), &context_, kNumThreads);

      Matrix dense;
      matrix->ToDenseMatrix(&dense);
      ASSERT_THAT(actual_y, MatrixRelativelyNear(dense * x, kTolerance))
          << "matrix:\n"
          << dense << "\nx: " << x.transpose();
    }
  }
}
INSTANTIATE_TEST_SUITE_P(ParallelProducts,
                         CompressedRowSparseMatrixParallelTest,
                         ::testing::Values(1, 2, 4, 8),
                         ::testing::PrintToStringParamName());

// TODO(sameeragarwal) Add tests for the random matrix creation methods.

}  // namespace ceres::internal
