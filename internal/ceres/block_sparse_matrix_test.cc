// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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

#include "ceres/block_sparse_matrix.h"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ceres/casts.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

namespace {

std::unique_ptr<BlockSparseMatrix> CreateTestMatrixFromId(int id) {
  if (id == 0) {
    // Create the following block sparse matrix:
    // [ 1 2 0 0  0 0 ]
    // [ 3 4 0 0  0 0 ]
    // [ 0 0 5 6  7 0 ]
    // [ 0 0 8 9 10 0 ]
    CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;
    bs->cols = {
        // Block size 2, position 0.
        Block(2, 0),
        // Block size 3, position 2.
        Block(3, 2),
        // Block size 1, position 5.
        Block(1, 5),
    };
    bs->rows = {CompressedRow(1), CompressedRow(1)};
    bs->rows[0].block = Block(2, 0);
    bs->rows[0].cells = {Cell(0, 0)};

    bs->rows[1].block = Block(2, 2);
    bs->rows[1].cells = {Cell(1, 4)};
    auto m = std::make_unique<BlockSparseMatrix>(bs);
    EXPECT_NE(m, nullptr);
    EXPECT_EQ(m->num_rows(), 4);
    EXPECT_EQ(m->num_cols(), 6);
    EXPECT_EQ(m->num_nonzeros(), 10);
    double* values = m->mutable_values();
    for (int i = 0; i < 10; ++i) {
      values[i] = i + 1;
    }
    return m;
  } else if (id == 1) {
    // Create the following block sparse matrix:
    // [ 1 2 0 5 6 0 ]
    // [ 3 4 0 7 8 0 ]
    // [ 0 0 9 0 0 0 ]
    CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;
    bs->cols = {
        // Block size 2, position 0.
        Block(2, 0),
        // Block size 1, position 2.
        Block(1, 2),
        // Block size 2, position 3.
        Block(2, 3),
        // Block size 1, position 5.
        Block(1, 5),
    };
    bs->rows = {CompressedRow(2), CompressedRow(1)};
    bs->rows[0].block = Block(2, 0);
    bs->rows[0].cells = {Cell(0, 0), Cell(2, 4)};

    bs->rows[1].block = Block(1, 2);
    bs->rows[1].cells = {Cell(1, 8)};
    auto m = std::make_unique<BlockSparseMatrix>(bs);
    EXPECT_NE(m, nullptr);
    EXPECT_EQ(m->num_rows(), 3);
    EXPECT_EQ(m->num_cols(), 6);
    EXPECT_EQ(m->num_nonzeros(), 9);
    double* values = m->mutable_values();
    for (int i = 0; i < 9; ++i) {
      values[i] = i + 1;
    }
    return m;
  }
  return nullptr;
}
}  // namespace

const int kNumThreads = 4;

class BlockSparseMatrixTest : public ::testing::Test {
 protected:
  void SetUp() final {
    std::unique_ptr<LinearLeastSquaresProblem> problem =
        CreateLinearLeastSquaresProblemFromId(2);
    CHECK(problem != nullptr);
    A_.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));

    problem = CreateLinearLeastSquaresProblemFromId(1);
    CHECK(problem != nullptr);
    B_.reset(down_cast<TripletSparseMatrix*>(problem->A.release()));

    CHECK_EQ(A_->num_rows(), B_->num_rows());
    CHECK_EQ(A_->num_cols(), B_->num_cols());
    CHECK_EQ(A_->num_nonzeros(), B_->num_nonzeros());
    context_.EnsureMinimumThreads(kNumThreads);
  }

  std::unique_ptr<BlockSparseMatrix> A_;
  std::unique_ptr<TripletSparseMatrix> B_;
  ContextImpl context_;
};

TEST_F(BlockSparseMatrixTest, SetZeroTest) {
  A_->SetZero();
  EXPECT_EQ(13, A_->num_nonzeros());
}

TEST_F(BlockSparseMatrixTest, RightMultiplyAndAccumulateTest) {
  Vector y_a = Vector::Zero(A_->num_rows());
  Vector y_b = Vector::Zero(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    A_->RightMultiplyAndAccumulate(x.data(), y_a.data());
    B_->RightMultiplyAndAccumulate(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, RightMultiplyAndAccumulateParallelTest) {
  Vector y_0 = Vector::Random(A_->num_rows());
  Vector y_s = y_0;
  Vector y_p = y_0;

  Vector x = Vector::Random(A_->num_cols());
  A_->RightMultiplyAndAccumulate(x.data(), y_s.data());

  A_->RightMultiplyAndAccumulate(x.data(), y_p.data(), &context_, kNumThreads);

  // Current parallel implementation is expected to be bit-exact
  EXPECT_EQ((y_s - y_p).norm(), 0.);
}

TEST_F(BlockSparseMatrixTest, LeftMultiplyAndAccumulateTest) {
  Vector y_a = Vector::Zero(A_->num_cols());
  Vector y_b = Vector::Zero(A_->num_cols());
  for (int i = 0; i < A_->num_rows(); ++i) {
    Vector x = Vector::Zero(A_->num_rows());
    x[i] = 1.0;
    A_->LeftMultiplyAndAccumulate(x.data(), y_a.data());
    B_->LeftMultiplyAndAccumulate(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, LeftMultiplyAndAccumulateParallelTest) {
  Vector y_0 = Vector::Random(A_->num_rows());
  Vector y_s = y_0;
  Vector y_p = y_0;

  Vector x = Vector::Random(A_->num_cols());
  A_->LeftMultiplyAndAccumulate(x.data(), y_s.data());

  A_->AddTransposeBlockStructure();
  A_->LeftMultiplyAndAccumulate(x.data(), y_p.data(), &context_, kNumThreads);

  // Parallel implementation for left products uses a different order of
  // traversal, thus results might be different
  EXPECT_LT((y_s - y_p).norm(), 1e-12);
}

TEST_F(BlockSparseMatrixTest, SquaredColumnNormTest) {
  Vector y_a = Vector::Zero(A_->num_cols());
  Vector y_b = Vector::Zero(A_->num_cols());
  A_->SquaredColumnNorm(y_a.data());
  B_->SquaredColumnNorm(y_b.data());
  EXPECT_LT((y_a - y_b).norm(), 1e-12);
}

TEST_F(BlockSparseMatrixTest, ToDenseMatrixTest) {
  Matrix m_a;
  Matrix m_b;
  A_->ToDenseMatrix(&m_a);
  B_->ToDenseMatrix(&m_b);
  EXPECT_LT((m_a - m_b).norm(), 1e-12);
}

TEST_F(BlockSparseMatrixTest, AppendRows) {
  std::unique_ptr<LinearLeastSquaresProblem> problem =
      CreateLinearLeastSquaresProblemFromId(2);
  std::unique_ptr<BlockSparseMatrix> m(
      down_cast<BlockSparseMatrix*>(problem->A.release()));
  A_->AppendRows(*m);
  EXPECT_EQ(A_->num_rows(), 2 * m->num_rows());
  EXPECT_EQ(A_->num_cols(), m->num_cols());

  problem = CreateLinearLeastSquaresProblemFromId(1);
  std::unique_ptr<TripletSparseMatrix> m2(
      down_cast<TripletSparseMatrix*>(problem->A.release()));
  B_->AppendRows(*m2);

  Vector y_a = Vector::Zero(A_->num_rows());
  Vector y_b = Vector::Zero(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    y_a.setZero();
    y_b.setZero();

    A_->RightMultiplyAndAccumulate(x.data(), y_a.data());
    B_->RightMultiplyAndAccumulate(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, AppendDeleteRowsTransposedStructure) {
  auto problem = CreateLinearLeastSquaresProblemFromId(2);
  std::unique_ptr<BlockSparseMatrix> m(
      down_cast<BlockSparseMatrix*>(problem->A.release()));

  A_->AddTransposeBlockStructure();
  auto block_structure = A_->block_structure();

  // Several AppendRows and DeleteRowBlocks operations are applied to matrix,
  // with regular and transpose block structures being compared after each
  // operation.
  //
  // Non-negative values encode number of row blocks to remove
  // -1 encodes appending matrix m
  const int num_row_blocks_to_delete[] = {0, -1, 1, -1, 8, -1, 10};
  for (auto& t : num_row_blocks_to_delete) {
    if (t == -1) {
      A_->AppendRows(*m);
    } else if (t > 0) {
      CHECK_GE(block_structure->rows.size(), t);
      A_->DeleteRowBlocks(t);
    }

    auto block_structure = A_->block_structure();
    auto transpose_block_structure = A_->transpose_block_structure();
    ASSERT_NE(block_structure, nullptr);
    ASSERT_NE(transpose_block_structure, nullptr);

    EXPECT_EQ(block_structure->rows.size(),
              transpose_block_structure->cols.size());
    EXPECT_EQ(block_structure->cols.size(),
              transpose_block_structure->rows.size());

    std::vector<int> nnz_col(transpose_block_structure->rows.size());
    for (int i = 0; i < block_structure->cols.size(); ++i) {
      EXPECT_EQ(block_structure->cols[i].position,
                transpose_block_structure->rows[i].block.position);
      const int col_size = transpose_block_structure->rows[i].block.size;
      EXPECT_EQ(block_structure->cols[i].size, col_size);

      for (auto& col_cell : transpose_block_structure->rows[i].cells) {
        int matches = 0;
        const int row_block_id = col_cell.block_id;
        nnz_col[i] +=
            col_size * transpose_block_structure->cols[row_block_id].size;
        for (auto& row_cell : block_structure->rows[row_block_id].cells) {
          if (row_cell.block_id != i) continue;
          EXPECT_EQ(row_cell.position, col_cell.position);
          ++matches;
        }
        EXPECT_EQ(matches, 1);
      }
      EXPECT_EQ(nnz_col[i], transpose_block_structure->rows[i].nnz);
      if (i > 0) {
        nnz_col[i] += nnz_col[i - 1];
      }
      EXPECT_EQ(nnz_col[i], transpose_block_structure->rows[i].cumulative_nnz);
    }
    for (int i = 0; i < block_structure->rows.size(); ++i) {
      EXPECT_EQ(block_structure->rows[i].block.position,
                transpose_block_structure->cols[i].position);
      EXPECT_EQ(block_structure->rows[i].block.size,
                transpose_block_structure->cols[i].size);

      for (auto& row_cell : block_structure->rows[i].cells) {
        int matches = 0;
        const int col_block_id = row_cell.block_id;
        for (auto& col_cell :
             transpose_block_structure->rows[col_block_id].cells) {
          if (col_cell.block_id != i) continue;
          EXPECT_EQ(col_cell.position, row_cell.position);
          ++matches;
        }
        EXPECT_EQ(matches, 1);
      }
    }
  }
}

TEST_F(BlockSparseMatrixTest, AppendAndDeleteBlockDiagonalMatrix) {
  const std::vector<Block>& column_blocks = A_->block_structure()->cols;
  const int num_cols =
      column_blocks.back().size + column_blocks.back().position;
  Vector diagonal(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    diagonal(i) = 2 * i * i + 1;
  }
  std::unique_ptr<BlockSparseMatrix> appendage(
      BlockSparseMatrix::CreateDiagonalMatrix(diagonal.data(), column_blocks));

  A_->AppendRows(*appendage);
  Vector y_a, y_b;
  y_a.resize(A_->num_rows());
  y_b.resize(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    y_a.setZero();
    y_b.setZero();

    A_->RightMultiplyAndAccumulate(x.data(), y_a.data());
    B_->RightMultiplyAndAccumulate(x.data(), y_b.data());
    EXPECT_LT((y_a.head(B_->num_rows()) - y_b.head(B_->num_rows())).norm(),
              1e-12);
    Vector expected_tail = Vector::Zero(A_->num_cols());
    expected_tail(i) = diagonal(i);
    EXPECT_LT((y_a.tail(A_->num_cols()) - expected_tail).norm(), 1e-12);
  }

  A_->DeleteRowBlocks(column_blocks.size());
  EXPECT_EQ(A_->num_rows(), B_->num_rows());
  EXPECT_EQ(A_->num_cols(), B_->num_cols());

  y_a.resize(A_->num_rows());
  y_b.resize(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    y_a.setZero();
    y_b.setZero();

    A_->RightMultiplyAndAccumulate(x.data(), y_a.data());
    B_->RightMultiplyAndAccumulate(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST(BlockSparseMatrix, CreateDiagonalMatrix) {
  std::vector<Block> column_blocks;
  column_blocks.emplace_back(2, 0);
  column_blocks.emplace_back(1, 2);
  column_blocks.emplace_back(3, 3);
  const int num_cols =
      column_blocks.back().size + column_blocks.back().position;
  Vector diagonal(num_cols);
  for (int i = 0; i < num_cols; ++i) {
    diagonal(i) = 2 * i * i + 1;
  }

  std::unique_ptr<BlockSparseMatrix> m(
      BlockSparseMatrix::CreateDiagonalMatrix(diagonal.data(), column_blocks));
  const CompressedRowBlockStructure* bs = m->block_structure();
  EXPECT_EQ(bs->cols.size(), column_blocks.size());
  for (int i = 0; i < column_blocks.size(); ++i) {
    EXPECT_EQ(bs->cols[i].size, column_blocks[i].size);
    EXPECT_EQ(bs->cols[i].position, column_blocks[i].position);
  }
  EXPECT_EQ(m->num_rows(), m->num_cols());
  Vector x = Vector::Ones(num_cols);
  Vector y = Vector::Zero(num_cols);
  m->RightMultiplyAndAccumulate(x.data(), y.data());
  for (int i = 0; i < num_cols; ++i) {
    EXPECT_NEAR(y[i], diagonal[i], std::numeric_limits<double>::epsilon());
  }
}

TEST(BlockSparseMatrix, ToDenseMatrix) {
  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(0);
    Matrix m_dense;
    m->ToDenseMatrix(&m_dense);
    EXPECT_EQ(m_dense.rows(), 4);
    EXPECT_EQ(m_dense.cols(), 6);
    Matrix m_expected(4, 6);
    m_expected << 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 8,
        9, 10, 0;
    EXPECT_EQ(m_dense, m_expected);
  }

  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(1);
    Matrix m_dense;
    m->ToDenseMatrix(&m_dense);
    EXPECT_EQ(m_dense.rows(), 3);
    EXPECT_EQ(m_dense.cols(), 6);
    Matrix m_expected(3, 6);
    m_expected << 1, 2, 0, 5, 6, 0, 3, 4, 0, 7, 8, 0, 0, 0, 9, 0, 0, 0;
    EXPECT_EQ(m_dense, m_expected);
  }
}

TEST(BlockSparseMatrix, ToCRSMatrix) {
  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(0);
    CompressedRowSparseMatrix m_crs(
        m->num_rows(), m->num_cols(), m->num_nonzeros());
    m->ToCompressedRowSparseMatrix(&m_crs);
    std::vector<int> rows_expected = {0, 2, 4, 7, 10};
    std::vector<int> cols_expected = {0, 1, 0, 1, 2, 3, 4, 2, 3, 4};
    std::vector<double> values_expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    for (int i = 0; i < rows_expected.size(); ++i) {
      EXPECT_EQ(m_crs.rows()[i], rows_expected[i]);
    }
    for (int i = 0; i < cols_expected.size(); ++i) {
      EXPECT_EQ(m_crs.cols()[i], cols_expected[i]);
    }
    for (int i = 0; i < values_expected.size(); ++i) {
      EXPECT_EQ(m_crs.values()[i], values_expected[i]);
    }
  }
  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(1);
    CompressedRowSparseMatrix m_crs(
        m->num_rows(), m->num_cols(), m->num_nonzeros());
    m->ToCompressedRowSparseMatrix(&m_crs);
    std::vector<int> rows_expected = {0, 4, 8, 9};
    std::vector<int> cols_expected = {0, 1, 3, 4, 0, 1, 3, 4, 2};
    std::vector<double> values_expected = {1, 2, 5, 6, 3, 4, 7, 8, 9};
    for (int i = 0; i < rows_expected.size(); ++i) {
      EXPECT_EQ(m_crs.rows()[i], rows_expected[i]);
    }
    for (int i = 0; i < cols_expected.size(); ++i) {
      EXPECT_EQ(m_crs.cols()[i], cols_expected[i]);
    }
    for (int i = 0; i < values_expected.size(); ++i) {
      EXPECT_EQ(m_crs.values()[i], values_expected[i]);
    }
  }
}

TEST(BlockSparseMatrix, CreateTranspose) {
  constexpr int kNumtrials = 10;
  BlockSparseMatrix::RandomMatrixOptions options;
  options.num_col_blocks = 10;
  options.min_col_block_size = 1;
  options.max_col_block_size = 3;

  options.num_row_blocks = 20;
  options.min_row_block_size = 1;
  options.max_row_block_size = 4;
  options.block_density = 0.25;
  std::mt19937 prng;

  for (int trial = 0; trial < kNumtrials; ++trial) {
    auto a = BlockSparseMatrix::CreateRandomMatrix(options, prng);

    auto ap_bs = std::make_unique<CompressedRowBlockStructure>();
    *ap_bs = *a->block_structure();
    BlockSparseMatrix ap(ap_bs.release());
    std::copy_n(a->values(), a->num_nonzeros(), ap.mutable_values());
    ap.AddTransposeBlockStructure();

    Vector x = Vector::Random(a->num_cols());
    Vector y = Vector::Random(a->num_rows());
    Vector a_x = Vector::Zero(a->num_rows());
    Vector a_t_y = Vector::Zero(a->num_cols());
    Vector ap_x = Vector::Zero(a->num_rows());
    Vector ap_t_y = Vector::Zero(a->num_cols());
    a->RightMultiplyAndAccumulate(x.data(), a_x.data());
    ap.RightMultiplyAndAccumulate(x.data(), ap_x.data());
    EXPECT_NEAR((a_x - ap_x).norm() / a_x.norm(),
                0.0,
                std::numeric_limits<double>::epsilon());
    a->LeftMultiplyAndAccumulate(y.data(), a_t_y.data());
    ap.LeftMultiplyAndAccumulate(y.data(), ap_t_y.data());
    EXPECT_NEAR((a_t_y - ap_t_y).norm() / a_t_y.norm(),
                0.0,
                std::numeric_limits<double>::epsilon());
  }
}

}  // namespace internal
}  // namespace ceres
