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

#include <memory>
#include <string>

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
template<typename T>
void CheckVectorEq(const std::vector<T>& a, const std::vector<T>& b) {
  EXPECT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b[i]);
  }
}

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
    bs->rows = {
      CompressedRow(1),
      CompressedRow(1)
    };
    bs->rows[0].block = Block(2, 0);
    bs->rows[0].cells = { Cell(0, 0) };

    bs->rows[1].block = Block(2, 2);
    bs->rows[1].cells = { Cell(1, 4) };
    std::unique_ptr<BlockSparseMatrix> m =
        std::make_unique<BlockSparseMatrix>(bs);
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
    bs->rows = {
      CompressedRow(2),
      CompressedRow(1)
    };
    bs->rows[0].block = Block(2, 0);
    bs->rows[0].cells = { Cell(0, 0), Cell(2, 4) };

    bs->rows[1].block = Block(1, 2);
    bs->rows[1].cells = { Cell(1, 8) };
    std::unique_ptr<BlockSparseMatrix> m =
        std::make_unique<BlockSparseMatrix>(bs);
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
  }

  std::unique_ptr<BlockSparseMatrix> A_;
  std::unique_ptr<TripletSparseMatrix> B_;
};

TEST_F(BlockSparseMatrixTest, SetZeroTest) {
  A_->SetZero();
  EXPECT_EQ(13, A_->num_nonzeros());
}

TEST_F(BlockSparseMatrixTest, RightMultiplyTest) {
  Vector y_a = Vector::Zero(A_->num_rows());
  Vector y_b = Vector::Zero(A_->num_rows());
  for (int i = 0; i < A_->num_cols(); ++i) {
    Vector x = Vector::Zero(A_->num_cols());
    x[i] = 1.0;
    A_->RightMultiply(x.data(), y_a.data());
    B_->RightMultiply(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
}

TEST_F(BlockSparseMatrixTest, LeftMultiplyTest) {
  Vector y_a = Vector::Zero(A_->num_cols());
  Vector y_b = Vector::Zero(A_->num_cols());
  for (int i = 0; i < A_->num_rows(); ++i) {
    Vector x = Vector::Zero(A_->num_rows());
    x[i] = 1.0;
    A_->LeftMultiply(x.data(), y_a.data());
    B_->LeftMultiply(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
  }
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

    A_->RightMultiply(x.data(), y_a.data());
    B_->RightMultiply(x.data(), y_b.data());
    EXPECT_LT((y_a - y_b).norm(), 1e-12);
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

    A_->RightMultiply(x.data(), y_a.data());
    B_->RightMultiply(x.data(), y_b.data());
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

    A_->RightMultiply(x.data(), y_a.data());
    B_->RightMultiply(x.data(), y_b.data());
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
  m->RightMultiply(x.data(), y.data());
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
    m_expected << 1, 2, 0, 0, 0, 0,
                  3, 4, 0, 0, 0, 0,
                  0, 0, 5, 6, 7, 0,
                  0, 0, 8, 9, 10, 0;
    EXPECT_EQ(m_dense, m_expected);
  }

  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(1);
    Matrix m_dense;
    m->ToDenseMatrix(&m_dense);
    EXPECT_EQ(m_dense.rows(), 3);
    EXPECT_EQ(m_dense.cols(), 6);
    Matrix m_expected(3, 6);
    m_expected << 1, 2, 0, 5, 6, 0,
                  3, 4, 0, 7, 8, 0,
                  0, 0, 9, 0, 0, 0;
    EXPECT_EQ(m_dense, m_expected);
  }
}

TEST(BlockSparseMatrix, ToCRSMatrix) {
  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(0);
    CRSMatrix m_crs;
    m->ToCRSMatrix(&m_crs);
    std::vector<int> rows_expected = {0, 2, 4, 7, 10};
    std::vector<int> cols_expected = {0, 1, 0, 1, 2, 3, 4, 2, 3, 4};
    std::vector<double> values_expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CheckVectorEq(rows_expected, m_crs.rows);
    CheckVectorEq(cols_expected, m_crs.cols);
    CheckVectorEq(values_expected, m_crs.values);
  }
  {
    std::unique_ptr<BlockSparseMatrix> m = CreateTestMatrixFromId(1);
    CRSMatrix m_crs;
    m->ToCRSMatrix(&m_crs);
    std::vector<int> rows_expected = {0, 4, 8, 9};
    std::vector<int> cols_expected = {0, 1, 3, 4, 0, 1, 3, 4, 2};
    std::vector<double> values_expected = {1, 2, 5, 6, 3, 4, 7, 8, 9};
    CheckVectorEq(rows_expected, m_crs.rows);
    CheckVectorEq(cols_expected, m_crs.cols);
    CheckVectorEq(values_expected, m_crs.values);
  }
}

}  // namespace internal
}  // namespace ceres
