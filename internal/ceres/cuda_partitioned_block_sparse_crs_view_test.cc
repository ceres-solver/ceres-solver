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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include "ceres/cuda_partitioned_block_sparse_crs_view.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#ifndef CERES_NO_CUDA

namespace ceres::internal {
class CudaPartitionedBlockSparseCRSViewTest : public ::testing::Test {
 protected:
  void SetUp() final {
    std::string message;
    CHECK(context_.InitCuda(&message))
        << "InitCuda() failed because: " << message;

    // CudaPartitionedBlockSparseCRSView does not make any assumptions on
    // structure on left-submatrix (as opposed to PartitionedMatrixView). Thus
    // we just create a random block-sparse matrix
    BlockSparseMatrix::RandomMatrixOptions options;
    // Left sub-matrix: 1234 x 89 blocks
    // Right sub-matrix: 1234 x 478 blocks
    options.num_row_blocks = 1234;
    options.num_col_blocks = 567;
    options.min_row_block_size = 1;
    options.max_row_block_size = 10;
    options.min_col_block_size = 1;
    options.max_col_block_size = 10;
    options.block_density = 0.2;
    std::mt19937 rng;
    A_ = BlockSparseMatrix::CreateRandomMatrix(options, rng);
    std::iota(
        A_->mutable_values(), A_->mutable_values() + A_->num_nonzeros(), 1);
  }

  std::unique_ptr<BlockSparseMatrix> A_;
  ContextImpl context_;
};

TEST_F(CudaPartitionedBlockSparseCRSViewTest, CreateUpdateValues) {
  const int num_col_blocks_e = 89;
  auto view =
      CudaPartitionedBlockSparseCRSView(*A_, num_col_blocks_e, &context_);

  const int num_rows = A_->num_rows();
  const int num_cols = A_->num_cols();

  const auto& bs = *A_->block_structure();
  const int num_cols_e = bs.cols[num_col_blocks_e].position;
  const int num_cols_f = num_cols - num_cols_e;

  // TODO: we definitely would like to use matrix() here, but
  // CudaSparseMatrix::RightMultiplyAndAccumulate is defined non-const because
  // it might allocate additional storage by request of cuSPARSE
  auto matrix_e = view.mutable_matrix_e();
  auto matrix_f = view.mutable_matrix_f();
  ASSERT_EQ(matrix_e->num_cols(), num_cols_e);
  ASSERT_EQ(matrix_e->num_rows(), num_rows);
  ASSERT_EQ(matrix_f->num_cols(), num_cols_f);
  ASSERT_EQ(matrix_f->num_rows(), num_rows);

  Vector x(num_cols);
  Vector x_left(num_cols_e);
  Vector x_right(num_cols_f);
  Vector y(num_rows);
  CudaVector x_cuda(&context_, num_cols);
  CudaVector x_left_cuda(&context_, num_cols_e);
  CudaVector x_right_cuda(&context_, num_cols_f);
  CudaVector y_cuda(&context_, num_rows);
  Vector y_cuda_host(num_rows);

  for (int i = 0; i < num_cols_e; ++i) {
    x.setZero();
    x_left.setZero();
    y.setZero();
    y_cuda.SetZero();
    x[i] = 1.;
    x_left[i] = 1.;
    x_left_cuda.CopyFromCpu(x_left);
    A_->RightMultiplyAndAccumulate(
        x.data(), y.data(), &context_, std::thread::hardware_concurrency());
    matrix_e->RightMultiplyAndAccumulate(x_left_cuda, &y_cuda);
    y_cuda.CopyTo(&y_cuda_host);
    // There will be up to 1 non-zero product per row, thus we expect an exact
    // match on 32-bit integer indices
    EXPECT_EQ((y - y_cuda_host).squaredNorm(), 0.);
  }
  for (int i = num_cols_e; i < num_cols_f; ++i) {
    x.setZero();
    x_right.setZero();
    y.setZero();
    y_cuda.SetZero();
    x[i] = 1.;
    x_right[i - num_cols_e] = 1.;
    x_right_cuda.CopyFromCpu(x_right);
    A_->RightMultiplyAndAccumulate(
        x.data(), y.data(), &context_, std::thread::hardware_concurrency());
    matrix_f->RightMultiplyAndAccumulate(x_right_cuda, &y_cuda);
    y_cuda.CopyTo(&y_cuda_host);
    // There will be up to 1 non-zero product per row, thus we expect an exact
    // match on 32-bit integer indices
    EXPECT_EQ((y - y_cuda_host).squaredNorm(), 0.);
  }
}
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
