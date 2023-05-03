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

#include "ceres/cuda_block_sparse_crs_view.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#ifndef CERES_NO_CUDA

namespace ceres::internal {
class CudaBlockSparseCRSViewTest : public ::testing::Test {
 protected:
  void SetUp() final {
    std::string message;
    CHECK(context_.InitCuda(&message))
        << "InitCuda() failed because: " << message;

    BlockSparseMatrix::RandomMatrixOptions options;
    options.num_row_blocks = 1234;
    options.min_row_block_size = 1;
    options.max_row_block_size = 10;
    options.num_col_blocks = 567;
    options.min_col_block_size = 1;
    options.max_col_block_size = 10;
    options.block_density = 0.2;
    std::mt19937 rng;
    A_ = BlockSparseMatrix::CreateRandomMatrix(options, rng);
    std::iota(
        A_->mutable_values(), A_->mutable_values() + A_->num_nonzeros(), 1);
  }

  std::unique_ptr<BlockSparseMatrix> A_;
  Vector x_;
  Vector b_;
  ContextImpl context_;
};

TEST_F(CudaBlockSparseCRSViewTest, CreateUpdateValues) {
  auto view = CudaBlockSparseCRSView(*A_, &context_, false);

  auto matrix = view.Matrix();
  ASSERT_EQ(matrix->num_cols(), A_->num_cols());
  ASSERT_EQ(matrix->num_rows(), A_->num_rows());
  ASSERT_EQ(matrix->num_nonzeros(), A_->num_nonzeros());

  view.UpdateValues(*A_);

  const int num_rows = A_->num_rows();
  const int num_cols = A_->num_cols();
  Vector x(num_cols);
  Vector y(num_rows);
  CudaVector x_cuda(&context_, num_cols);
  CudaVector y_cuda(&context_, num_rows);
  Vector y_cuda_host(num_rows);

  for (int i = 0; i < num_cols; ++i) {
    x.setZero();
    y.setZero();
    y_cuda.SetZero();
    x[i] = 1.;
    x_cuda.CopyFromCpu(x);
    A_->RightMultiplyAndAccumulate(
        x.data(), y.data(), &context_, std::thread::hardware_concurrency());
    matrix->RightMultiplyAndAccumulate(x_cuda, &y_cuda);
    y_cuda.CopyTo(&y_cuda_host);
    // There will be up to 1 non-zero product per row, thus we expect an exact
    // match on 32-bit integer indices
    EXPECT_EQ((y - y_cuda_host).squaredNorm(), 0.);
  }
}
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
