// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: joydeepb@cs.utexas.edu (Joydeep Biswas)

#include <string>

#include "ceres/casts.h"
#include "ceres/internal/config.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

#ifndef CERES_NO_CUDA

TEST(CudaSparseMatrix, InvalidOptionOnInit) {
  CudaSparseMatrix m;
  ContextImpl* context = nullptr;
  std::string message;
  EXPECT_FALSE(m.Init(context, &message));
}

class CudaSparseMatrixTest : public ::testing::Test {
 protected:
  void SetUp() final {
    std::unique_ptr<LinearLeastSquaresProblem> problem =
        CreateLinearLeastSquaresProblemFromId(2);
    CHECK(problem != nullptr);
    A_.reset(down_cast<BlockSparseMatrix*>(problem->A.release()));
    CHECK(A_ != nullptr);
    CHECK(problem->b != nullptr);
    CHECK(problem->x != nullptr);
    b_.resize(A_->num_rows());
    for (int i = 0; i < A_->num_rows(); ++i) {
      b_[i] = problem->b[i];
    }
    x_.resize(A_->num_cols());
    for (int i = 0; i < A_->num_cols(); ++i) {
      x_[i] = problem->x[i];
    }
    CHECK_EQ(A_->num_rows(), b_.rows());
    CHECK_EQ(A_->num_cols(), x_.rows());
  }

  std::unique_ptr<BlockSparseMatrix> A_;
  Vector x_;
  Vector b_;
  ContextImpl context_;
};

TEST_F(CudaSparseMatrixTest, RightMultiplyTest) {
  CudaSparseMatrix A_gpu;
  CudaVector x_gpu;
  CudaVector res_gpu;
  std::string message;
  EXPECT_TRUE(A_gpu.Init(&context_, &message));
  EXPECT_TRUE(x_gpu.Init(&context_, &message));
  EXPECT_TRUE(res_gpu.Init(&context_, &message));
  A_gpu.CopyFrom(*A_);
  x_gpu.CopyFrom(x_);

  Vector minus_b = -b_;
  // res = -b
  res_gpu.CopyFrom(minus_b);
  // res += A * x
  A_gpu.RightMultiply(x_gpu, &res_gpu);

  Vector res;
  res_gpu.CopyTo(&res);

  Vector res_expected = minus_b;
  A_->RightMultiply(x_.data(), res_expected.data());

  EXPECT_LE((res - res_expected).norm(),
            std::numeric_limits<double>::epsilon() * 1e3);
}

#endif  // CERES_NO_CUDA

}  // namespace internal
}  // namespace ceres
