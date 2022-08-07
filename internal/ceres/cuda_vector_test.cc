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

#include "ceres/internal/config.h"
#include "ceres/internal/eigen.h"
#include "ceres/cuda_vector.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

#ifndef CERES_NO_CUDA

TEST(CUDAVector, InvalidOptionOnCreate) {
  CudaVector x;
  ContextImpl* context = nullptr;
  std::string message;
  EXPECT_FALSE(x.Init(context, &message));
}

TEST(CUDAVector, ValidOptionOnCreate) {
  CudaVector x;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x.Init(&context, &message));
}

TEST(CUDAVector, CopyVector) {
  Vector x(3);
  x << 1, 2, 3;
  CudaVector y;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(y.Init(&context, &message));
  y.CopyFrom(x);
  EXPECT_EQ(y.num_rows(), 3);

  Vector z(3);
  z << 0, 0, 0;
  y.CopyTo(&z);
  EXPECT_EQ(x, z);

  CudaVector y_copy;
  y_copy.Init(&context, &message);
  y_copy.CopyFrom(y);
  EXPECT_EQ(y_copy.num_rows(), 3);
  y_copy.CopyTo(&z);
  EXPECT_EQ(x, z);
}

TEST(CUDAVector, CopyMemory) {
  Vector x(3);
  x << 1, 2, 3;
  CudaVector y;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(y.Init(&context, &message));
  y.CopyFrom(x.data(), x.rows());
  EXPECT_EQ(y.num_rows(), 3);

  Vector z(3);
  z << 0, 0, 0;
  y.CopyTo(z.data());
  EXPECT_EQ(x, z);
}

TEST(CUDAVector, Dot) {
  Vector x(3);
  Vector y(3);
  x << 1, 2, 3;
  y << 100, 10, 1;
  CudaVector x_gpu;
  CudaVector y_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  EXPECT_TRUE(y_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);
  y_gpu.CopyFrom(y);

  EXPECT_EQ(x_gpu.dot(y_gpu), 123.0);
}

TEST(CUDAVector, Norm) {
  Vector x(3);
  x << 1, 2, 3;
  CudaVector x_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);

  EXPECT_NEAR(x_gpu.norm(),
              sqrt(1.0 + 4.0 + 9.0),
              std::numeric_limits<double>::epsilon());
}

TEST(CUDAVector, SetZero) {
  Vector x(4);
  x << 1, 1, 1, 1;
  CudaVector x_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);

  EXPECT_NEAR(x_gpu.norm(),
              2.0,
              std::numeric_limits<double>::epsilon());

  x_gpu.setZero();
  EXPECT_NEAR(x_gpu.norm(),
              0.0,
              std::numeric_limits<double>::epsilon());
}

TEST(CUDAVector, Resize) {
  CudaVector x_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  EXPECT_EQ(x_gpu.num_rows(), 0);
  x_gpu.resize(4);
  EXPECT_EQ(x_gpu.num_rows(), 4);
}

TEST(CUDAVector, Axpy) {
  Vector x(4);
  Vector y(4);
  x << 1, 1, 1, 1;
  y << 100, 10, 1, 0;
  CudaVector x_gpu;
  CudaVector y_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  EXPECT_TRUE(y_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);
  y_gpu.CopyFrom(y);

  x_gpu.Axpy(2.0, y_gpu);
  Vector result;
  Vector expected(4);
  expected << 201, 21, 3, 1;
  x_gpu.CopyTo(&result);
  EXPECT_EQ(result, expected);
}

TEST(CUDAVector, Axpby) {
  Vector x(4);
  Vector y(4);
  x << 1, 1, 1, 1;
  y << 100, 10, 1, 0;
  CudaVector x_gpu;
  CudaVector y_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  EXPECT_TRUE(y_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);
  y_gpu.CopyFrom(y);

  x_gpu.Axpby(2.0, y_gpu, 3.0);
  Vector result;
  Vector expected(4);
  expected << 203, 23, 5, 3;
  x_gpu.CopyTo(&result);
  EXPECT_EQ(result, expected);
}

TEST(CUDAVector, DtDxpy) {
  Vector x(4);
  Vector y(4);
  Vector D(4);
  x << 1, 2, 3, 4;
  y << 100, 10, 1, 0;
  D << 4, 3, 2, 1;
  CudaVector x_gpu;
  CudaVector y_gpu;
  CudaVector D_gpu;
  ContextImpl context;
  std::string message;
  EXPECT_TRUE(x_gpu.Init(&context, &message));
  EXPECT_TRUE(y_gpu.Init(&context, &message));
  EXPECT_TRUE(D_gpu.Init(&context, &message));
  x_gpu.CopyFrom(x);
  y_gpu.CopyFrom(y);
  D_gpu.CopyFrom(D);

  y_gpu.DtDxpy(D_gpu, x_gpu);
  Vector result;
  Vector expected(4);
  expected << 116, 28, 13, 4;
  y_gpu.CopyTo(&result);
  EXPECT_EQ(result, expected);
}

#endif  // CERES_NO_CUDA

}  // namespace internal
}  // namespace ceres
