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

#include <math.h>

#include <limits>
#include <string>
#include <vector>

#include "ceres/ceres_cuda_kernels.h"
#include "ceres/cuda_buffer.h"
#include "ceres/internal/config.h"
#include "ceres/internal/eigen.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

#ifndef CERES_NO_CUDA

TEST(CudaFP64ToFP32, SimpleConversions) {
  std::vector<double> fp64_cpu = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  CudaBuffer<double> fp64_gpu;
  fp64_gpu.CopyFromCpuVector(fp64_cpu, cudaStreamDefault);
  CudaBuffer<float> fp32_gpu;
  fp32_gpu.Reserve(fp64_cpu.size());
  CudaFP64ToFP32(
      fp64_gpu.data(), fp32_gpu.data(), fp64_cpu.size(), cudaStreamDefault);
  std::vector<float> fp32_cpu(fp64_cpu.size());
  fp32_gpu.CopyToCpu(fp32_cpu.data(), fp32_cpu.size());
  for (int i = 0; i < fp32_cpu.size(); ++i) {
    EXPECT_EQ(fp32_cpu[i], static_cast<float>(fp64_cpu[i]));
  }
}

TEST(CudaFP64ToFP32, NumericallyExtremeValues) {
  std::vector<double> fp64_cpu = {
      DBL_MIN, 10.0 * DBL_MIN, DBL_MAX, 0.1 * DBL_MAX};
  // First just make sure that the compiler has represented these values
  // accurately as fp64.
  EXPECT_GT(fp64_cpu[0], 0.0);
  EXPECT_GT(fp64_cpu[1], 0.0);
  EXPECT_TRUE(std::isfinite(fp64_cpu[2]));
  EXPECT_TRUE(std::isfinite(fp64_cpu[3]));
  CudaBuffer<double> fp64_gpu;
  fp64_gpu.CopyFromCpuVector(fp64_cpu, cudaStreamDefault);
  CudaBuffer<float> fp32_gpu;
  fp32_gpu.Reserve(fp64_cpu.size());
  CudaFP64ToFP32(
      fp64_gpu.data(), fp32_gpu.data(), fp64_cpu.size(), cudaStreamDefault);
  std::vector<float> fp32_cpu(fp64_cpu.size());
  fp32_gpu.CopyToCpu(fp32_cpu.data(), fp32_cpu.size());
  EXPECT_EQ(fp32_cpu[0], 0.0f);
  EXPECT_EQ(fp32_cpu[1], 0.0f);
  EXPECT_EQ(fp32_cpu[2], std::numeric_limits<float>::infinity());
  EXPECT_EQ(fp32_cpu[3], std::numeric_limits<float>::infinity());
}

TEST(CudaFP32ToFP64, SimpleConversions) {
  std::vector<float> fp32_cpu = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  CudaBuffer<float> fp32_gpu;
  fp32_gpu.CopyFromCpuVector(fp32_cpu, cudaStreamDefault);
  CudaBuffer<double> fp64_gpu;
  fp64_gpu.Reserve(fp32_cpu.size());
  CudaFP32ToFP64(
      fp32_gpu.data(), fp64_gpu.data(), fp32_cpu.size(), cudaStreamDefault);
  std::vector<double> fp64_cpu(fp32_cpu.size());
  fp64_gpu.CopyToCpu(fp64_cpu.data(), fp64_cpu.size());
  for (int i = 0; i < fp64_cpu.size(); ++i) {
    EXPECT_EQ(fp64_cpu[i], static_cast<double>(fp32_cpu[i]));
  }
}

TEST(CudaSetZeroFP32, NonZeroInput) {
  std::vector<float> fp32_cpu = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  CudaBuffer<float> fp32_gpu;
  fp32_gpu.CopyFromCpuVector(fp32_cpu, cudaStreamDefault);
  CudaSetZeroFP32(fp32_gpu.data(), fp32_cpu.size(), cudaStreamDefault);
  std::vector<float> fp32_cpu_zero(fp32_cpu.size());
  fp32_gpu.CopyToCpu(fp32_cpu_zero.data(), fp32_cpu_zero.size());
  for (int i = 0; i < fp32_cpu_zero.size(); ++i) {
    EXPECT_EQ(fp32_cpu_zero[i], 0.0f);
  }
}

TEST(CudaSetZeroFP64, NonZeroInput) {
  std::vector<double> fp64_cpu = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  CudaBuffer<double> fp64_gpu;
  fp64_gpu.CopyFromCpuVector(fp64_cpu, cudaStreamDefault);
  CudaSetZeroFP64(fp64_gpu.data(), fp64_cpu.size(), cudaStreamDefault);
  std::vector<double> fp64_cpu_zero(fp64_cpu.size());
  fp64_gpu.CopyToCpu(fp64_cpu_zero.data(), fp64_cpu_zero.size());
  for (int i = 0; i < fp64_cpu_zero.size(); ++i) {
    EXPECT_EQ(fp64_cpu_zero[i], 0.0);
  }
}

TEST(CudaDsxpy, DoubleValues) {
  std::vector<float> fp32_cpu_a = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  std::vector<double> fp64_cpu_b = {
      1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
  CudaBuffer<float> fp32_gpu_a;
  fp32_gpu_a.CopyFromCpuVector(fp32_cpu_a, cudaStreamDefault);
  CudaBuffer<double> fp64_gpu_b;
  fp64_gpu_b.CopyFromCpuVector(fp64_cpu_b, cudaStreamDefault);
  CudaDsxpy(fp64_gpu_b.data(),
            fp32_gpu_a.data(),
            fp32_gpu_a.size(),
            cudaStreamDefault);
  fp64_gpu_b.CopyToCpu(fp64_cpu_b.data(), fp64_cpu_b.size());
  for (int i = 0; i < fp64_cpu_b.size(); ++i) {
    EXPECT_DOUBLE_EQ(fp64_cpu_b[i], 2.0 * fp32_cpu_a[i]);
  }
}

TEST(CudaDtDxpy, ComputeFourItems) {
  std::vector<double> x_cpu = {1, 2, 3, 4};
  std::vector<double> y_cpu = {4, 3, 2, 1};
  std::vector<double> d_cpu = {10, 20, 30, 40};
  CudaBuffer<double> x_gpu;
  x_gpu.CopyFromCpuVector(x_cpu, cudaStreamDefault);
  CudaBuffer<double> y_gpu;
  y_gpu.CopyFromCpuVector(y_cpu, cudaStreamDefault);
  CudaBuffer<double> d_gpu;
  d_gpu.CopyFromCpuVector(d_cpu, cudaStreamDefault);
  CudaDtDxpy(y_gpu.data(),
             d_gpu.data(),
             x_gpu.data(),
             y_gpu.size(),
             cudaStreamDefault);
  y_gpu.CopyToCpu(y_cpu.data(), y_cpu.size());
  EXPECT_DOUBLE_EQ(y_cpu[0], 4.0 + 10.0 * 10.0 * 1.0);
  EXPECT_DOUBLE_EQ(y_cpu[1], 3.0 + 20.0 * 20.0 * 2.0);
  EXPECT_DOUBLE_EQ(y_cpu[2], 2.0 + 30.0 * 30.0 * 3.0);
  EXPECT_DOUBLE_EQ(y_cpu[3], 1.0 + 40.0 * 40.0 * 4.0);
}

#endif  // CERES_NO_CUDA

}  // namespace internal
}  // namespace ceres
