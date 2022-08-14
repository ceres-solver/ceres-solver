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
// Authors: joydeepb@cs.utexas.edu (Joydeep Biswas)

#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/context_impl.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "ceres/internal/config.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "gflags/gflags.h"

#ifndef CERES_NO_CUDA
#include "cuda_runtime.h"
#endif

namespace ceres::internal {

// TODO(Joydeep Biswas): Add a matrix of benchmark sizes to test.

namespace {
// Generate a synthetic BA-style Jacobian with n camera poses, m landmarks, n_d
// parameters per camera, m_d parameters per landmark, and k residuals per
// camera.
// TODO: Unify the synthetic Jacobian generation code with the code from
// schur_eliminator_benchmark.cc since they are very similar.
std::unique_ptr<BlockSparseMatrix> GenerateSyntheticJacobian(
    int n, int m, int n_d, int m_d, int k) {
  static const int kResidualSize = 2;
  CompressedRowBlockStructure* bs = new CompressedRowBlockStructure;
  int c = 0;
  // Add column blocks for each camera.
  for (int i = 0; i < n; ++i) {
    bs->cols.push_back(Block(n_d, c));
    c += n_d;
  }
  // Add column blocks for each landmark.
  for (int i = 0; i < m; ++i) {
    bs->cols.push_back(Block(m_d, c));
    c += m_d;
  }
  // Total number of row blocks = k * n.
  bs->rows.resize(k * n);
  int values_offset = 0;
  std::mt19937 prng;
  std::uniform_real_distribution uniform_0_m(0.0, static_cast<double>(m));
  // Generate structure of the Jacobian.
  // For n cameras:
  for (int i = 0; i < n; ++i) {
    const int camera_block_id = i;
    // For k residuals per camera:
    for (int j = 0; j < k; ++j) {
      // Pick the landmark of the residual randomly from [0, m).
      const int landmark_id = uniform_0_m(prng);
      const int landmark_block_id = n + landmark_id;
      const int row_idx = i * k + j;
      const int row = kResidualSize * row_idx;
      bs->rows[row_idx].block = Block(kResidualSize, row);
      bs->rows[row_idx].cells.resize(2);
      // The camera part of the jacobian of this residual.
      bs->rows[row_idx].cells[0] = Cell(camera_block_id, values_offset);
      values_offset += n_d * kResidualSize;
      // The landmark part of the jacobian of this residual.
      bs->rows[row_idx].cells[1] = Cell(landmark_block_id, values_offset);
      values_offset += m_d * kResidualSize;
    }
  }
  std::unique_ptr<BlockSparseMatrix> jacobian =
      std::make_unique<BlockSparseMatrix>(bs);
  VectorRef(jacobian->mutable_values(), jacobian->num_nonzeros()).setRandom();
  return jacobian;
}
}  // namespace

DEFINE_int32(num_cameras, 1000, "Number of cameras.");
DEFINE_int32(num_landmarks, 10000, "Number of landmarks.");
DEFINE_int32(num_parameters_per_camera, 6, "Number of parameters per camera.");
DEFINE_int32(num_parameters_per_landmark,
             3,
             "Number of parameters per landmark.");
DEFINE_int32(num_residuals_per_camera, 100, "Number of residuals per camera.");

static void BM_CpuRightMultiplyAndAccumulate(benchmark::State& state) {
  // Perform setup here
  std::unique_ptr<BlockSparseMatrix> jacobian =
      GenerateSyntheticJacobian(FLAGS_num_cameras,
                                FLAGS_num_landmarks,
                                FLAGS_num_parameters_per_camera,
                                FLAGS_num_parameters_per_landmark,
                                FLAGS_num_residuals_per_camera);
  Vector x(jacobian->num_cols());
  Vector y(jacobian->num_rows());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    // This code gets timed
    jacobian->RightMultiplyAndAccumulate(x.data(), y.data());
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}

static void BM_CpuLeftMultiplyAndAccumulate(benchmark::State& state) {
  // Perform setup here
  std::unique_ptr<BlockSparseMatrix> jacobian =
      GenerateSyntheticJacobian(FLAGS_num_cameras,
                                FLAGS_num_landmarks,
                                FLAGS_num_parameters_per_camera,
                                FLAGS_num_parameters_per_landmark,
                                FLAGS_num_residuals_per_camera);
  Vector x(jacobian->num_rows());
  Vector y(jacobian->num_cols());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    // This code gets timed
    jacobian->LeftMultiplyAndAccumulate(x.data(), y.data());
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}

BENCHMARK(BM_CpuRightMultiplyAndAccumulate);
BENCHMARK(BM_CpuLeftMultiplyAndAccumulate);

#ifndef CERES_NO_CUDA
static void BM_CudaRightMultiplyAndAccumulate(benchmark::State& state) {
  // Perform setup here
  std::unique_ptr<BlockSparseMatrix> jacobian =
      GenerateSyntheticJacobian(FLAGS_num_cameras,
                                FLAGS_num_landmarks,
                                FLAGS_num_parameters_per_camera,
                                FLAGS_num_parameters_per_landmark,
                                FLAGS_num_residuals_per_camera);
  ContextImpl context;
  std::string message;
  context.InitCUDA(&message);
  CompressedRowSparseMatrix jacobian_crs(
      jacobian->num_rows(), jacobian->num_cols(), jacobian->num_nonzeros());
  jacobian->ToCompressedRowSparseMatrix(&jacobian_crs);
  CudaSparseMatrix cuda_jacobian(&context, jacobian_crs);
  CudaVector cuda_x(&context, 0);
  CudaVector cuda_y(&context, 0);

  Vector x(jacobian->num_cols());
  Vector y(jacobian->num_rows());
  x.setRandom();
  y.setRandom();

  cuda_x.CopyFromCpu(x);
  cuda_y.CopyFromCpu(y);
  double sum = 0;
  for (auto _ : state) {
    // This code gets timed
    cuda_jacobian.RightMultiplyAndAccumulate(cuda_x, &cuda_y);
    sum += cuda_y.Norm();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_NE(sum, 0.0);
}

static void BM_CudaLeftMultiplyAndAccumulate(benchmark::State& state) {
  // Perform setup here
  std::unique_ptr<BlockSparseMatrix> jacobian =
      GenerateSyntheticJacobian(FLAGS_num_cameras,
                                FLAGS_num_landmarks,
                                FLAGS_num_parameters_per_camera,
                                FLAGS_num_parameters_per_landmark,
                                FLAGS_num_residuals_per_camera);
  ContextImpl context;
  std::string message;
  context.InitCUDA(&message);
  CompressedRowSparseMatrix jacobian_crs(
      jacobian->num_rows(), jacobian->num_cols(), jacobian->num_nonzeros());
  jacobian->ToCompressedRowSparseMatrix(&jacobian_crs);
  CudaSparseMatrix cuda_jacobian(&context, jacobian_crs);
  CudaVector cuda_x(&context, 0);
  CudaVector cuda_y(&context, 0);

  Vector x(jacobian->num_rows());
  Vector y(jacobian->num_cols());
  x.setRandom();
  y.setRandom();

  cuda_x.CopyFromCpu(x);
  cuda_y.CopyFromCpu(y);
  double sum = 0;
  for (auto _ : state) {
    // This code gets timed
    cuda_jacobian.LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
    sum += cuda_y.Norm();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_NE(sum, 0.0);
}

BENCHMARK(BM_CudaRightMultiplyAndAccumulate);
BENCHMARK(BM_CudaLeftMultiplyAndAccumulate);
#endif

BENCHMARK_MAIN();

}  // namespace ceres::internal

BENCHMARK_MAIN();
