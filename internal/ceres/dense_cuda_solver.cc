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
//
// A C++ interface to dense CUDA solvers.

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <cstring>
#include <string>
#include <vector>

#include "ceres/dense_cuda_solver.h"
#include "ceres/execution_summary.h"
#include "ceres/linear_solver.h"
#include "ceres/map_util.h"

#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "glog/logging.h"


namespace ceres {
namespace internal {

DenseCudaSolver::DenseCudaSolver() :
    cusolver_handle_(nullptr),
    stream_(nullptr),
    num_rows_(0),
    num_cols_(0),
    host_scratch_(nullptr),
    host_scratch_size_(0),
    gpu_error_(nullptr) {
  CHECK_EQ(cusolverDnCreate(&cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  CHECK_EQ(cusolverDnSetStream(cusolver_handle_, stream_),
      CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaMalloc(&gpu_error_, sizeof(int)), cudaSuccess);
}

DenseCudaSolver::~DenseCudaSolver() {
  CHECK_EQ(cudaFree(gpu_error_), cudaSuccess);
  if (host_scratch_) {
    free(host_scratch_);
  }
  CHECK_EQ(cusolverDnDestroy(cusolver_handle_), CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaStreamDestroy(stream_), cudaSuccess);
}

// Perform Cholesky factorization of a symmetric matrix A.
LinearSolverTerminationType DenseCudaSolver::CholeskyFactorize(
    int num_cols, double* A, std::string* message) {
  // Allocate GPU memory if necessary.
  gpu_a_.Reserve(num_cols * num_cols);
  num_cols_ = num_cols;
  num_rows_ = num_cols;

  // Copy A to GPU.
  gpu_a_.CopyToGpu(A, num_cols * num_cols);

#ifdef CUDA_PRE_11_1
    // CUDA < 11.1 did not have the 64-bit APIs, so use the legacy versions.
  int device_scratch_size = 0;
  CHECK_EQ(cusolverDnDpotrf_bufferSize(cusolver_handle_,
                                      CUBLAS_FILL_MODE_LOWER,
                                      num_cols,
                                      gpu_a_.data(),
                                      num_cols,
                                      &device_scratch_size),
          CUSOLVER_STATUS_SUCCESS);
#else  // CUDA_PRE_11_1
  size_t host_scratch_size = 0;
  size_t device_scratch_size = 0;

  CHECK_EQ(cusolverDnXpotrf_bufferSize(cusolver_handle_,
                                      nullptr,
                                      CUBLAS_FILL_MODE_LOWER,
                                      num_cols,
                                      CUDA_R_64F,
                                      gpu_a_.data(),
                                      num_cols,
                                      CUDA_R_64F,
                                      &device_scratch_size,
                                      &host_scratch_size),
          CUSOLVER_STATUS_SUCCESS);
  // Allocate host scratch memory.
  if (host_scratch_size > host_scratch_size_) {
    CHECK_NOTNULL(realloc(
        reinterpret_cast<void**>(&host_scratch_), host_scratch_size));
    host_scratch_size_ = host_scratch_size;
  }
#endif  // CUDA_PRE_11_1
  // ALlocate GPU scratch memory.
  gpu_scratch_.Reserve(device_scratch_size);

#ifdef CUDA_PRE_11_1
  // CUDA < 11.1 did not have the 64-bit APIs, so use the legacy versions.

  CHECK_EQ(cusolverDnDpotrf(cusolver_handle_,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            gpu_a_.data(),
                            num_cols,
                            reinterpret_cast<double*>(gpu_scratch_.data()),
                            gpu_scratch_.size(),
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
#else  // CUDA_PRE_11_1
  // Perform Cholesky factorization.
  CHECK_EQ(cusolverDnXpotrf(cusolver_handle_,
                            nullptr,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols,
                            CUDA_R_64F,
                            gpu_a_.data(),
                            num_cols,
                            CUDA_R_64F,
                            gpu_scratch_.data(),
                            gpu_scratch_.size(),
                            host_scratch_,
                            host_scratch_size_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
#endif  // CUDA_PRE_11_1
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  int error = 0;
  // Check for errors.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  if (error != 0) {
    *message = "cuSOLVER Cholesky factorization failed.";
    return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

LinearSolverTerminationType DenseCudaSolver::CholeskySolve(
    const double* B, double* X, std::string* message) {
  gpu_b_.CopyToGpu(B, num_cols_);
  // Solve the system.

#ifdef CUDA_PRE_11_1
  CHECK_EQ(cusolverDnDpotrs(cusolver_handle_,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols_,
                            1,
                            gpu_a_.data(),
                            num_cols_,
                            gpu_b_.data(),
                            num_cols_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
#else  // CUDA_PRE_11_1
  CHECK_EQ(cusolverDnXpotrs(cusolver_handle_,
                            nullptr,
                            CUBLAS_FILL_MODE_LOWER,
                            num_cols_,
                            1,
                            CUDA_R_64F,
                            gpu_a_.data(),
                            num_cols_,
                            CUDA_R_64F,
                            gpu_b_.data(),
                            num_cols_,
                            gpu_error_),
          CUSOLVER_STATUS_SUCCESS);
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  CHECK_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#endif  // CUDA_PRE_11_1
  // Check for errors.
  int error = 0;
  // Copy error variable from GPU to host.
  CHECK_EQ(cudaMemcpy(&error,
                      gpu_error_,
                      sizeof(int),
                      cudaMemcpyDeviceToHost),
          cudaSuccess);
  // Copy X from GPU to host.
  gpu_b_.CopyToHost(X, num_cols_);
  if (error != 0) {
    *message = "cuSOLVER Cholesky solve failed.";
    return LinearSolverTerminationType::LINEAR_SOLVER_FATAL_ERROR;
  }
  return LinearSolverTerminationType::LINEAR_SOLVER_SUCCESS;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_CUDA
