// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

#include "ceres/context_impl.h"

#include <string>

#include "ceres/internal/config.h"
#include "ceres/stringprintf.h"
#include "ceres/wall_time.h"

#ifndef CERES_NO_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"
#endif  // CERES_NO_CUDA

namespace ceres::internal {

ContextImpl::ContextImpl() = default;

#ifndef CERES_NO_CUDA
void ContextImpl::TearDown() {
  if (cusolver_handle_ != nullptr) {
    cusolverDnDestroy(cusolver_handle_);
    cusolver_handle_ = nullptr;
  }
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
    cublas_handle_ = nullptr;
  }
  if (cusolver_handle_ != nullptr) {
    cusparseDestroy(cusparse_handle_);
    cusparse_handle_ = nullptr;
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  is_cuda_initialized_ = false;
}

void ContextImpl::DetectGpuAndCudaVersions() {
  CHECK_EQ(cudaGetDevice(&gpu_device_id_in_use_), cudaSuccess);
  int cuda_version;
  CHECK_EQ(cudaRuntimeGetVersion(&cuda_version), cudaSuccess);
  cuda_version_major_ = cuda_version / 1000;
  cuda_version_minor_ = (cuda_version % 1000) / 10;
  CHECK_EQ(cudaGetDeviceProperties(&gpu_device_properties_,
                                   gpu_device_id_in_use_), cudaSuccess);
}

std::string ContextImpl::GetCudaDevicePropertiesString() const {
  return ceres::internal::StringPrintf(
      "======================= CUDA Device Properties ======================\n"
      "Cuda version         : %d.%d\n"
      "Device ID            : %d\n"
      "Device name          : %s\n"
      "Total global memory  : %.3f GB\n"
      "Compute capability   : %d.%d\n"
      "Warp size            : %d\n"
      "Max threads per block: %d\n"
      "Max threads per dim  : %d %d %d\n"
      "Max grid size        : %d %d %d\n"
      "Multiprocessor count : %d\n"
      "====================================================================",
      cuda_version_major_,
      cuda_version_minor_,
      gpu_device_properties_.device,
      gpu_device_properties_.name,
      gpu_device_properties_.totalGlobalMem / 1024.0 / 1024.0 / 1024.0,
      gpu_device_properties_.major,
      gpu_device_properties_.minor,
      gpu_device_properties_.warpSize,
      gpu_device_properties_.maxThreadsPerBlock,
      gpu_device_properties_.maxThreadsDim[0],
      gpu_device_properties_.maxThreadsDim[1],
      gpu_device_properties_.maxThreadsDim[2],
      gpu_device_properties_.maxGridSize[0],
      gpu_device_properties_.maxGridSize[1],
      gpu_device_properties_.maxGridSize[2],
      gpu_device_properties_.multiProcessorCount);
}

bool ContextImpl::InitCUDA(std::string* message) {
  if (is_cuda_initialized_) {
    return true;
  }
  DetectGpuAndCudaVersions();
  VLOG(3) << "\n" << GetCudaDevicePropertiesString();
  EventLogger event_logger("InitCuda");
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    *message = "cuBLAS::cublasCreate failed.";
    cublas_handle_ = nullptr;
    return false;
  }
  event_logger.AddEvent("cublasCreate");
  if (cusolverDnCreate(&cusolver_handle_) != CUSOLVER_STATUS_SUCCESS) {
    *message = "cuSolverDN::cusolverDnCreate failed.";
    TearDown();
    return false;
  }
  event_logger.AddEvent("cusolverDnCreate");
  if (cusparseCreate(&cusparse_handle_) != CUSPARSE_STATUS_SUCCESS) {
    *message = "cuSPARSE::cusparseCreate failed.";
    TearDown();
    return false;
  }
  event_logger.AddEvent("cusparseCreate");
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
      cudaSuccess) {
    *message = "CUDA::cudaStreamCreateWithFlags failed.";
    TearDown();
    return false;
  }
  event_logger.AddEvent("cudaStreamCreateWithFlags");
  if (cusolverDnSetStream(cusolver_handle_, stream_) !=
          CUSOLVER_STATUS_SUCCESS ||
      cublasSetStream(cublas_handle_, stream_) != CUBLAS_STATUS_SUCCESS ||
      cusparseSetStream(cusparse_handle_, stream_) != CUSPARSE_STATUS_SUCCESS) {
    *message = "CUDA [Solver|BLAS|Sparse] SetStream failed.";
    TearDown();
    return false;
  }
  event_logger.AddEvent("SetStream");
  is_cuda_initialized_ = true;
  return true;
}
#endif  // CERES_NO_CUDA

ContextImpl::~ContextImpl() {
#ifndef CERES_NO_CUDA
  TearDown();
#endif  // CERES_NO_CUDA
}
void ContextImpl::EnsureMinimumThreads(int num_threads) {
#ifdef CERES_USE_CXX_THREADS
  thread_pool.Resize(num_threads);
#endif  // CERES_USE_CXX_THREADS
}
}  // namespace ceres::internal
