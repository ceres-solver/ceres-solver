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

#ifndef CERES_INTERNAL_CONTEXT_IMPL_H_
#define CERES_INTERNAL_CONTEXT_IMPL_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <string>

#include "ceres/context.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"

#ifndef CERES_NO_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"
#include "cusparse.h"
#endif  // CERES_NO_CUDA

#ifdef CERES_USE_CXX_THREADS
#include "ceres/thread_pool.h"
#endif  // CERES_USE_CXX_THREADS

namespace ceres::internal {

class CERES_NO_EXPORT ContextImpl final : public Context {
 public:
  ContextImpl();
  ~ContextImpl() override;
  ContextImpl(const ContextImpl&) = delete;
  void operator=(const ContextImpl&) = delete;

  // When compiled with C++ threading support, resize the thread pool to have
  // at min(num_thread, num_hardware_threads) where num_hardware_threads is
  // defined by the hardware.  Otherwise this call is a no-op.
  void EnsureMinimumThreads(int num_threads);

#ifdef CERES_USE_CXX_THREADS
  ThreadPool thread_pool;
#endif  // CERES_USE_CXX_THREADS

#ifndef CERES_NO_CUDA
  // Note on Ceres' use of CUDA Devices on multi-GPU systems:
  // 1. On a multi-GPU system, if nothing special is done, the "default" CUDA
  //    device will be used, which is device 0.
  // 2. If the user masks out GPUs using the  CUDA_VISIBLE_DEVICES  environment
  //    variable, Ceres will still use device 0 visible to the program, but
  //    device 0 will be the first GPU indicated in the environment variable.
  // 3. If the user explicitly selects a GPU in the host process before calling
  //    Ceres, Ceres will use that GPU.

  // Note on Ceres' use of CUDA Streams:
  // All operations on the GPU are performed using a single stream. This ensures
  // that the order of operations are stream-ordered, but we do not need to
  // explicitly synchronize the stream at the end of every operation. Stream
  // synchronization occurs only before GPU to CPU transfers, and is handled by
  // CudaBuffer.

  // Initializes cuBLAS, cuSOLVER, and cuSPARSE contexts, creates an
  // asynchronous CUDA stream, and associates the stream with the contexts.
  // Returns true iff initialization was successful, else it returns false and a
  // human-readable error message is returned.
  bool InitCuda(std::string* message);
  void TearDown();
  inline bool IsCudaInitialized() const { return is_cuda_initialized_; }
  // Returns a human-readable string describing the capabilities of the current
  // CUDA device. CudaConfigAsString can only be called after InitCuda has been
  // called.
  std::string CudaConfigAsString() const;
  // Returns the number of bytes of available global memory on the current CUDA
  // device. If it is called before InitCuda, it returns 0.
  size_t GpuMemoryAvailable() const;

  cusolverDnHandle_t cusolver_handle_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cusparseHandle_t cusparse_handle_ = nullptr;
  bool is_cuda_initialized_ = false;
  int gpu_device_id_in_use_ = -1;
  cudaDeviceProp gpu_device_properties_;
  int cuda_version_major_ = 0;
  int cuda_version_minor_ = 0;
#endif  // CERES_NO_CUDA
};

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_CONTEXT_IMPL_H_
