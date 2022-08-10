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

#ifndef CERES_INTERNAL_CUDA_BUFFER_H_
#define CERES_INTERNAL_CUDA_BUFFER_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <vector>

#include "cuda_runtime.h"
#include "glog/logging.h"

// An encapsulated buffer to maintain GPU memory, and handle transfers between
// GPU and system memory. It is the responsibility of the user to ensure that
// the appropriate GPU device is selected before each subroutine is called. This
// is particularly important when using multiple GPU devices on different CPU
// threads, since active Cuda devices are determined by the cuda runtime on a
// per-thread basis.
template <typename T>
class CudaBuffer {
 public:
  CudaBuffer() = default;
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  ~CudaBuffer() {
    if (data_ != nullptr) {
      CHECK_EQ(cudaFree(data_), cudaSuccess);
    }
  }

  // Grow the GPU memory buffer if needed to accommodate data of the specified
  // size
  void Reserve(const size_t size) {
    if (size > size_) {
      if (data_ != nullptr) {
        CHECK_EQ(cudaFree(data_), cudaSuccess);
      }
      CHECK_EQ(cudaMalloc(&data_, size * sizeof(T)), cudaSuccess)
          << "Failed to allocate " << size * sizeof(T)
          << " bytes of GPU memory";
      size_ = size;
    }
  }

  // Perform an asynchronous copy from CPU memory to GPU memory managed by this
  // CudaBuffer instance using the stream provided.
  void CopyFromCpu(const T* data, const size_t size, cudaStream_t stream) {
    Reserve(size);
    CHECK_EQ(cudaMemcpyAsync(
                 data_, data, size * sizeof(T), cudaMemcpyHostToDevice, stream),
             cudaSuccess);
  }

  // Perform an asynchronous copy from a vector in CPU memory to GPU memory
  // managed by this CudaBuffer instance.
  void CopyFromCpuVector(const std::vector<T>& data, cudaStream_t stream) {
    Reserve(data.size());
    CHECK_EQ(cudaMemcpyAsync(data_,
                             data.data(),
                             data.size() * sizeof(T),
                             cudaMemcpyHostToDevice,
                             stream),
             cudaSuccess);
  }

  // Perform an asynchronous copy from another GPU memory array to the GPU
  // memory managed by this CudaBuffer instance using the stream provided.
  void CopyFromGPUArray(const T* data, const size_t size, cudaStream_t stream) {
    Reserve(size);
    CHECK_EQ(
        cudaMemcpyAsync(
            data_, data, size * sizeof(T), cudaMemcpyDeviceToDevice, stream),
        cudaSuccess);
  }

  // Copy data from the GPU memory managed by this CudaBuffer instance to CPU
  // memory. It is the caller's responsibility to ensure that the CPU memory
  // pointer is valid, i.e. it is not null, and that it points to memory of
  // at least this->size() size. This copy is necessarily synchronous since any
  // potential GPU kernels that may be writing to the buffer must finish before
  // the transfer happens.
  void CopyToCpu(T* data, const size_t size) const {
    CHECK(data_ != nullptr);
    CHECK_EQ(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost),
             cudaSuccess);
  }

  // Copy N items from another GPU memory array to the GPU memory managed by
  // this CudaBuffer instance, growing this buffer's size if needed. This copy
  // is asynchronous, and operates on the stream provided.
  void CopyNItemsFrom(int n, const CudaBuffer<T>& other, cudaStream_t stream) {
    Reserve(n);
    CHECK(other.data_ != nullptr);
    CHECK(data_ != nullptr);
    CHECK_EQ(cudaMemcpyAsync(data_,
                             other.data_,
                             size_ * sizeof(T),
                             cudaMemcpyDeviceToDevice,
                             stream),
             cudaSuccess);
  }

  // Return a pointer to the GPU memory managed by this CudaBuffer instance.
  T* data() { return data_; }
  const T* data() const { return data_; }
  // Return the number of items of type T that can fit in the GPU memory
  // allocated so far by this CudaBuffer instance.
  size_t size() const { return size_; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

#endif  // CERES_NO_CUDA

#endif  // CERES_INTERNAL_CUDA_BUFFER_H_