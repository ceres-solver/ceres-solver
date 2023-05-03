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

#ifndef CERES_INTERNAL_CUDA_STREAMER_H_
#define CERES_INTERNAL_CUDA_STREAMER_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_buffer.h"

namespace ceres::internal {

// Most contemporary CUDA devices are capable of simultaneous code execution and
// host-to-device transfer. This class copies batches of data to GPU memory and
// executes processing of copied data in parallel (asynchronously).
// Data is copied to a fixed-size buffer on GPU (containing at most
// kMaxTemporaryArraySize values), and this memory is re-used when the previous
// batch of values is processed by user-provided callback
// Host-to-device copy uses a temporary buffer if required. Each batch of values
// has size of kValuesPerBatch, except the last one.
template <typename T>
class CudaStreamer {
 public:
  // As long as one host-to-device copy is able to reach peak bandwidth and
  // hardware supports only one host-to-device copy at a given time, only two
  // streams are  utilized
  static constexpr int kNumBatches = 2;
  // kMaxTemporaryArraySize is the maximal size (in elements of type T) of array
  // to be pre-allocated in gpu memory. The size of array determines size of
  // batch of values for simultaneous copying and processing. It should be large
  // enough to allow highly-parallel execution of user kernels; making it too
  // large increases latency.
  CudaStreamer(ContextImpl* context, const int kMaxTemporaryArraySize)
      : kValuesPerBatch(kMaxTemporaryArraySize / kNumBatches),
        context_(context),
        values_gpu_(context, kValuesPerBatch * kNumBatches) {
    static_assert(ContextImpl::kNumCudaStreams >= kNumBatches);
    CHECK_GE(kMaxTemporaryArraySize, kNumBatches);
    CHECK_EQ(cudaSuccess,
             cudaHostAlloc(&values_cpu_pinned_,
                           sizeof(T) * kValuesPerBatch * kNumBatches,
                           cudaHostAllocWriteCombined));
    for (auto& e : copy_finished_) {
      CHECK_EQ(cudaSuccess,
               cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
  }

  CudaStreamer(const CudaStreamer&) = delete;

  ~CudaStreamer() {
    CHECK_EQ(cudaSuccess, cudaFreeHost(values_cpu_pinned_));
    for (auto& e : copy_finished_) {
      CHECK_EQ(cudaSuccess, cudaEventDestroy(e));
    }
  }

  // Transfer num_values at host-memory pointer from, calling
  // callback(device_pointer, size_of_batch, offset_of_batch, stream_to_use)
  // after scheduling transfer of each batch of data. User-provided callback
  // should perform processing of data at device_pointer only in
  // DefaultStream()to_use stream (device_pointer will be re-used in the next
  // callback invocation with the same DefaultStream()to_use).
  template <typename Fun>
  void CopyToGpu(const T* from, const int num_values, Fun&& callback) {
    // This synchronization is not required in some cases, but we perform it in
    // order to avoid situation when user call-back depends on data that is
    // still to be computed in default stream
    CHECK_EQ(cudaSuccess, cudaStreamSynchronize(context_->DefaultStream()));

    const bool requires_copy = HostMemoryRequiresCopy(from);
    T* batch_values_gpu[kNumBatches];
    T* batch_values_cpu[kNumBatches];
    auto streams = context_->streams_;
    for (int i = 0; i < kNumBatches; ++i) {
      batch_values_gpu[i] = values_gpu_.data() + kValuesPerBatch * i;
      batch_values_cpu[i] = values_cpu_pinned_ + kValuesPerBatch * i;
    }
    int batch_id = 0;
    for (int offset = 0; offset < num_values; offset += kValuesPerBatch) {
      const int num_values_batch =
          std::min(num_values - offset, kValuesPerBatch);
      const T* batch_from = from + offset;
      T* batch_to = batch_values_gpu[batch_id];
      auto stream = streams[batch_id];
      auto copy_finished = copy_finished_[batch_id];

      if (requires_copy) {
        // Copying values to a temporary buffer should be started only after the
        // previous copy from temporary buffer to device is completed.
        CHECK_EQ(cudaSuccess, cudaEventSynchronize(copy_finished));
        std::copy_n(batch_from, num_values_batch, batch_values_cpu[batch_id]);
        batch_from = batch_values_cpu[batch_id];
      }
      CHECK_EQ(cudaSuccess,
               cudaMemcpyAsync(batch_to,
                               batch_from,
                               sizeof(T) * num_values_batch,
                               cudaMemcpyHostToDevice,
                               stream));
      // Next copy to a temporary buffer can start straight after asynchronous
      // copy is completed (and might be started before kernels asynchronously
      // executed in stream by user-supplied callback are completed).
      CHECK_EQ(cudaSuccess, cudaEventRecord(copy_finished, stream));
      callback(batch_to, num_values_batch, offset, stream);
      batch_id = (batch_id + 1) % kNumBatches;
    }
    // Explicitly synchronize on all CUDA streams that were utilized.
    for (int i = 0; i < kNumBatches; ++i) {
      CHECK_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
    }
  }

 private:
  static bool HostMemoryRequiresCopy(const void* ptr) {
    cudaPointerAttributes attributes;
#if CUDART_VERSION < 11000
    auto status = cudaPointerGetAttributes(&attributes, ptr);
    if (status == cudaErrorInvalidValue) {
      return true;
    }
    CHECK_EQ(status, cudaSuccess);
#else
    CHECK_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, ptr));
#endif
    return attributes.type == cudaMemoryTypeHost;
  }
  const int kValuesPerBatch;
  ContextImpl* context_ = nullptr;
  CudaBuffer<T> values_gpu_;
  T* values_cpu_pinned_ = nullptr;
  cudaEvent_t copy_finished_[kNumBatches] = {nullptr};
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_STREAMER_H_
