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
//
// We use cuSPARSE library for SpMV operations. However, it does not support
// block-sparse format with varying size of the blocks. Thus, we perform the
// following operations in order to compute products of block-sparse matrices
// and dense vectors:
//  - Compute CRS structure from block-sparse structure (once each time we
//  observe a new sparsity pattern)
//  - Compute permutation from block-sparse values to CRS values
//  - Update values in CRS matrix with values of block-sparse matrix
//
// Because new values have to be transfered from host to device via PCIe link,
// we should be able to hide additional costs of permuting values with the help
// of very high memory bandwidth of GPU.

#ifndef CERES_INTERNAL_CUDA_BLOCK_SPARSE_CRS_VIEW_H_
#define CERES_INTERNAL_CUDA_BLOCK_SPARSE_CRS_VIEW_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include <memory>
#include <vector>

#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/cuda_buffer.h"
#include "ceres/cuda_sparse_matrix.h"

namespace ceres::internal {

// CompressedRowBlockStructure is comprised from vector of blocks with each
// block containing a vector of cells, requiring a deep copy in order to be
// transferred to GPU
class CudaBlockSparseStructure {
 public:
  CudaBlockSparseStructure(const CompressedRowBlockStructure& block_structure,
                           ContextImpl* context);

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  int num_cells() const { return num_cells_; }
  int num_nonzeros() const { return num_nonzeros_; }
  int num_row_blocks() const { return num_row_blocks_; }
  int num_col_blocks() const { return num_col_blocks_; }

  // Device pointer to array of num_row_blocks + 1 indices of the first cell of
  // row block
  const int* row_block_offsets() const { return row_block_offsets_.data(); }
  // Device pointer to array of num_cells cells, sorted by row-block
  const Cell* cells() const { return cells_.data(); }
  // Device pointer to array of row blocks
  const Block* row_blocks() const { return row_blocks_.data(); }
  // Device pointer to array of column blocks
  const Block* col_blocks() const { return col_blocks_.data(); }

 private:
  int num_rows_;
  int num_cols_;
  int num_cells_;
  int num_nonzeros_;
  int num_row_blocks_;
  int num_col_blocks_;
  CudaBuffer<int> row_block_offsets_;
  CudaBuffer<Cell> cells_;
  CudaBuffer<Block> row_blocks_;
  CudaBuffer<Block> col_blocks_;
};

// Helper class for streaming continious memory into gpu with simultaneous
// processing
template <typename T, int kValuesPerBatch>
class CudaStreamer {
 public:
  // Only one H->D copy possible at a given time, thus we won't be able to
  // utilize more than 2 streams
  static constexpr int kNumBatches = 2;
  CudaStreamer(ContextImpl* context)
      : context_(context), values_gpu_(context, kValuesPerBatch * kNumBatches) {
    static_assert(ContextImpl::kNumCudaStreams >= kNumBatches);
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
  // function(device_pointer, size_of_batch, offset_of_batch, stream_to_use)
  // after succesful transfer of each batch of data
  template <typename Fun>
  void Stream(const T* from,
              const int num_values,
              Fun&& function,
              bool from_unpinned = true) {
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

      if (from_unpinned) {
        CHECK_EQ(cudaSuccess, cudaEventSynchronize(copy_finished));
        std::copy_n(batch_from, num_values_batch, batch_values_cpu[batch_id]);
        batch_from = batch_values_cpu[batch_id];
      }
      CHECK_EQ(cudaSuccess,
               cudaMemcpyAsync(batch_to,
                               batch_from,
                               sizeof(double) * num_values_batch,
                               cudaMemcpyHostToDevice,
                               stream));
      CHECK_EQ(cudaSuccess, cudaEventRecord(copy_finished, stream));
      function(batch_to, num_values_batch, offset, stream);
      batch_id = (batch_id + 1) % kNumBatches;
    }
    for (int i = 0; i < kNumBatches; ++i) {
      CHECK_EQ(cudaSuccess, cudaStreamSynchronize(streams[i]));
    }
  }

 private:
  CudaBuffer<T> values_gpu_;
  T* values_cpu_pinned_ = nullptr;
  cudaEvent_t copy_finished_[kNumBatches] = {nullptr};
  ContextImpl* context_ = nullptr;
};

// Pre-calculates permutation from block-sparse to CRS order.
// O(nnz) additional memory required, device-side block-sparse structure is
// discarded after construction
class CudaBlockSparseCRSView {
  // Should be large enough to shuffle values as fast as possible
  // Should be small enough to start shuffling values as soon as possible (and
  // minimize memory requirements)
  static constexpr int kValuesPerBatch = 1 * 1024 * 1024;
  using ValueStreamer = CudaStreamer<double, kValuesPerBatch>;

 public:
  CudaBlockSparseCRSView(const BlockSparseMatrix& bsm,
                         ContextImpl* context,
                         bool copy_values);
  const CudaSparseMatrix* Matrix() const { return crs_matrix_.get(); }
  CudaSparseMatrix* Matrix() { return crs_matrix_.get(); }
  void UpdateValues(const BlockSparseMatrix& bsm);

 private:
  std::unique_ptr<CudaSparseMatrix> crs_matrix_;
  CudaBuffer<int> permutation_;
  ValueStreamer streamer_;
};

}  // namespace ceres::internal

#endif
#endif
