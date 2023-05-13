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

//#include "ceres/cuda_kernels.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "block_structure.h"
#include "cuda_runtime.h"

namespace ceres {
namespace internal {

// As the CUDA Toolkit documentation says, "although arbitrary in this case, is
// a common choice". This is determined by the warp size, max block size, and
// multiprocessor sizes of recent GPUs. For complex kernels with significant
// register usage and unusual memory patterns, the occupancy calculator API
// might provide better performance. See "Occupancy Calculator" under the CUDA
// toolkit documentation.
constexpr int kCudaBlockSize = 256;
inline int NumBlocks(int size) {
  return (size + kCudaBlockSize - 1) / kCudaBlockSize;
}

template <typename SrcType, typename DstType>
__global__ void TypeConversionKernel(const SrcType* __restrict__ input,
                                     DstType* __restrict__ output,
                                     const int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = static_cast<DstType>(input[i]);
  }
}

void CudaFP64ToFP32(const double* input,
                    float* output,
                    const int size,
                    cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  TypeConversionKernel<double, float>
      <<<num_blocks, kCudaBlockSize, 0, stream>>>(input, output, size);
}

void CudaFP32ToFP64(const float* input,
                    double* output,
                    const int size,
                    cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  TypeConversionKernel<float, double>
      <<<num_blocks, kCudaBlockSize, 0, stream>>>(input, output, size);
}

template <typename T>
__global__ void SetZeroKernel(T* __restrict__ output, const int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = T(0.0);
  }
}

void CudaSetZeroFP32(float* output, const int size, cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  SetZeroKernel<float><<<num_blocks, kCudaBlockSize, 0, stream>>>(output, size);
}

void CudaSetZeroFP64(double* output, const int size, cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  SetZeroKernel<double>
      <<<num_blocks, kCudaBlockSize, 0, stream>>>(output, size);
}

template <typename SrcType, typename DstType>
__global__ void XPlusEqualsYKernel(DstType* __restrict__ x,
                                   const SrcType* __restrict__ y,
                                   const int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    x[i] = x[i] + DstType(y[i]);
  }
}

void CudaDsxpy(double* x, float* y, const int size, cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  XPlusEqualsYKernel<float, double>
      <<<num_blocks, kCudaBlockSize, 0, stream>>>(x, y, size);
}

__global__ void CudaDtDxpyKernel(double* __restrict__ y,
                                 const double* D,
                                 const double* __restrict__ x,
                                 const int size) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    y[i] = y[i] + D[i] * D[i] * x[i];
  }
}

void CudaDtDxpy(double* y,
                const double* D,
                const double* x,
                const int size,
                cudaStream_t stream) {
  const int num_blocks = NumBlocks(size);
  CudaDtDxpyKernel<<<num_blocks, kCudaBlockSize, 0, stream>>>(y, D, x, size);
}

// Fill row block id and nnz for each row using block-sparse structure
// represented by a set of flat arrays.
// Inputs:
// - num_row_blocks: number of row-blocks in block-sparse structure
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// Outputs:
// - rows: rows[i + 1] will contain number of non-zeros in i-th row, rows[0]
// will be set to 0; rows are filled with a shift by one element in order
// to obtain row-index array of CRS matrix with a inclusive scan afterwards
// - row_block_ids: row_block_ids[i] will be set to index of row-block that
// contains i-th row.
// Computation is perform row-block-wise
__global__ void RowBlockIdAndNNZ(int num_row_blocks,
                                 const int* __restrict__ row_block_offsets,
                                 const Cell* __restrict__ cells,
                                 const Block* __restrict__ row_blocks,
                                 const Block* __restrict__ col_blocks,
                                 int* __restrict__ rows,
                                 int* __restrict__ row_block_ids) {
  const int row_block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_block_id > num_row_blocks) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  if (row_block_id == num_row_blocks) {
    // one extra thread sets the first element
    rows[0] = 0;
    return;
  }
  const auto& row_block = row_blocks[row_block_id];
  int row_nnz = 0;
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  for (auto cell = first_cell; cell < last_cell; ++cell) {
    row_nnz += col_blocks[cell->block_id].size;
  }
  const int first_row = row_block.position;
  const int last_row = first_row + row_block.size;
  for (int i = first_row; i < last_row; ++i) {
    rows[i + 1] = row_nnz;
    row_block_ids[i] = row_block_id;
  }
}

// Fill row block id and nnz for each row using block-sparse structure
// represented by a set of flat arrays.
// Inputs:
// - num_row_blocks: number of row-blocks in block-sparse structure
// - num_cols_e: number of columns in left sub-matrix
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// Outputs:
// - rows_e: rows_e[i + 1] will contain number of non-zeros in i-th row of
// sub-matrix E
// - rows_f: rows_f[i + 1] will contain number of non-zeros in i-th row of
// sub-matrix F
// - row_block_ids: row_block_ids[i] will be set to index of row-block that
// contains i-th row.
// Computation is perform row-block-wise
__global__ void RowBlockIdAndNNZPartitioned(
    int num_row_blocks,
    int num_cols_e,
    const int* __restrict__ row_block_offsets,
    const Cell* __restrict__ cells,
    const Block* __restrict__ row_blocks,
    const Block* __restrict__ col_blocks,
    int* __restrict__ rows_e,
    int* __restrict__ rows_f,
    int* __restrict__ row_block_ids) {
  const int row_block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_block_id > num_row_blocks) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  if (row_block_id == num_row_blocks) {
    // one extra thread sets the first element
    rows_e[0] = 0;
    rows_f[0] = 0;
    return;
  }
  const auto& row_block = row_blocks[row_block_id];
  int row_nnz = 0;
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  // Compute number of non-zeros in row of sub-matrix E
  auto cell = first_cell;
  for (; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    if (col_block.position >= num_cols_e) break;
    row_nnz += col_block.size;
  }
  const int first_row = row_block.position;
  const int last_row = first_row + row_block.size;
  for (int i = first_row; i < last_row; ++i) {
    rows_e[i + 1] = row_nnz;
    row_block_ids[i] = row_block_id;
  }
  row_nnz = 0;
  // Compute number of non-zeros in row of sub-matrix F
  for (; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    row_nnz += col_block.size;
  }
  for (int i = first_row; i < last_row; ++i) {
    rows_f[i + 1] = row_nnz;
  }
}

// Row-wise creation of CRS structure
// Inputs:
// - num_rows: number of rows in matrix
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// - row_block_ids: index of row-block that corresponds to row
// - rows: row-index array of CRS structure
// Outputs:
// - cols: column-index array of CRS structure
// - permutation: permutation from block-sparse to crs order
// Computaion is perform row-wise
__global__ void ComputeColumnsAndPermutation(
    int num_rows,
    const int* __restrict__ row_block_offsets,
    const Cell* __restrict__ cells,
    const Block* __restrict__ row_blocks,
    const Block* __restrict__ col_blocks,
    const int* __restrict__ row_block_ids,
    const int* __restrict__ rows,
    int* __restrict__ cols,
    int* __restrict__ permutation) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  const int row_block_id = row_block_ids[row];
  // zero-based index of row in row-block
  const int row_in_block = row - row_blocks[row_block_id].position;
  // position in crs matrix
  int crs_position = rows[row];
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  // For reach cell of row-block only current row is being filled
  for (auto cell = first_cell; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    const int col_block_size = col_block.size;
    int column_idx = col_block.position;
    int bs_position = cell->position + row_in_block * col_block_size;
    // Fill permutation and column indices for each element of row_in_block
    // row of current cell
    for (int i = 0; i < col_block_size; ++i, ++crs_position) {
      permutation[bs_position++] = crs_position;
      cols[crs_position] = column_idx++;
    }
  }
}

// Row-wise creation of partitioned CRS structure
// Inputs:
// - num_rows: number of rows in matrix
// - num_cols_e: number of columns in left sub-matrix
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// - row_block_ids: index of row-block that corresponds to row
// - rows_e: row-index array of CRS structure
// - rows_f: row-index array of CRS structure
// Outputs:
// - cols_e: column-index array of CRS structure
// - cols_f: column-index array of CRS structure
// - permutation: permutation from block-sparse to crs order
// Computaion is perform row-wise
__global__ void ComputeColumnsAndPermutation(
    const int num_rows,
    const int num_cols_e,
    const int* __restrict__ row_block_offsets,
    const Cell* __restrict__ cells,
    const Block* __restrict__ row_blocks,
    const Block* __restrict__ col_blocks,
    const int* __restrict__ row_block_ids,
    const int* __restrict__ rows_e,
    int* __restrict__ cols_e,
    const int* __restrict__ rows_f,
    int* __restrict__ cols_f,
    int* __restrict__ permutation) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  const int row_block_id = row_block_ids[row];
  // zero-based index of row in row-block
  const int row_in_block = row - row_blocks[row_block_id].position;
  // position in crs matrix
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  // For reach cell of row-block only current row is being filled
  // E submatrix: column indices are left as is, permutation indices are
  // negative
  auto cell = first_cell;
  int crs_position = rows_e[row];
  for (; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    int column_idx = col_block.position;
    if (column_idx >= num_cols_e) break;
    const int col_block_size = col_block.size;
    int bs_position = cell->position + row_in_block * col_block_size;
    // Fill permutation and column indices for each element of row_in_block
    // row of current cell
    for (int i = 0; i < col_block_size; ++i, ++crs_position) {
      permutation[bs_position++] = -(crs_position + 1);
      cols_e[crs_position] = column_idx++;
    }
  }
  // F submatrix: num_cols_e is subtracted from column indices, permutation
  // indices are left as-is
  crs_position = rows_f[row];
  for (; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    const int col_block_size = col_block.size;
    int column_idx = col_block.position - num_cols_e;
    int bs_position = cell->position + row_in_block * col_block_size;
    // Fill permutation and column indices for each element of row_in_block
    // row of current cell
    for (int i = 0; i < col_block_size; ++i, ++crs_position) {
      permutation[bs_position++] = crs_position;
      cols_f[crs_position] = column_idx++;
    }
  }
}

void FillCRSStructure(const int num_row_blocks,
                      const int num_rows,
                      const int* row_block_offsets,
                      const Cell* cells,
                      const Block* row_blocks,
                      const Block* col_blocks,
                      int* rows,
                      int* cols,
                      int* row_block_ids,
                      int* permutation,
                      cudaStream_t stream) {
  // Set number of non-zeros per row in rows array and row to row-block map in
  // row_block_ids array
  const int num_blocks_blockwise = NumBlocks(num_row_blocks + 1);
  RowBlockIdAndNNZ<<<num_blocks_blockwise, kCudaBlockSize, 0, stream>>>(
      num_row_blocks,
      row_block_offsets,
      cells,
      row_blocks,
      col_blocks,
      rows,
      row_block_ids);
  // Finalize row-index array of CRS strucure by computing prefix sum
  thrust::inclusive_scan(
      thrust::cuda::par.on(stream), rows, rows + num_rows + 1, rows);

  // Fill cols array of CRS structure and permutation from block-sparse to CRS
  const int num_blocks_rowwise = NumBlocks(num_rows);
  ComputeColumnsAndPermutation<<<num_blocks_rowwise,
                                 kCudaBlockSize,
                                 0,
                                 stream>>>(num_rows,
                                           row_block_offsets,
                                           cells,
                                           row_blocks,
                                           col_blocks,
                                           row_block_ids,
                                           rows,
                                           cols,
                                           permutation);
}

__global__ void ComputeNonZerosInColumnBlockSubMatrixKernel(
    const int num_row_blocks,
    const int num_cols_e,
    const int* __restrict__ row_block_offsets,
    const Cell* __restrict__ cells,
    const Block* __restrict__ row_blocks,
    const Block* __restrict__ col_blocks,
    int* __restrict__ row_block_nnz_e) {
  const int row_block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_block_id >= num_row_blocks) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  const auto& row_block = row_blocks[row_block_id];
  int row_nnz = 0;
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  for (auto cell = first_cell; cell < last_cell; ++cell) {
    if (col_blocks[cell->block_id].position >= num_cols_e) break;
    row_nnz += col_blocks[cell->block_id].size;
  }
  row_block_nnz_e[row_block_id] = row_nnz * row_block.size;
}

int ComputeNonZerosInColumnBlockSubMatrix(const int num_row_blocks,
                                          const int num_cols_e,
                                          const int* row_block_offsets,
                                          const Cell* cells,
                                          const Block* row_blocks,
                                          const Block* col_blocks,
                                          int* row_block_nnz_e,
                                          cudaStream_t stream) {
  // Compute row-block-wise non-zero counts
  const int num_blocks_blockwise = NumBlocks(num_row_blocks);
  ComputeNonZerosInColumnBlockSubMatrixKernel<<<num_blocks_blockwise,
                                                kCudaBlockSize,
                                                0,
                                                stream>>>(num_row_blocks,
                                                          num_cols_e,
                                                          row_block_offsets,
                                                          cells,
                                                          row_blocks,
                                                          col_blocks,
                                                          row_block_nnz_e);
  // Perform reduction
  return thrust::reduce(thrust::cuda::par.on(stream),
                        row_block_nnz_e,
                        row_block_nnz_e + num_row_blocks);
}

void FillCRSStructurePartitioned(const int num_row_blocks,
                                 const int num_cols_e,
                                 const int num_rows,
                                 const int* row_block_offsets,
                                 const Cell* cells,
                                 const Block* row_blocks,
                                 const Block* col_blocks,
                                 int* rows_e,
                                 int* cols_e,
                                 int* rows_f,
                                 int* cols_f,
                                 int* row_block_ids,
                                 int* permutation,
                                 cudaStream_t stream) {
  // Set number of non-zeros per row in rows array and row to row-block map in
  // row_block_ids array
  const int num_blocks_blockwise = NumBlocks(num_row_blocks + 1);
  RowBlockIdAndNNZPartitioned<<<num_blocks_blockwise,
                                kCudaBlockSize,
                                0,
                                stream>>>(num_row_blocks,
                                          num_cols_e,
                                          row_block_offsets,
                                          cells,
                                          row_blocks,
                                          col_blocks,
                                          rows_e,
                                          rows_f,
                                          row_block_ids);
  // Finalize row-index arrays of CRS strucures by computing prefix sum
  thrust::inclusive_scan(
      thrust::cuda::par.on(stream), rows_e, rows_e + num_rows + 1, rows_e);
  thrust::inclusive_scan(
      thrust::cuda::par.on(stream), rows_f, rows_f + num_rows + 1, rows_f);

  // Fill cols array of CRS structure and permutation from block-sparse to CRS
  const int num_blocks_rowwise = NumBlocks(num_rows);
  ComputeColumnsAndPermutation<<<num_blocks_rowwise,
                                 kCudaBlockSize,
                                 0,
                                 stream>>>(num_rows,
                                           num_cols_e,
                                           row_block_offsets,
                                           cells,
                                           row_blocks,
                                           col_blocks,
                                           row_block_ids,
                                           rows_e,
                                           cols_e,
                                           rows_f,
                                           cols_f,
                                           permutation);
}

// Updates values of CRS matrix with a continuous block of values of
// block-sparse matrix. With permutation[i] being an index of i-th value of
// block-sparse matrix in values of CRS matrix updating values is quite
// efficient when performed element-wise (reads of permutation and values arrays
// are coalesced when offset is divisable by 16)
__global__ void PermuteValuesKernel(const int offset,
                                    const int num_values,
                                    const int* permutation,
                                    const double* block_sparse_values,
                                    double* crs_values) {
  const int value_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (value_id < num_values) {
    crs_values[permutation[offset + value_id]] = block_sparse_values[value_id];
  }
}

void PermuteValues(const int offset,
                   const int num_values,
                   const int* permutation,
                   const double* block_sparse_values,
                   double* crs_values,
                   cudaStream_t stream) {
  const int num_blocks = NumBlocks(num_values);
  PermuteValuesKernel<<<num_blocks, kCudaBlockSize, 0, stream>>>(
      offset, num_values, permutation, block_sparse_values, crs_values);
}

// Updates values of a pair of CRS matrices with a continuous block of values of
// block-sparse matrix. Magnitude of negative indices is base-1 index in E
// matrix, non-negative indices are base-0 indices in F matrix
__global__ void PermuteValuesPartitionedKernel(
    const int offset,
    const int num_values,
    const int* permutation,
    const double* block_sparse_values,
    double* crs_values_e,
    double* crs_values_f) {
  const int value_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (value_id >= num_values) {
    return;
  }
  const auto index = permutation[offset + value_id];
  if (index < 0) {
    crs_values_e[-(index + 1)] = block_sparse_values[value_id];
  } else {
    crs_values_f[index] = block_sparse_values[value_id];
  }
}

void PermuteValuesPartitioned(const int offset,
                              const int num_values,
                              const int* permutation,
                              const double* block_sparse_values,
                              double* crs_values_e,
                              double* crs_values_f,
                              cudaStream_t stream) {
  const int num_blocks = NumBlocks(num_values);
  PermuteValuesPartitionedKernel<<<num_blocks, kCudaBlockSize, 0, stream>>>(
      offset,
      num_values,
      permutation,
      block_sparse_values,
      crs_values_e,
      crs_values_f);
}

}  // namespace internal
}  // namespace ceres
