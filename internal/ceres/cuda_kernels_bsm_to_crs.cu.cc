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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "ceres/block_structure.h"
#include "ceres/cuda_kernels.h"
#include "ceres/cuda_kernels_utils.h"

namespace ceres {
namespace internal {

// Fill row block id and nnz for each row using block-sparse structure
// represented by a set of flat arrays.
// Inputs:
// - num_row_blocks: number of row-blocks in block-sparse structure
// - num_col_blocks_e: number of columns in left sub-matrix
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// Outputs:
// - row_nnz_e: row_nnz_e[i + 1] will contain number of non-zeros in i-th row of
// sub-matrix E
// - row_nnz_f: row_nnz_f[i + 1] will contain number of non-zeros in i-th row of
// sub-matrix F
// - row_block_ids: row_block_ids[i] will be set to index of row-block that
// contains i-th row.
// Computation is perform row-block-wise
// For non-partitioned matrices use only F matrix and set num_col_blocks_e to
// zero
template <bool partitioned>
__global__ void RowBlockIdAndNNZ(int num_row_blocks,
                                 int num_col_blocks_e,
                                 const int* __restrict__ row_block_offsets,
                                 const Cell* __restrict__ cells,
                                 const Block* __restrict__ row_blocks,
                                 const Block* __restrict__ col_blocks,
                                 int* __restrict__ row_nnz_e,
                                 int* __restrict__ row_nnz_f,
                                 int* __restrict__ row_block_ids) {
  const int row_block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_block_id > num_row_blocks) {
    // No synchronization is performed in this kernel, thus it is safe to return
    return;
  }
  if (row_block_id == num_row_blocks) {
    // one extra thread sets the first element
    if constexpr (partitioned) {
      row_nnz_e[0] = 0;
    }
    row_nnz_f[0] = 0;
    return;
  }
  const auto& row_block = row_blocks[row_block_id];
  int row_nnz = 0;
  const auto first_cell = cells + row_block_offsets[row_block_id];
  const auto last_cell = cells + row_block_offsets[row_block_id + 1];
  const int first_row = row_block.position;
  const int last_row = first_row + row_block.size;
  // Compute number of non-zeros in row of sub-matrix E
  auto cell = first_cell;
  if constexpr (partitioned) {
    for (; cell < last_cell; ++cell) {
      if (cell->block_id >= num_col_blocks_e) break;
      row_nnz += col_blocks[cell->block_id].size;
    }
    for (int i = first_row; i < last_row; ++i) {
      row_nnz_e[i + 1] = row_nnz;
    }
  }
  row_nnz = 0;
  // Compute number of non-zeros in row of sub-matrix F
  for (; cell < last_cell; ++cell) {
    const auto& col_block = col_blocks[cell->block_id];
    row_nnz += col_block.size;
  }
  for (int i = first_row; i < last_row; ++i) {
    row_nnz_f[i + 1] = row_nnz;
    row_block_ids[i] = row_block_id;
  }
}

// Row-wise creation of [partitioned] CRS structure
// Inputs:
// - num_rows: number of rows in matrix
// - num_col_blocks_e: number of column blocks in left sub-matrix
// - num_cols_e: number of columns in left sub-matrix
// - row_block_offsets: index of the first cell of the row-block; size:
// num_row_blocks + 1
// - cells: cells of block-sparse structure as a continuous array
// - row_blocks: row blocks of block-sparse structure stored sequentially
// - col_blocks: column blocks of block-sparse structure stored sequentially
// - row_block_ids: index of row-block that corresponds to row
// - row_nnz_e: row-index array of CRS structure
// - row_nnz_f: row-index array of CRS structure
// Outputs:
// - cols_e: column-index array of CRS structure
// - cols_f: column-index array of CRS structure
// - permutation: permutation from block-sparse to crs order
// Computaion is performed row-wise
// For non-partitioned matrices use only F matrix and set num_col_blocks_e,
// num_cols_e to zero
template <bool partitioned>
__global__ void ComputeColumnsAndPermutation(
    const int num_rows,
    const int num_col_blocks_e,
    const int num_cols_e,
    const int* __restrict__ row_block_offsets,
    const Cell* __restrict__ cells,
    const Block* __restrict__ row_blocks,
    const Block* __restrict__ col_blocks,
    const int* __restrict__ row_block_ids,
    const int* __restrict__ row_nnz_e,
    int* __restrict__ cols_e,
    const int* __restrict__ row_nnz_f,
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
  if constexpr (partitioned) {
    int crs_position = row_nnz_e[row];
    for (; cell < last_cell; ++cell) {
      if (cell->block_id >= num_col_blocks_e) break;
      const auto& col_block = col_blocks[cell->block_id];
      int column_idx = col_block.position;
      const int col_block_size = col_block.size;
      int bs_position = cell->position + row_in_block * col_block_size;
      // Fill permutation and column indices for each element of row_in_block
      // row of current cell
      for (int i = 0; i < col_block_size; ++i, ++crs_position) {
        permutation[bs_position++] = -(crs_position + 1);
        cols_e[crs_position] = column_idx++;
      }
    }
  }
  // F submatrix: num_cols_e is subtracted from column indices, permutation
  // indices are left as-is
  int crs_position = row_nnz_f[row];
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
                      int* permutation,
                      cudaStream_t stream) {
  // Set number of non-zeros per row in rows array and row to row-block map in
  // row_block_ids array
  const int num_blocks_blockwise = NumBlocks(num_row_blocks + 1);
  int* row_block_ids;
  cudaMallocAsync(&row_block_ids, sizeof(int) * num_rows, stream);
  RowBlockIdAndNNZ<false>
      <<<num_blocks_blockwise, kCudaBlockSize, 0, stream>>>(num_row_blocks,
                                                            0,
                                                            row_block_offsets,
                                                            cells,
                                                            row_blocks,
                                                            col_blocks,
                                                            nullptr,
                                                            rows,
                                                            row_block_ids);
  // Finalize row-index array of CRS strucure by computing prefix sum
  thrust::inclusive_scan(
      thrust::cuda::par.on(stream), rows, rows + num_rows + 1, rows);

  // Fill cols array of CRS structure and permutation from block-sparse to CRS
  const int num_blocks_rowwise = NumBlocks(num_rows);
  ComputeColumnsAndPermutation<false>
      <<<num_blocks_rowwise, kCudaBlockSize, 0, stream>>>(num_rows,
                                                          0,
                                                          0,
                                                          row_block_offsets,
                                                          cells,
                                                          row_blocks,
                                                          col_blocks,
                                                          row_block_ids,
                                                          nullptr,
                                                          nullptr,
                                                          rows,
                                                          cols,
                                                          permutation);
  cudaFreeAsync(row_block_ids, stream);
}

__global__ void ComputeNonZerosInColumnBlockSubMatrixKernel(
    const int num_row_blocks,
    const int num_col_blocks_e,
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
    if (cell->block_id >= num_col_blocks_e) break;
    row_nnz += col_blocks[cell->block_id].size;
  }
  row_block_nnz_e[row_block_id] = row_nnz * row_block.size;
}

int ComputeNonZerosInColumnBlockSubMatrix(const int num_row_blocks,
                                          const int num_col_blocks_e,
                                          const int* row_block_offsets,
                                          const Cell* cells,
                                          const Block* row_blocks,
                                          const Block* col_blocks,
                                          cudaStream_t stream) {
  // Compute row-block-wise non-zero counts
  const int num_blocks_blockwise = NumBlocks(num_row_blocks);
  int* row_block_nnz_e;
  cudaMallocAsync(&row_block_nnz_e, sizeof(int) * num_row_blocks, stream);
  ComputeNonZerosInColumnBlockSubMatrixKernel<<<num_blocks_blockwise,
                                                kCudaBlockSize,
                                                0,
                                                stream>>>(num_row_blocks,
                                                          num_col_blocks_e,
                                                          row_block_offsets,
                                                          cells,
                                                          row_blocks,
                                                          col_blocks,
                                                          row_block_nnz_e);
  // Perform reduction
  const int num_nonzeros_e = thrust::reduce(thrust::cuda::par.on(stream),
                                            row_block_nnz_e,
                                            row_block_nnz_e + num_row_blocks);
  cudaFreeAsync(row_block_nnz_e, stream);
  return num_nonzeros_e;
}

void FillCRSStructurePartitioned(const int num_row_blocks,
                                 const int num_col_blocks_e,
                                 const int num_cols_e,
                                 const int num_rows,
                                 const int* row_block_offsets,
                                 const Cell* cells,
                                 const Block* row_blocks,
                                 const Block* col_blocks,
                                 int* row_nnz_e,
                                 int* cols_e,
                                 int* row_nnz_f,
                                 int* cols_f,
                                 int* permutation,
                                 cudaStream_t stream) {
  // Set number of non-zeros per row in rows array and row to row-block map in
  // row_block_ids array
  const int num_blocks_blockwise = NumBlocks(num_row_blocks + 1);
  int* row_block_ids;
  cudaMallocAsync(&row_block_ids, sizeof(int) * num_rows, stream);
  RowBlockIdAndNNZ<true>
      <<<num_blocks_blockwise, kCudaBlockSize, 0, stream>>>(num_row_blocks,
                                                            num_col_blocks_e,
                                                            row_block_offsets,
                                                            cells,
                                                            row_blocks,
                                                            col_blocks,
                                                            row_nnz_e,
                                                            row_nnz_f,
                                                            row_block_ids);
  // Finalize row-index arrays of CRS strucures by computing prefix sum
  thrust::inclusive_scan(thrust::cuda::par.on(stream),
                         row_nnz_e,
                         row_nnz_e + num_rows + 1,
                         row_nnz_e);
  thrust::inclusive_scan(thrust::cuda::par.on(stream),
                         row_nnz_f,
                         row_nnz_f + num_rows + 1,
                         row_nnz_f);

  // Fill cols array of CRS structure and permutation from block-sparse to CRS
  const int num_blocks_rowwise = NumBlocks(num_rows);
  ComputeColumnsAndPermutation<true>
      <<<num_blocks_rowwise, kCudaBlockSize, 0, stream>>>(num_rows,
                                                          num_col_blocks_e,
                                                          num_cols_e,
                                                          row_block_offsets,
                                                          cells,
                                                          row_blocks,
                                                          col_blocks,
                                                          row_block_ids,
                                                          row_nnz_e,
                                                          cols_e,
                                                          row_nnz_f,
                                                          cols_f,
                                                          permutation);
  cudaFreeAsync(row_block_ids, stream);
}

// Updates values of a pair of CRS matrices with a continuous block of values of
// block-sparse matrix. Magnitude of negative indices is base-1 index in E
// matrix, non-negative indices are base-0 indices in F matrix
// For non-partitioned matrices use only F matrix
template <bool partitioned>
__global__ void PermuteValuesKernel(const int num_values,
                                    const int* permutation,
                                    const double* bsm_values,
                                    double* crs_values_e,
                                    double* crs_values_f) {
  const int value_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (value_id >= num_values) {
    return;
  }
  const auto index = permutation[value_id];
  if (partitioned && index < 0) {
    crs_values_e[-(index + 1)] = bsm_values[value_id];
  } else {
    crs_values_f[index] = bsm_values[value_id];
  }
}

void PermuteValues(const int num_values,
                   const int* permutation,
                   const double* bsm_values,
                   double* crs_values,
                   cudaStream_t stream) {
  const int num_blocks = NumBlocks(num_values);
  PermuteValuesKernel<false><<<num_blocks, kCudaBlockSize, 0, stream>>>(
      num_values, permutation, bsm_values, nullptr, crs_values);
}

void PermuteValuesPartitioned(const int num_values,
                              const int* permutation,
                              const double* bsm_values,
                              double* crs_values_e,
                              double* crs_values_f,
                              cudaStream_t stream) {
  const int num_blocks = NumBlocks(num_values);
  PermuteValuesKernel<true><<<num_blocks, kCudaBlockSize, 0, stream>>>(
      num_values, permutation, bsm_values, crs_values_e, crs_values_f);
}

}  // namespace internal
}  // namespace ceres
