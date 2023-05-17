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

#ifndef CERES_INTERNAL_CUDA_BLOCK_STRUCTURE_H_
#define CERES_INTERNAL_CUDA_BLOCK_STRUCTURE_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA

#include "ceres/block_structure.h"
#include "ceres/cuda_buffer.h"

namespace ceres::internal {
class CudaBlockStructureTest;

// This class stores a read-only block-sparse structure in gpu memory.
// Invariants are the same as those of CompressedRowBlockStructure.
// In order to simplify allocation and copying data to gpu, cells from all
// row-blocks are stored in a single array sequentially. Additional array
// row_block_offsets of size num_row_blocks + 1 allows to identify range of
// cells corresponding to a row-block.
// Cells corresponding to i-th row-block are stored in sub-array
// cells[row_block_offsets[i]; ... row_block_offsets[i + 1] - 1], and their
// order is preserved.
class CERES_NO_EXPORT CudaBlockSparseStructure {
 public:
  // CompressedRowBlockStructure is contains a vector of CompressedLists, with
  // each CompressedList containing a vector of Cells. We precompute a flat
  // array of cells on cpu and transfer it to the gpu.
  CudaBlockSparseStructure(const CompressedRowBlockStructure& block_structure,
                           ContextImpl* context);

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  int num_cells() const { return num_cells_; }
  int num_nonzeros() const { return num_nonzeros_; }
  int num_row_blocks() const { return num_row_blocks_; }
  int num_col_blocks() const { return num_col_blocks_; }

  // Inferred structure details
  bool is_partitioned() const { return is_partitioned_; }
  int num_nonzeros_e() const { return num_nonzeros_e_; }
  int num_nonzeros_f() const { return num_nonzeros_f_; }
  bool e_is_crs_compatible() const { return e_is_crs_compatible_; }
  bool f_is_crs_compatible() const { return f_is_crs_compatible_; }

  // Device pointer to array of num_row_blocks + 1 indices of the first cell of
  // row block
  const int* row_block_offsets() const { return row_block_offsets_.data(); }
  // Device pointer to array of num_cells cells, sorted by row-block
  const Cell* cells() const { return cells_.data(); }
  // Device pointer to array of row blocks
  const Block* row_blocks() const { return row_blocks_.data(); }
  // Device pointer to array of column blocks
  const Block* col_blocks() const { return col_blocks_.data(); }
  // Position of the first cell of E sub-matrix in row-block (only populated for
  // partitioned matrices)
  const int* first_cell_pos_e() const { return first_cell_pos_e_.data(); }
  // Position of the first cell of F sub-matrix in row-block (only populated for
  // partitioned matrices)
  const int* first_cell_pos_f() const { return first_cell_pos_f_.data(); }

 private:
  int num_rows_;
  int num_cols_;
  int num_cells_;
  int num_nonzeros_;
  int num_row_blocks_;
  int num_col_blocks_;
  int num_nonzeros_e_;
  int num_nonzeros_f_;
  bool is_partitioned_;
  bool e_is_crs_compatible_;
  bool f_is_crs_compatible_;
  CudaBuffer<int> row_block_offsets_;
  CudaBuffer<int> first_cell_pos_e_;
  CudaBuffer<int> first_cell_pos_f_;
  CudaBuffer<Cell> cells_;
  CudaBuffer<Block> row_blocks_;
  CudaBuffer<Block> col_blocks_;
  friend class CudaBlockStructureTest;
};
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_BLOCK_SPARSE_STRUCTURE_H_
