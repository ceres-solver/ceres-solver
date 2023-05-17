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

#include "ceres/cuda_block_structure.h"

#ifndef CERES_NO_CUDA

namespace ceres::internal {
namespace {
// Dimension of a sorted array of blocks
inline int Dimension(const std::vector<Block>& blocks) {
  if (blocks.empty()) {
    return 0;
  }
  const auto& last = blocks.back();
  return last.size + last.position;
}
}  // namespace

CudaBlockSparseStructure::CudaBlockSparseStructure(
    const CompressedRowBlockStructure& block_structure, ContextImpl* context)
    : first_cell_in_row_block_(context),
      cells_(context),
      row_blocks_(context),
      col_blocks_(context) {
  // Row blocks extracted from CompressedRowBlockStructure::rows
  std::vector<Block> row_blocks;
  // Column blocks can be reused as-is
  const auto& col_blocks = block_structure.cols;

  // Row block offset is an index of the first cell corresponding to row block
  std::vector<int> first_cell_in_row_block;
  // Flat array of all cells from all row-blocks
  std::vector<Cell> cells;

  int f_values_offset = 0;
  size_t max_cells = 0;
  int max_row_block_size = 0;
  num_row_blocks_ = block_structure.rows.size();
  num_col_blocks_ = col_blocks.size();

  row_blocks.reserve(num_row_blocks_);
  first_cell_in_row_block.reserve(num_row_blocks_ + 1);
  num_nonzeros_ = 0;
  sequential_layout_ = true;
  for (const auto& r : block_structure.rows) {
    const int row_block_size = r.block.size;
    max_cells = std::max(max_cells, r.cells.size());
    max_row_block_size = std::max(row_block_size, max_row_block_size);
    row_blocks.emplace_back(r.block);
    first_cell_in_row_block.push_back(cells.size());
    for (const auto& c : r.cells) {
      const int col_block_size = col_blocks[c.block_id].size;
      const int cell_size = col_block_size * row_block_size;
      cells.push_back(c);
      sequential_layout_ &= c.position == num_nonzeros_;
      num_nonzeros_ += cell_size;
    }
  }
  first_cell_in_row_block.push_back(cells.size());
  num_cells_ = cells.size();

  num_rows_ = Dimension(row_blocks);
  num_cols_ = Dimension(col_blocks);

  crs_compatible_ =
      sequential_layout_ && (max_cells <= 1 || max_row_block_size == 1);

  if (VLOG_IS_ON(3)) {
    const size_t first_cell_in_row_block_size =
        first_cell_in_row_block.size() * sizeof(int);
    const size_t cells_size = cells.size() * sizeof(Cell);
    const size_t row_blocks_size = row_blocks.size() * sizeof(Block);
    const size_t col_blocks_size = col_blocks.size() * sizeof(Block);
    const size_t total_size = first_cell_in_row_block_size + cells_size +
                              col_blocks_size + row_blocks_size;
    const double ratio =
        (100. * total_size) / (num_nonzeros_ * (sizeof(int) + sizeof(double)) +
                               num_rows_ * sizeof(int));
    VLOG(3) << "\nCudaBlockSparseStructure:\n"
               "\tRow block offsets: "
            << first_cell_in_row_block_size
            << " bytes\n"
               "\tColumn blocks: "
            << col_blocks_size
            << " bytes\n"
               "\tRow blocks: "
            << row_blocks_size
            << " bytes\n"
               "\tCells: "
            << cells_size << " bytes\n\tTotal: " << total_size
            << " bytes of GPU memory (" << ratio << "% of CRS matrix size)";
  }

  first_cell_in_row_block_.CopyFromCpuVector(first_cell_in_row_block);
  cells_.CopyFromCpuVector(cells);
  row_blocks_.CopyFromCpuVector(row_blocks);
  col_blocks_.CopyFromCpuVector(col_blocks);
}
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
