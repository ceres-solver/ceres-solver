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
    : row_block_offsets_(context),
      first_cell_pos_e_(context),
      first_cell_pos_f_(context),
      cells_(context),
      row_blocks_(context),
      col_blocks_(context) {
  // Row blocks extracted from CompressedRowBlockStructure::rows
  std::vector<Block> row_blocks;
  // Column blocks can be reused as-is
  const auto& col_blocks = block_structure.cols;

  // Row block offset is an index of the first cell corresponding to row block
  std::vector<int> row_block_offsets;
  // Cumulative (by row-blocks) non-zero count in E sub-matrix
  std::vector<int> first_cell_pos_e = {0};
  // Cumulative (by row-blocks) non-zero count in F sub-matrix
  std::vector<int> first_cell_pos_f;
  // Flat array of all cells from all row-blocks
  std::vector<Cell> cells;

  num_nonzeros_e_ = 0;
  int f_values_offset = 0;
  num_nonzeros_f_ = 0;
  int max_cells_e = 0;
  int max_cells_f = 0;
  int max_row_block_size = 0;
  int num_col_blocks_e = col_blocks.size();
  num_row_blocks_ = block_structure.rows.size();
  num_col_blocks_ = col_blocks.size();

  row_blocks.reserve(num_row_blocks_);
  row_block_offsets.reserve(num_row_blocks_ + 1);
  num_nonzeros_ = 0;
  num_cells_ = 0;
  for (const auto& r : block_structure.rows) {
    const int row_block_size = r.block.size;
    max_row_block_size = std::max(row_block_size, max_row_block_size);
    row_blocks.emplace_back(r.block);
    row_block_offsets.push_back(num_cells_);
    int num_cells_e = 0;
    int num_cells_f = 0;
    for (const auto& c : r.cells) {
      const int col_block_size = col_blocks[c.block_id].size;
      const int cell_size = col_block_size * row_block_size;
      num_nonzeros_ += cell_size;
      bool cell_in_f = false;
      if (c.position == num_nonzeros_e_) {
        // Cell that follows cell in E is in F iff it is to the right of
        // boundary
        cell_in_f = c.block_id >= num_col_blocks_e;
      } else {
        // Every cell that follows cell from F is cell from F
        CHECK(num_nonzeros_f_ == 0 ||
              c.position == f_values_offset + num_nonzeros_f_);
        cell_in_f = true;
      }

      if (cell_in_f) {
        ++num_cells_f;
        if (f_values_offset == 0) {
          f_values_offset = c.position;
          first_cell_pos_f.push_back(f_values_offset);
        }
        num_col_blocks_e = std::min(num_col_blocks_e, c.block_id);
        CHECK_EQ(c.position, f_values_offset + num_nonzeros_f_);
        num_nonzeros_f_ += cell_size;

      } else {
        ++num_cells_e;
        CHECK(f_values_offset == 0 ||
              c.position + cell_size <= f_values_offset);
        CHECK_GE(num_nonzeros_e_, 0);
        CHECK_LT(c.block_id, num_col_blocks_e);
        CHECK_EQ(c.position, num_nonzeros_e_);
        num_nonzeros_e_ += cell_size;
      }
      cells.emplace_back(c);
      ++num_cells_;
    }
    max_cells_e = std::max(max_cells_e, num_cells_e);
    max_cells_f = std::max(max_cells_f, num_cells_f);
    first_cell_pos_e.push_back(num_nonzeros_e_);
    first_cell_pos_f.push_back(num_nonzeros_f_ + f_values_offset);
  }
  CHECK_EQ(num_nonzeros_e_ + num_nonzeros_f_, num_nonzeros_);
  CHECK(num_nonzeros_f_ == 0 || num_nonzeros_e_ == f_values_offset);
  row_block_offsets.push_back(num_cells_);

  num_rows_ = Dimension(row_blocks);
  num_cols_ = Dimension(col_blocks);

  // Sub-matrix is crs-compatible if either:
  //  - There is atmost one cell per row-block
  //  - All row-blocks have size = 1
  e_is_crs_compatible_ = max_cells_e <= 1 || max_row_block_size == 1;
  f_is_crs_compatible_ = max_cells_f <= 1 || max_row_block_size == 1;
  is_partitioned_ = f_values_offset != 0;

  VLOG(3) << "Inferred matrix structure: "
          << (is_partitioned_ ? "partitioned" : "non-partitioned")
          << ", num_col_blocks_e = " << num_col_blocks_e
          << ", max_cells_e = " << max_cells_e
          << ", max_cells_f = " << max_cells_f
          << ", max_row_block_size = " << max_row_block_size;
  if (VLOG_IS_ON(3)) {
    const size_t row_block_offsets_size =
        row_block_offsets.size() * sizeof(int);
    const size_t cells_size = cells.size() * sizeof(Cell);
    const size_t row_blocks_size = row_blocks.size() * sizeof(Block);
    const size_t col_blocks_size = col_blocks.size() * sizeof(Block);
    const size_t first_cell_pos_e_size =
        is_partitioned_ ? sizeof(int) * (num_row_blocks_ + 1) : 0;
    const size_t first_cell_pos_f_size =
        is_partitioned_ ? sizeof(int) * (num_row_blocks_ + 1) : 0;
    const size_t total_size = row_block_offsets_size + cells_size +
                              col_blocks_size + row_blocks_size +
                              first_cell_pos_e_size + first_cell_pos_f_size;
    const double ratio =
        (100. * total_size) / (num_nonzeros_ * (sizeof(int) + sizeof(double)) +
                               num_rows_ * sizeof(int));
    VLOG(3) << "\nCudaBlockSparseStructure:\n"
               "\tRow block offsets: "
            << row_block_offsets_size
            << " bytes\n"
               "\tColumn blocks: "
            << col_blocks_size
            << " bytes\n"
               "\tRow blocks: "
            << row_blocks_size
            << " bytes\n"
               "\tCells: "
            << cells_size
            << " bytes\n\tCumulative nnz (E): " << first_cell_pos_e_size
            << " bytes\n\tCumulative nnz (F): " << first_cell_pos_f_size
            << " bytes\n\tTotal: " << total_size << " bytes of GPU memory ("
            << ratio << "% of CRS matrix size)";
  }

  row_block_offsets_.CopyFromCpuVector(row_block_offsets);
  cells_.CopyFromCpuVector(cells);
  row_blocks_.CopyFromCpuVector(row_blocks);
  col_blocks_.CopyFromCpuVector(col_blocks);

  if (is_partitioned_) {
    // For non-partitioned matrix this vector contains
    // cell[row_block_offsets[i]].position
    first_cell_pos_e_.CopyFromCpuVector(first_cell_pos_e);
    // For non-partitioned matrix this vector only contains zeros
    first_cell_pos_f_.CopyFromCpuVector(first_cell_pos_f);
  }
}
}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
