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

#include "ceres/cuda_block_sparse_crs_view.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_kernels.h"
#include "ceres/parallel_vector_ops.h"

namespace ceres::internal {
namespace {
// Assumes that blocks array is sorted
int Dimension(const std::vector<Block>& blocks) {
  if (blocks.empty()) return 0;
  const auto& last = blocks.back();
  return last.size + last.position;
}
}  // namespace

CudaBlockSparseStructure::CudaBlockSparseStructure(
    const CompressedRowBlockStructure& block_structure, ContextImpl* context)
    : row_block_offsets_(context),
      cells_(context),
      row_blocks_(context),
      col_blocks_(context) {
  // Row blocks extracted from CompressedRowBlockStructure::rows
  std::vector<Block> row_blocks;
  // Column blocks can be reused as-is
  const auto& col_blocks = block_structure.cols;

  // Row block offset is an index of the first cell corresponding to row block
  std::vector<int> row_block_offsets;
  // Flat array of all cells from all row-blocks
  std::vector<Cell> cells;

  num_row_blocks_ = block_structure.rows.size();
  num_col_blocks_ = col_blocks.size();

  row_blocks.reserve(num_row_blocks_);
  row_block_offsets.reserve(num_row_blocks_ + 1);
  num_nonzeros_ = 0;
  num_cells_ = 0;
  for (auto& r : block_structure.rows) {
    const int row_block_size = r.block.size;
    row_blocks.emplace_back(r.block);
    row_block_offsets.push_back(num_cells_);
    for (auto& c : r.cells) {
      cells.emplace_back(c);
      const int col_block_size = col_blocks[c.block_id].size;
      num_nonzeros_ += col_block_size * row_block_size;
      ++num_cells_;
    }
  }
  row_block_offsets.push_back(num_cells_);

  num_rows_ = Dimension(row_blocks);
  num_cols_ = Dimension(col_blocks);

  if (VLOG_IS_ON(5)) {
    const size_t row_block_offsets_size =
        row_block_offsets.size() * sizeof(int);
    const size_t cells_size = cells.size() * sizeof(Cell);
    const size_t row_blocks_size = row_blocks.size() * sizeof(Block);
    const size_t col_blocks_size = col_blocks.size() * sizeof(Block);
    const size_t total_size =
        row_block_offsets_size + cells_size + col_blocks_size + row_blocks_size;
    VLOG(5) << "\nCudaBlockSparseStructure:\n"
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
            << " bytes\n"
               "\tTotal: "
            << total_size << " bytes of GPU memory";
  }

  row_block_offsets_.CopyFromCpuVector(row_block_offsets);
  cells_.CopyFromCpuVector(cells);
  row_blocks_.CopyFromCpuVector(row_blocks);
  col_blocks_.CopyFromCpuVector(col_blocks);
}

CudaBlockSparseCRSView::CudaBlockSparseCRSView(const BlockSparseMatrix& bsm,
                                               ContextImpl* context,
                                               bool copy_values)
    : permutation_(context, bsm.num_nonzeros()), streamer_(context) {
  CudaBlockSparseStructure block_structure(*bsm.block_structure(), context);
  crs_matrix_ = std::make_unique<CudaSparseMatrix>(
      bsm.num_rows(), bsm.num_cols(), bsm.num_nonzeros(), context);
  CudaBuffer<int> temp_rows(context, bsm.num_rows());
  FillCRSStructure(block_structure.num_row_blocks(),
                   bsm.num_rows(),
                   block_structure.row_block_offsets(),
                   block_structure.cells(),
                   block_structure.row_blocks(),
                   block_structure.col_blocks(),
                   crs_matrix_->rows(),
                   crs_matrix_->cols(),
                   temp_rows.data(),
                   permutation_.data(),
                   context->stream_);
  if (copy_values) {
    UpdateValues(bsm);
  }
}
void CudaBlockSparseCRSView::UpdateValues(const BlockSparseMatrix& bsm) {
  auto values_to = crs_matrix_->values();
  auto permutation = permutation_.data();
  streamer_.Stream(
      bsm.values(),
      bsm.num_nonzeros(),
      [permutation, values_to](
          const double* values, int num_values, int offset, auto stream) {
        PermuteValues(
            offset, num_values, permutation, values, values_to, stream);
      });
}

}  // namespace ceres::internal
#endif
