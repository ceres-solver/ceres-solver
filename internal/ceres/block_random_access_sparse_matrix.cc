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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/block_random_access_sparse_matrix.h"

#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "ceres/internal/export.h"
#include "ceres/parallel_vector_ops.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres::internal {

BlockRandomAccessSparseMatrix::BlockRandomAccessSparseMatrix(
    const std::vector<Block>& blocks,
    const std::set<std::pair<int, int>>& block_pairs,
    ContextImpl* context,
    int num_threads)
    : blocks_(blocks), context_(context), num_threads_(num_threads) {
  CHECK_LE(blocks.size(), std::numeric_limits<std::int32_t>::max());

  const int num_cols = NumScalarEntries(blocks);
  const int num_col_blocks = blocks.size();

  // block_pairs is already sorted
  std::vector<int> num_cells_at_row(num_col_blocks);
  for (auto& p : block_pairs) {
    ++num_cells_at_row[p.first];
  }
  auto block_structure_ = new CompressedRowBlockStructure;
  block_structure_->cols = blocks;
  block_structure_->rows.resize(num_col_blocks);
  auto p = block_pairs.begin();
  int num_nonzeros = 0;
  for (int i = 0; i < num_col_blocks; ++i) {
    auto& row = block_structure_->rows[i];
    row.block = blocks[i];
    row.cells.reserve(num_cells_at_row[i]);
    for (; p != block_pairs.end() && i == p->first; ++p) {
      row.cells.emplace_back(p->second, num_nonzeros);
      num_nonzeros += blocks[i].size * blocks[p->second].size;
    }
  }
  bsm_ = std::make_unique<BlockSparseMatrix>(block_structure_);
  VLOG(1) << "Matrix Size [" << num_cols << "," << num_cols << "] "
          << num_nonzeros;
  double* values = bsm_->mutable_values();
  for (int row_block_id = 0; row_block_id < num_col_blocks; ++row_block_id) {
    const auto& cells = block_structure_->rows[row_block_id].cells;
    for (auto& c : cells) {
      const int col_block_id = c.block_id;
      double* const data = values + c.position;
      const auto block_pair = std::make_pair(row_block_id, col_block_id);
      cell_values_.emplace_back(block_pair, data);
      layout_[IntPairToInt64(row_block_id, col_block_id)] =
          std::make_unique<CellInfo>(data);
    }
  }
}

CellInfo* BlockRandomAccessSparseMatrix::GetCell(int row_block_id,
                                                 int col_block_id,
                                                 int* row,
                                                 int* col,
                                                 int* row_stride,
                                                 int* col_stride) {
  const auto it = layout_.find(IntPairToInt64(row_block_id, col_block_id));
  if (it == layout_.end()) {
    return nullptr;
  }

  // Each cell is stored contiguously as its own little dense matrix.
  *row = 0;
  *col = 0;
  *row_stride = blocks_[row_block_id].size;
  *col_stride = blocks_[col_block_id].size;
  return it->second.get();
}

// Assume that the user does not hold any locks on any cell blocks
// when they are calling SetZero.
void BlockRandomAccessSparseMatrix::SetZero() {
  bsm_->SetZero(context_, num_threads_);
}

void BlockRandomAccessSparseMatrix::SymmetricRightMultiplyAndAccumulate(
    const double* x, double* y) const {
  for (const auto& cell_position_and_data : cell_values_) {
    const int row = cell_position_and_data.first.first;
    const int row_block_size = blocks_[row].size;
    const int row_block_pos = blocks_[row].position;

    const int col = cell_position_and_data.first.second;
    const int col_block_size = blocks_[col].size;
    const int col_block_pos = blocks_[col].position;

    MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
        cell_position_and_data.second,
        row_block_size,
        col_block_size,
        x + col_block_pos,
        y + row_block_pos);

    // Since the matrix is symmetric, but only the upper triangular
    // part is stored, if the block being accessed is not a diagonal
    // block, then use the same block to do the corresponding lower
    // triangular multiply also.
    if (row != col) {
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          cell_position_and_data.second,
          row_block_size,
          col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}

}  // namespace ceres::internal
