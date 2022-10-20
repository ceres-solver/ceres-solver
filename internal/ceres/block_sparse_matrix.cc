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

#include "ceres/block_sparse_matrix.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "ceres/block_structure.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/eigen.h"
#include "ceres/parallel_for.h"
#include "ceres/small_blas.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"

#if defined(__clang__) || defined(__GNUC__)
#define PREFETCH(addr, rw, hint) __builtin_prefetch(addr, rw, hint)
#else
#define NO_PREFETCH
#endif

namespace ceres::internal {

BlockSparseMatrix::BlockSparseMatrix(
    CompressedRowBlockStructure* block_structure)
    : num_rows_(0),
      num_cols_(0),
      num_nonzeros_(0),
      block_structure_(block_structure) {
  CHECK(block_structure_ != nullptr);

  // Count the number of columns in the matrix.
  for (auto& col : block_structure_->cols) {
    num_cols_ += col.size;
  }

  // Count the number of non-zero entries and the number of rows in
  // the matrix.
  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_size = block_structure_->rows[i].block.size;
    num_rows_ += row_block_size;

    const std::vector<Cell>& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      num_nonzeros_ += col_block_size * row_block_size;
    }
  }

  CHECK_GE(num_rows_, 0);
  CHECK_GE(num_cols_, 0);
  CHECK_GE(num_nonzeros_, 0);
  VLOG(2) << "Allocating values array with " << num_nonzeros_ * sizeof(double)
          << " bytes.";  // NOLINT
  values_ = std::make_unique<double[]>(num_nonzeros_);
  max_num_nonzeros_ = num_nonzeros_;
  CHECK(values_ != nullptr);
}

void BlockSparseMatrix::AddTransposeBlockStructure() {
  // Should this always compute new structure?
  if (transpose_block_structure_ == nullptr) {
    transpose_block_structure_ = CreateTranspose(*block_structure_);
    auto& transpose_rows = transpose_block_structure_->rows;
    if (transpose_rows.size()) {
      max_num_nonzeros_col_ = transpose_rows[0].block.nnz;
      for (int i = 1; i < transpose_rows.size(); ++i) {
        const int curr_nnz = transpose_rows[i].block.nnz;
        if (curr_nnz > max_num_nonzeros_col_) {
          max_num_nonzeros_col_ = curr_nnz;
        }
      }
    }
  }
}

void BlockSparseMatrix::SetZero() {
  std::fill(values_.get(), values_.get() + num_nonzeros_, 0.0);
}

void BlockSparseMatrix::RightMultiplyAndAccumulate(const double* x,
                                                   double* y) const {
  RightMultiplyAndAccumulate(x, y, nullptr, 1);
}

void BlockSparseMatrix::RightMultiplyAndAccumulate(const double* x,
                                                   double* y,
                                                   ContextImpl* context,
                                                   int num_threads) const {
  CHECK(x != nullptr);
  CHECK(y != nullptr);

  const auto values = values_.get();
  const auto block_structure = block_structure_.get();
  const auto num_row_blocks = block_structure->rows.size();

  ParallelFor(context,
              0,
              num_row_blocks,
              num_threads,
              [values, block_structure, x, y](int row_block_id) {
                const int row_block_pos =
                    block_structure->rows[row_block_id].block.position;
                const int row_block_size =
                    block_structure->rows[row_block_id].block.size;
                const auto& cells = block_structure->rows[row_block_id].cells;
                for (const auto& cell : cells) {
                  const int col_block_id = cell.block_id;
                  const int col_block_size =
                      block_structure->cols[col_block_id].size;
                  const int col_block_pos =
                      block_structure->cols[col_block_id].position;
                  MatrixVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                      values + cell.position,
                      row_block_size,
                      col_block_size,
                      x + col_block_pos,
                      y + row_block_pos);
                }
              });
}

void BlockSparseMatrix::LeftMultiplyAndAccumulate(const double* x,
                                                  double* y,
                                                  ContextImpl* context,
                                                  int num_threads) const {
  // While utilizing transposed structure allows to perform parallel
  // left-multiplication by dense vector, it makes access patterns to matrix
  // elements scattered. Thus, parallel exectution makes sense only for parallel
  // execution
  CHECK(x != nullptr);
  CHECK(y != nullptr);
  if (transpose_block_structure_ == nullptr || num_threads == 1) {
    LeftMultiplyAndAccumulate(x, y);
    return;
  }

  auto transpose_bs = transpose_block_structure_.get();
  const int num_cols = transpose_bs->rows.size();
  if (!num_cols) return;

  // Partition column blocks into disjoint sets of consecutive column blocks
  // with balanced distribution of non-zero elements
  //
  // We use partitioning algorithm that provides block sizes that are
  // within [partition_threshold, partition_threshold +
  // max_number_of_nonzeros_per_block)
  //
  // Assuming that execution time is proportional to number of non-zeros, and
  // having column blocks with at least max_num_nonzeros_col_ non-zero elements,
  // we set partition size threshold to the value that limits maximal
  int nnz_total = transpose_bs->rows.back().block.cumulative_nnz;
  const int num_partitions = num_threads * 2;
  const int maximum_partition_size = std::max(
      (nnz_total + num_partitions - 1) / num_partitions, max_num_nonzeros_col_);
  const int partition_size = (maximum_partition_size + 1) / 2;

  std::vector<int> column_blocks_partition;
  CreateFairRowPartition(
      *transpose_bs, 0, num_cols, partition_size, &column_blocks_partition);

  // Last element is the right boundary of the last column block set
  const int num_partitions_real = column_blocks_partition.size() - 1;
  const double* values = values_.get();
#ifndef NO_PREFETCH
  ParallelFor(
      context,
      0,
      num_partitions_real,
      num_threads,
      [values, transpose_bs, &column_blocks_partition, x, y](int partition_id) {
        const int first_col_id = column_blocks_partition[partition_id];
        const int last_col_id = column_blocks_partition[partition_id + 1];
        auto first_col = transpose_bs->rows.data() + first_col_id;
        auto last_col = transpose_bs->rows.data() + last_col_id;
        auto prefetch_col = first_col;
        auto prefetch_cell = prefetch_col->cells.begin();
        auto prefetch_last_cell = prefetch_col->cells.end();
        int prefetch_row_pos = prefetch_col->block.position;
        int prefetch_row_size = prefetch_col->block.size;

        const int kPrefetchDepth = 8;
        int prefetched = 0;

        int process_id = 0;
        int prefetch_id = 0;
        const double* prefetch_values[kPrefetchDepth];
        int prefetch_col_block_size[kPrefetchDepth];
        int prefetch_row_block_size[kPrefetchDepth];
        const double* prefetch_x[kPrefetchDepth];
        double* prefetch_y[kPrefetchDepth];

        while (prefetched > 0 || prefetch_col < last_col) {
          while (prefetch_cell == prefetch_last_cell) {
            ++prefetch_col;
            if (prefetch_col < last_col) {
              prefetch_cell = prefetch_col->cells.begin();
              prefetch_last_cell = prefetch_col->cells.end();
              prefetch_row_pos = prefetch_col->block.position;
              prefetch_row_size = prefetch_col->block.size;
            } else {
              break;
            }
          }
          if (prefetch_col < last_col) {
            // prefetch next cell
            prefetch_values[prefetch_id] = values + prefetch_cell->position;
            const int col_block_id = prefetch_cell->block_id;
            const auto& col_block = transpose_bs->cols[col_block_id];

            prefetch_row_block_size[prefetch_id] = prefetch_row_size;
            prefetch_y[prefetch_id] = y + prefetch_row_pos;

            prefetch_col_block_size[prefetch_id] = col_block.size;
            const int col_block_pos = col_block.position;
            prefetch_x[prefetch_id] = x + col_block_pos;

            PREFETCH(prefetch_values[prefetch_id], 0, 3);
            PREFETCH(prefetch_x[prefetch_id], 0, 3);
            PREFETCH(prefetch_y[prefetch_id], 1, 3);

            ++prefetched;
            ++prefetch_cell;
            prefetch_id = (prefetch_id + 1) % kPrefetchDepth;
          }
          if (prefetched != kPrefetchDepth && prefetch_col < last_col) continue;
          // multiply the oldest prefetched cell
          MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
              prefetch_values[process_id],
              prefetch_col_block_size[process_id],
              prefetch_row_block_size[process_id],
              prefetch_x[process_id],
              prefetch_y[process_id]);
          process_id = (process_id + 1) % kPrefetchDepth;
          --prefetched;
        }
      });
#else
  ParallelFor(
      context,
      0,
      num_partitions_real,
      num_threads,
      [values, transpose_bs, &column_blocks_partition, x, y](int partition_id) {
        for (int i = column_blocks_partition[partition_id];
             i < column_blocks_partition[partition_id + 1];
             ++i) {
          int row_block_pos = transpose_bs->rows[i].block.position;
          int row_block_size = transpose_bs->rows[i].block.size;
          auto& cells = transpose_bs->rows[i].cells;

          for (auto& cell : cells) {
            const int col_block_id = cell.block_id;
            const int col_block_size = transpose_bs->cols[col_block_id].size;
            const int col_block_pos = transpose_bs->cols[col_block_id].position;
            MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
                values + cell.position,
                col_block_size,
                row_block_size,
                x + col_block_pos,
                y + row_block_pos);
          }
        }
      });

#endif
}

void BlockSparseMatrix::LeftMultiplyAndAccumulate(const double* x,
                                                  double* y) const {
  CHECK(x != nullptr);
  CHECK(y != nullptr);
  // Single-threaded left products are always computed using a non-transpose
  // block structure, because it has linear acess pattern to matrix elements
  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_pos = block_structure_->rows[i].block.position;
    int row_block_size = block_structure_->rows[i].block.size;
    const auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      int col_block_pos = block_structure_->cols[col_block_id].position;
      MatrixTransposeVectorMultiply<Eigen::Dynamic, Eigen::Dynamic, 1>(
          values_.get() + cell.position,
          row_block_size,
          col_block_size,
          x + row_block_pos,
          y + col_block_pos);
    }
  }
}

void BlockSparseMatrix::SquaredColumnNorm(double* x) const {
  CHECK(x != nullptr);
  VectorRef(x, num_cols_).setZero();
  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_size = block_structure_->rows[i].block.size;
    auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      int col_block_pos = block_structure_->cols[col_block_id].position;
      const MatrixRef m(
          values_.get() + cell.position, row_block_size, col_block_size);
      VectorRef(x + col_block_pos, col_block_size) += m.colwise().squaredNorm();
    }
  }
}

void BlockSparseMatrix::ScaleColumns(const double* scale) {
  CHECK(scale != nullptr);

  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_size = block_structure_->rows[i].block.size;
    auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      int col_block_pos = block_structure_->cols[col_block_id].position;
      MatrixRef m(
          values_.get() + cell.position, row_block_size, col_block_size);
      m *= ConstVectorRef(scale + col_block_pos, col_block_size).asDiagonal();
    }
  }
}

void BlockSparseMatrix::ToCompressedRowSparseMatrix(
    CompressedRowSparseMatrix* crs_matrix) const {
  {
    TripletSparseMatrix ts_matrix;
    this->ToTripletSparseMatrix(&ts_matrix);
    *crs_matrix =
        *CompressedRowSparseMatrix::FromTripletSparseMatrix(ts_matrix);
  }

  int num_row_blocks = block_structure_->rows.size();
  auto& row_blocks = *crs_matrix->mutable_row_blocks();
  row_blocks.resize(num_row_blocks);
  for (int i = 0; i < num_row_blocks; ++i) {
    row_blocks[i] = block_structure_->rows[i].block;
  }

  int num_col_blocks = block_structure_->cols.size();
  auto& col_blocks = *crs_matrix->mutable_col_blocks();
  col_blocks.resize(num_col_blocks);
  for (int i = 0; i < num_col_blocks; ++i) {
    col_blocks[i] = block_structure_->cols[i];
  }
}

void BlockSparseMatrix::ToDenseMatrix(Matrix* dense_matrix) const {
  CHECK(dense_matrix != nullptr);

  dense_matrix->resize(num_rows_, num_cols_);
  dense_matrix->setZero();
  Matrix& m = *dense_matrix;

  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_pos = block_structure_->rows[i].block.position;
    int row_block_size = block_structure_->rows[i].block.size;
    auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      int col_block_pos = block_structure_->cols[col_block_id].position;
      int jac_pos = cell.position;
      m.block(row_block_pos, col_block_pos, row_block_size, col_block_size) +=
          MatrixRef(values_.get() + jac_pos, row_block_size, col_block_size);
    }
  }
}

void BlockSparseMatrix::ToTripletSparseMatrix(
    TripletSparseMatrix* matrix) const {
  CHECK(matrix != nullptr);

  matrix->Reserve(num_nonzeros_);
  matrix->Resize(num_rows_, num_cols_);
  matrix->SetZero();

  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    int row_block_pos = block_structure_->rows[i].block.position;
    int row_block_size = block_structure_->rows[i].block.size;
    const auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      int col_block_id = cell.block_id;
      int col_block_size = block_structure_->cols[col_block_id].size;
      int col_block_pos = block_structure_->cols[col_block_id].position;
      int jac_pos = cell.position;
      for (int r = 0; r < row_block_size; ++r) {
        for (int c = 0; c < col_block_size; ++c, ++jac_pos) {
          matrix->mutable_rows()[jac_pos] = row_block_pos + r;
          matrix->mutable_cols()[jac_pos] = col_block_pos + c;
          matrix->mutable_values()[jac_pos] = values_[jac_pos];
        }
      }
    }
  }
  matrix->set_num_nonzeros(num_nonzeros_);
}

// Return a pointer to the block structure. We continue to hold
// ownership of the object though.
const CompressedRowBlockStructure* BlockSparseMatrix::block_structure() const {
  return block_structure_.get();
}

// Return a pointer to the block structure of matrix transpose. We continue to
// hold ownership of the object though.
const CompressedRowBlockStructure*
BlockSparseMatrix::transpose_block_structure() const {
  return transpose_block_structure_.get();
}

void BlockSparseMatrix::ToTextFile(FILE* file) const {
  CHECK(file != nullptr);
  for (int i = 0; i < block_structure_->rows.size(); ++i) {
    const int row_block_pos = block_structure_->rows[i].block.position;
    const int row_block_size = block_structure_->rows[i].block.size;
    const auto& cells = block_structure_->rows[i].cells;
    for (const auto& cell : cells) {
      const int col_block_id = cell.block_id;
      const int col_block_size = block_structure_->cols[col_block_id].size;
      const int col_block_pos = block_structure_->cols[col_block_id].position;
      int jac_pos = cell.position;
      for (int r = 0; r < row_block_size; ++r) {
        for (int c = 0; c < col_block_size; ++c) {
          fprintf(file,
                  "% 10d % 10d %17f\n",
                  row_block_pos + r,
                  col_block_pos + c,
                  values_[jac_pos++]);
        }
      }
    }
  }
}

std::unique_ptr<BlockSparseMatrix> BlockSparseMatrix::CreateDiagonalMatrix(
    const double* diagonal, const std::vector<Block>& column_blocks) {
  // Create the block structure for the diagonal matrix.
  auto* bs = new CompressedRowBlockStructure();
  bs->cols = column_blocks;
  int position = 0;
  bs->rows.resize(column_blocks.size(), CompressedRow(1));
  for (int i = 0; i < column_blocks.size(); ++i) {
    CompressedRow& row = bs->rows[i];
    row.block = column_blocks[i];
    Cell& cell = row.cells[0];
    cell.block_id = i;
    cell.position = position;
    position += row.block.size * row.block.size;
  }

  // Create the BlockSparseMatrix with the given block structure.
  auto matrix = std::make_unique<BlockSparseMatrix>(bs);
  matrix->SetZero();

  // Fill the values array of the block sparse matrix.
  double* values = matrix->mutable_values();
  for (const auto& column_block : column_blocks) {
    const int size = column_block.size;
    for (int j = 0; j < size; ++j) {
      // (j + 1) * size is compact way of accessing the (j,j) entry.
      values[j * (size + 1)] = diagonal[j];
    }
    diagonal += size;
    values += size * size;
  }

  return matrix;
}

void BlockSparseMatrix::AppendRows(const BlockSparseMatrix& m) {
  CHECK_EQ(m.num_cols(), num_cols());
  const CompressedRowBlockStructure* m_bs = m.block_structure();
  CHECK_EQ(m_bs->cols.size(), block_structure_->cols.size());

  const int old_num_nonzeros = num_nonzeros_;
  const int old_num_row_blocks = block_structure_->rows.size();
  block_structure_->rows.resize(old_num_row_blocks + m_bs->rows.size());

  for (int i = 0; i < m_bs->rows.size(); ++i) {
    const CompressedRow& m_row = m_bs->rows[i];
    const int row_block_id = old_num_row_blocks + i;
    CompressedRow& row = block_structure_->rows[row_block_id];
    row.block.size = m_row.block.size;
    row.block.position = num_rows_;
    num_rows_ += m_row.block.size;
    row.cells.resize(m_row.cells.size());
    if (transpose_block_structure_) {
      transpose_block_structure_->cols.emplace_back(row.block);
    }
    for (int c = 0; c < m_row.cells.size(); ++c) {
      const int block_id = m_row.cells[c].block_id;
      row.cells[c].block_id = block_id;
      row.cells[c].position = num_nonzeros_;

      const int block_size = m_row.block.size * m_bs->cols[block_id].size;
      if (transpose_block_structure_) {
        transpose_block_structure_->rows[block_id].cells.emplace_back(
            row_block_id, num_nonzeros_);
        transpose_block_structure_->rows[block_id].block.nnz += block_size;
      }

      num_nonzeros_ += block_size;
    }
  }

  if (num_nonzeros_ > max_num_nonzeros_) {
    auto new_values = std::make_unique<double[]>(num_nonzeros_);
    std::copy_n(values_.get(), old_num_nonzeros, new_values.get());
    values_ = std::move(new_values);
    max_num_nonzeros_ = num_nonzeros_;
  }

  std::copy(m.values(),
            m.values() + m.num_nonzeros(),
            values_.get() + old_num_nonzeros);

  if (transpose_block_structure_ && transpose_block_structure_->rows.size()) {
    auto& transpose_rows = transpose_block_structure_->rows;

    transpose_rows[0].block.cumulative_nnz = transpose_rows[0].block.nnz;
    max_num_nonzeros_col_ = transpose_rows[0].block.nnz;
    for (int c = 1; c < transpose_rows.size(); ++c) {
      const int curr_nnz = transpose_rows[c].block.nnz;
      if (curr_nnz > max_num_nonzeros_col_) max_num_nonzeros_col_ = curr_nnz;
      transpose_rows[c].block.cumulative_nnz =
          curr_nnz + transpose_rows[c - 1].block.cumulative_nnz;
    }
  }
}

void BlockSparseMatrix::DeleteRowBlocks(const int delta_row_blocks) {
  const int num_row_blocks = block_structure_->rows.size();
  const int new_num_row_blocks = num_row_blocks - delta_row_blocks;
  int delta_num_nonzeros = 0;
  int delta_num_rows = 0;
  const std::vector<Block>& column_blocks = block_structure_->cols;
  for (int i = 0; i < delta_row_blocks; ++i) {
    const CompressedRow& row = block_structure_->rows[num_row_blocks - i - 1];
    delta_num_rows += row.block.size;
    for (int c = 0; c < row.cells.size(); ++c) {
      const Cell& cell = row.cells[c];
      delta_num_nonzeros += row.block.size * column_blocks[cell.block_id].size;

      if (transpose_block_structure_) {
        auto& col_cells = transpose_block_structure_->rows[cell.block_id].cells;
        while (col_cells.size() &&
               col_cells.back().block_id >= new_num_row_blocks) {
          const int del_block_id = col_cells.back().block_id;
          const int del_block_rows =
              block_structure_->rows[del_block_id].block.size;
          const int del_block_cols = column_blocks[cell.block_id].size;
          const int del_cell_size = del_block_rows * del_block_cols;
          transpose_block_structure_->rows[cell.block_id].block.nnz -=
              del_cell_size;
          col_cells.pop_back();
        }
      }
    }
  }
  num_nonzeros_ -= delta_num_nonzeros;
  num_rows_ -= delta_num_rows;
  block_structure_->rows.resize(num_row_blocks - delta_row_blocks);
  if (transpose_block_structure_) {
    for (int i = 0; i < delta_row_blocks; ++i) {
      transpose_block_structure_->cols.pop_back();
    }
    if (transpose_block_structure_->rows.size()) {
      auto& transpose_rows = transpose_block_structure_->rows;

      transpose_rows[0].block.cumulative_nnz = transpose_rows[0].block.nnz;
      max_num_nonzeros_col_ = transpose_rows[0].block.nnz;
      for (int c = 1; c < transpose_rows.size(); ++c) {
        const int curr_nnz = transpose_rows[c].block.nnz;
        if (curr_nnz > max_num_nonzeros_col_) {
          max_num_nonzeros_col_ = curr_nnz;
        }
        transpose_rows[c].block.cumulative_nnz =
            curr_nnz + transpose_rows[c - 1].block.cumulative_nnz;
      }
    }
  }
}

std::unique_ptr<BlockSparseMatrix> BlockSparseMatrix::CreateRandomMatrix(
    const BlockSparseMatrix::RandomMatrixOptions& options, std::mt19937& prng) {
  CHECK_GT(options.num_row_blocks, 0);
  CHECK_GT(options.min_row_block_size, 0);
  CHECK_GT(options.max_row_block_size, 0);
  CHECK_LE(options.min_row_block_size, options.max_row_block_size);
  CHECK_GT(options.block_density, 0.0);
  CHECK_LE(options.block_density, 1.0);

  std::uniform_int_distribution<int> col_distribution(
      options.min_col_block_size, options.max_col_block_size);
  std::uniform_int_distribution<int> row_distribution(
      options.min_row_block_size, options.max_row_block_size);
  auto* bs = new CompressedRowBlockStructure();
  if (options.col_blocks.empty()) {
    CHECK_GT(options.num_col_blocks, 0);
    CHECK_GT(options.min_col_block_size, 0);
    CHECK_GT(options.max_col_block_size, 0);
    CHECK_LE(options.min_col_block_size, options.max_col_block_size);

    // Generate the col block structure.
    int col_block_position = 0;
    for (int i = 0; i < options.num_col_blocks; ++i) {
      const int col_block_size = col_distribution(prng);
      bs->cols.emplace_back(col_block_size, col_block_position);
      col_block_position += col_block_size;
    }
  } else {
    bs->cols = options.col_blocks;
  }

  bool matrix_has_blocks = false;
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  while (!matrix_has_blocks) {
    VLOG(1) << "Clearing";
    bs->rows.clear();
    int row_block_position = 0;
    int value_position = 0;
    for (int r = 0; r < options.num_row_blocks; ++r) {
      const int row_block_size = row_distribution(prng);
      bs->rows.emplace_back();
      CompressedRow& row = bs->rows.back();
      row.block.size = row_block_size;
      row.block.position = row_block_position;
      row_block_position += row_block_size;
      for (int c = 0; c < bs->cols.size(); ++c) {
        if (uniform01(prng) > options.block_density) continue;

        row.cells.emplace_back();
        Cell& cell = row.cells.back();
        cell.block_id = c;
        cell.position = value_position;
        value_position += row_block_size * bs->cols[c].size;
        matrix_has_blocks = true;
      }
    }
  }

  auto matrix = std::make_unique<BlockSparseMatrix>(bs);
  double* values = matrix->mutable_values();
  std::normal_distribution<double> standard_normal_distribution;
  std::generate_n(
      values, matrix->num_nonzeros(), [&standard_normal_distribution, &prng] {
        return standard_normal_distribution(prng);
      });

  return matrix;
}

std::unique_ptr<CompressedRowBlockStructure> CreateTranspose(
    const CompressedRowBlockStructure& bs) {
  auto transpose = std::make_unique<CompressedRowBlockStructure>();

  transpose->rows.resize(bs.cols.size());
  for (int i = 0; i < bs.cols.size(); ++i) {
    transpose->rows[i].block = bs.cols[i];
    transpose->rows[i].block.nnz = 0;
  }

  transpose->cols.resize(bs.rows.size());
  for (int i = 0; i < bs.rows.size(); ++i) {
    auto& row = bs.rows[i];
    transpose->cols[i] = row.block;

    const int nrows = row.block.size;
    for (auto& cell : row.cells) {
      transpose->rows[cell.block_id].cells.emplace_back(i, cell.position);

      const int ncols = transpose->rows[cell.block_id].block.size;
      transpose->rows[cell.block_id].block.nnz += nrows * ncols;
    }
  }
  transpose->rows[0].block.cumulative_nnz = transpose->rows[0].block.nnz;
  for (int i = 1; i < bs.cols.size(); ++i) {
    const int curr_nnz = transpose->rows[i].block.nnz;
    transpose->rows[i].block.cumulative_nnz =
        curr_nnz + transpose->rows[i - 1].block.cumulative_nnz;
  }
  return transpose;
}

void CreateFairRowPartition(const CompressedRowBlockStructure& block_structure,
                            int first_row_block,
                            int last_row_block,
                            int partition_threshold,
                            std::vector<int>* partition) {
  CHECK(partition != nullptr);
  partition->clear();
  if (!block_structure.rows.size()) return;
  if (first_row_block == last_row_block) return;
  CHECK_GE(first_row_block, 0);
  CHECK_LT(first_row_block, last_row_block);
  CHECK_GE(block_structure.rows[first_row_block].block.cumulative_nnz, 0);
  CHECK_LE(last_row_block, block_structure.rows.size());

  int nnz_before = 0;
  if (first_row_block > 0) {
    nnz_before = block_structure.rows[first_row_block - 1].block.cumulative_nnz;
  }
  const int max_partitions =
      (block_structure.rows[last_row_block - 1].block.cumulative_nnz +
       partition_threshold - 1) /
      partition_threshold;
  partition->reserve(max_partitions + 1);

  partition->push_back(first_row_block);
  auto last_block = block_structure.rows.begin() + last_row_block;
  while (first_row_block < last_row_block) {
    const int target = partition_threshold + nnz_before;
    auto next = std::partition_point(
        block_structure.rows.begin() + first_row_block,
        last_block,
        [target](const auto& v) { return v.block.cumulative_nnz < target; });
    int row_block_id = std::distance(block_structure.rows.begin(), next);
    if (row_block_id != last_row_block) ++row_block_id;

    partition->push_back(row_block_id);
    first_row_block = row_block_id;
    nnz_before = block_structure.rows[row_block_id - 1].block.cumulative_nnz;
  }
}

}  // namespace ceres::internal
