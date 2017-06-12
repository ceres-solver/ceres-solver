// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
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

#include "ceres/outer_product.h"

#include <algorithm>

namespace ceres {
namespace internal {
namespace {

EIGEN_STRONG_INLINE void MatrixTransposeMatrixMultiply(const int ab_rows,
                                                       const double* a,
                                                       const int a_cols,
                                                       const double* b,
                                                       const int b_cols,
                                                       double* c,
                                                       int c_col_stride) {
  for (int r = 0; r < ab_rows; ++r) {
    double* c_r = c;
    for (int i1 = 0; i1 < a_cols; ++i1) {
      const double a_v = a[i1];
      for (int i2 = 0; i2 < b_cols; ++i2) {
        c_r[i2] += a_v * b[i2];
      }
      c_r += c_col_stride;
    }
    a += a_cols;
    b += b_cols;
  }
}

}  // namespace

// Create outer product matrix based on the block product information.
// The input block product is already sorted. This function does not
// set the sparse rows/cols information. Instead, it only collects the
// nonzeros for each compressed row and puts in row_nnz. The caller of
// this function will traverse the block product in a second round to
// generate the sparse rows/cols information. This function also
// computes the block offset information for the outer product matrix,
// which is used in outer product computation.
CompressedRowSparseMatrix* OuterProduct::CreateOuterProductMatrix(
    const CompressedRowSparseMatrix::StorageType storage_type,
    const std::vector<OuterProduct::ProductTerm>& product_terms,
    std::vector<int>* row_nnz) {
  // Count the number of unique product term, which in turn is the
  // number of non-zeros in the outer product. Also count the number
  // of non-zeros in each row.
  const CompressedRowBlockStructure* bs = m_.block_structure();
  const std::vector<Block>& blocks = bs->cols;
  row_nnz->resize(blocks.size());
  std::fill(row_nnz->begin(), row_nnz->end(), 0);
  (*row_nnz)[product_terms[0].row] = blocks[product_terms[0].col].size;
  int num_nonzeros =
      blocks[product_terms[0].row].size * blocks[product_terms[0].col].size;
  for (int i = 1; i < product_terms.size(); ++i) {
    const ProductTerm& previous = product_terms[i - 1];
    const ProductTerm& current = product_terms[i];

    // Each (row, col) block counts only once.
    // This check depends on product sorted on (row, col).
    if (current.row != previous.row || current.col != previous.col) {
      (*row_nnz)[current.row] += blocks[current.col].size;
      num_nonzeros += blocks[current.row].size * blocks[current.col].size;
    }
  }

  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(m_.num_cols(), m_.num_cols(), num_nonzeros);
  matrix->set_storage_type(storage_type);
  matrix->mutable_row_blocks()->resize(blocks.size());
  matrix->mutable_col_blocks()->resize(blocks.size());
  for (int i = 0; i < blocks.size(); ++i) {
    (*(matrix->mutable_row_blocks()))[i] = blocks[i].size;
    (*(matrix->mutable_col_blocks()))[i] = blocks[i].size;
  }

  return matrix;
}

OuterProduct::ProductTerm::ProductTerm(const int row,
                                        const int col,
                                        const int index)
    : row(row), col(col), index(index) {}

inline bool OuterProduct::ProductTerm::operator<(
    const OuterProduct::ProductTerm& right) const {
  if (row == right.row) {
    if (col == right.col) {
      return index < right.index;
    }
    return col < right.col;
  }
  return row < right.row;
}

OuterProduct::OuterProduct(const BlockSparseMatrix& m,
                             const int start_row_block,
                             const int end_row_block)
    : m_(m), start_row_block_(start_row_block), end_row_block_(end_row_block) {}

// Compute the sparsity structure of the product m.transpose() * m
// and create a CompressedRowSparseMatrix corresponding to it.
//
// Also compute the "program" vector, which for every term in the
// block outer product provides the information for the entry in the
// values array of the result matrix where it should be accumulated.
//
// Since the entries of the program are the same for rows with the
// same sparsity structure, the program only stores the result for
// one row per row block. The ComputeProduct function reuses this
// information for each row in the row block.
//
// product_storage_type controls the form of the output matrix. It
// can be LOWER_TRIANGULAR or UPPER_TRIANGULAR.
OuterProduct* OuterProduct::Create(
    const BlockSparseMatrix& m,
    CompressedRowSparseMatrix::StorageType product_storage_type) {
  return OuterProduct::Create(
      m, 0, m.block_structure()->rows.size(), product_storage_type);
}

OuterProduct* OuterProduct::Create(
    const BlockSparseMatrix& m,
    const int start_row_block,
    const int end_row_block,
    CompressedRowSparseMatrix::StorageType product_storage_type) {
  CHECK(product_storage_type == CompressedRowSparseMatrix::LOWER_TRIANGULAR ||
        product_storage_type == CompressedRowSparseMatrix::UPPER_TRIANGULAR);
  CHECK_GT(m.num_nonzeros(), 0)
      << "Congratulations, you found a bug in Ceres. Please report it.";
  scoped_ptr<OuterProduct> outer_product(
      new OuterProduct(m, start_row_block, end_row_block));
  outer_product->Init(product_storage_type);
  return outer_product.release();
}

void OuterProduct::Init(
    const CompressedRowSparseMatrix::StorageType product_storage_type) {
  std::vector<OuterProduct::ProductTerm> product_terms;
  const CompressedRowBlockStructure* bs = m_.block_structure();

  // Give input matrix m in Block Sparse format
  //     (row_block, col_block)
  // represent each block multiplication
  //     (row_block, col_block1)' X (row_block, col_block2)
  // by its product term index and sort the product terms
  //     (col_block1, col_block2, index)
  for (int row_block = start_row_block_; row_block < end_row_block_;
       ++row_block) {
    const CompressedRow& row = bs->rows[row_block];
    for (int c1 = 0; c1 < row.cells.size(); ++c1) {
      const Cell& cell1 = row.cells[c1];
      int c2_begin, c2_end;
      if (product_storage_type == CompressedRowSparseMatrix::LOWER_TRIANGULAR) {
        c2_begin = 0;
        c2_end = c1 + 1;
      } else {
        c2_begin = c1;
        c2_end = row.cells.size();
      }

      for (int c2 = c2_begin; c2 < c2_end; ++c2) {
        const Cell& cell2 = row.cells[c2];
        product_terms.push_back(OuterProduct::ProductTerm(
            cell1.block_id, cell2.block_id, product_terms.size()));
      }
    }
  }

  CHECK_GT(product_terms.size(), 0);
  std::sort(product_terms.begin(), product_terms.end());
  CompressAndFillProgram(product_storage_type, product_terms);
}

void OuterProduct::CompressAndFillProgram(
    const CompressedRowSparseMatrix::StorageType product_storage_type,
    const std::vector<OuterProduct::ProductTerm>& product_terms) {
  CHECK_GT(product_terms.size(), 0);
  const int num_cols = m_.num_cols();
  std::vector<int> row_nnz;
  matrix_.reset(
      CreateOuterProductMatrix(product_storage_type, product_terms, &row_nnz));

  int* crsm_rows = matrix_->mutable_rows();
  std::fill(crsm_rows, crsm_rows + num_cols + 1, 0);
  int* crsm_cols = matrix_->mutable_cols();
  std::fill(crsm_cols, crsm_cols + matrix_->num_nonzeros(), 0);

  // Non zero elements are not stored consecutively across rows in a block.
  // We seperate nonzero into three categories:
  //   nonzeros in all previous row blocks counted in nnz
  //   nonzeros in current row counted in row_nnz
  //   nonzeros in previous col blocks of current row counted in col_nnz
  //
  // Give an element (j, k) within a block such that j and k
  // represent the relative position to the starting row and starting col of
  // the block, the row and col for the element is
  //   block_offsets_[current.row] + j
  //   block_offsets_[current.col] + k
  // The total number of nonzero to the element is
  //   nnz + row_nnz[current.row] * j + col_nnz + k
  //
  // program keeps col_nnz for block product, which is used later for
  // outer product computation.
  //
  // There is no special handling for diagonal blocks as we generate
  // BLOCK triangular matrix (diagonal block is full block) instead of
  // standard triangular matrix.

  const std::vector<int>& blocks = matrix_->col_blocks();
  for (int i = 0; i < blocks.size(); ++i) {
    for (int j = 0; j < blocks[i]; ++j, ++crsm_rows) {
      *(crsm_rows + 1) = *crsm_rows + row_nnz[i];
    }
  }

#define FILL_CRSM_COL_BLOCK                                      \
  const int row_block = current->row;                            \
  const int col_block = current->col;                            \
  const int nnz_in_row = row_nnz[row_block];                     \
  for (int j = 0; j < blocks[row_block]; ++j) {                  \
    for (int k = 0; k < blocks[col_block]; ++k) {                \
      crsm_cols[nnz + j * nnz_in_row + col_nnz + k] =            \
          col_blocks[col_block].position + k;                    \
    }                                                            \
  }                                                              \
  program_[current->index] = nnz + col_nnz;


  const std::vector<Block>& col_blocks = m_.block_structure()->cols;
  int nnz = 0;
  int col_nnz = 0;
  program_.resize(product_terms.size());

  const OuterProduct::ProductTerm* current;
  // Process first product term.
  current = &product_terms[0];
  FILL_CRSM_COL_BLOCK;

  // Process rest product terms.
  for (int i = 1; i < product_terms.size(); ++i) {
    current = &product_terms[i];
    const OuterProduct::ProductTerm* previous = &product_terms[i - 1];
    if (previous->row == current->row && previous->col == current->col) {
      // If the previous term is the same as the current one, nothing to do.
      program_[current->index] = program_[previous->index];
      continue;
    }

    col_nnz += blocks[previous->col];
    if (previous->row != current->row) {
      // Moved to a new row block, so we are starting at no entries in this row.
      col_nnz = 0;
      nnz += row_nnz[previous->row] * blocks[previous->row];
    }

    FILL_CRSM_COL_BLOCK;
  }
}

// TODO(sameeragarwal): Redo this comment.
//
// Note this function produces a triangular matrix in block unit (i.e.
// diagonal block is a normal block) instead of standard triangular matrix.
// So there is no special handling for diagonal blocks.
//
// TODO(sameeragarwal): Multithreading support.
void OuterProduct::ComputeProduct() {
  const CompressedRowSparseMatrix::StorageType storage_type =
      matrix_->storage_type();
  matrix_->SetZero();
  double* values = matrix_->mutable_values();
  const int* rows = matrix_->rows();

  int cursor = 0;
  const double* m_values = m_.values();
  const CompressedRowBlockStructure* bs = m_.block_structure();

  // Iterate row blocks.
  for (int r = start_row_block_; r < end_row_block_; ++r) {
    const CompressedRow& m_row = bs->rows[r];
    for (int c1 = 0; c1 < m_row.cells.size(); ++c1) {
      const Cell& cell1 = m_row.cells[c1];
      const int c1_size = bs->cols[cell1.block_id].size;
      int c2_begin, c2_end;
      const int* rows = matrix_->rows() + bs->cols[cell1.block_id].position;
      const int col_stride = rows[1] - rows[0];
      if (storage_type == CompressedRowSparseMatrix::LOWER_TRIANGULAR) {
        c2_begin = 0;
        c2_end = c1 + 1;
      } else {
        c2_begin = c1;
        c2_end = m_row.cells.size();
      }

      for (int c2 = c2_begin; c2 < c2_end; ++c2, ++cursor) {
        const Cell& cell2 = m_row.cells[c2];
        const int c2_size = bs->cols[cell2.block_id].size;
        MatrixTransposeMatrixMultiply(m_row.block.size,
                                      m_values + cell1.position, c1_size,
                                      m_values + cell2.position, c2_size,
                                      values + program_[cursor], col_stride);
      }
    }
  }

  CHECK_EQ(cursor, program_.size());
}

}  // namespace internal
}  // namespace ceres
