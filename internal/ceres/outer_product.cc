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

// input is a matrix of dimesion <row_block_size, input_cols>
// output is a matrix of dimension <col_block1_size, output_cols>
//
// Implement block multiplication O = I1' * I2.
// I1 is block(0, col_block1_begin, row_block_size, col_block1_size) of input
// I2 is block(0, col_block2_begin, row_block_size, col_block2_size) of input
// O is block(0, 0, col_block1_size, col_block2_size) of output
void ComputeBlockMultiplication(const int row_block_size,
                                const int col_block1_size,
                                const int col_block2_size,
                                const int col_block1_begin,
                                const int col_block2_begin,
                                const int input_cols,
                                const double* input,
                                const int output_cols,
                                double* output) {
  for (int r = 0; r < row_block_size; ++r) {
    for (int idx1 = 0; idx1 < col_block1_size; ++idx1) {
      for (int idx2 = 0; idx2 < col_block2_size; ++idx2) {
        output[output_cols * idx1 + idx2] +=
            input[input_cols * r + col_block1_begin + idx1] *
            input[input_cols * r + col_block2_begin + idx2];
      }
    }
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
  const std::vector<int> blocks = m_.col_blocks();
  row_nnz->resize(blocks.size());
  std::fill(row_nnz->begin(), row_nnz->end(), 0);
  (*row_nnz)[product_terms[0].row] = blocks[product_terms[0].col];
  int num_nonzeros =
      blocks[product_terms[0].row] * blocks[product_terms[0].col];
  for (int i = 1; i < product_terms.size(); ++i) {
    const ProductTerm& previous = product_terms[i - 1];
    const ProductTerm& current = product_terms[i];

    // Each (row, col) block counts only once.
    // This check depends on product sorted on (row, col).
    if (current.row != previous.row || current.col != previous.col) {
      (*row_nnz)[current.row] += blocks[current.col];
      num_nonzeros += blocks[current.row] * blocks[current.col];
    }
  }

  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(m_.num_cols(), m_.num_cols(), num_nonzeros);
  matrix->set_storage_type(storage_type);
  *(matrix->mutable_row_blocks()) = blocks;
  *(matrix->mutable_col_blocks()) = blocks;

  // Compute block offsets for outer product matrix, which is used in
  // ComputeOuterProduct.
  block_offsets_.resize(blocks.size() + 1);
  block_offsets_[0] = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    block_offsets_[i + 1] = block_offsets_[i] + blocks[i];
  }

  return matrix;
}

OuterProduct::ProductTerm::ProductTerm(const int row,
                                       const int col,
                                       const int index)
    : row(row), col(col), index(index) {}

bool OuterProduct::ProductTerm::operator<(
    const OuterProduct::ProductTerm& right) const {
  if (row == right.row) {
    if (col == right.col) {
      return index < right.index;
    }
    return col < right.col;
  }
  return row < right.row;
}

OuterProduct::OuterProduct(const CompressedRowSparseMatrix& m,
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
    const CompressedRowSparseMatrix& m,
    CompressedRowSparseMatrix::StorageType product_storage_type) {
  return OuterProduct::Create(
      m, 0, m.row_blocks().size(), product_storage_type);
}

OuterProduct* OuterProduct::Create(
    const CompressedRowSparseMatrix& m,
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

  const std::vector<int>& crsb_rows = m_.crsb_rows();
  const std::vector<int>& crsb_cols = m_.crsb_cols();

  // Give input matrix m in Compressed Row Sparse Block format
  //     (row_block, col_block)
  // represent each block multiplication
  //     (row_block, col_block1)' X (row_block, col_block2)
  // by its product term index and sort the product terms
  //     (col_block1, col_block2, index)
  //
  // Due to the compression on rows, col_block is accessed through idx to
  // crsb_cols.  So col_block is accessed as crsb_cols[idx] in the code.
  for (int row_block = start_row_block_ + 1; row_block < end_row_block_ + 1; ++row_block) {
    for (int idx1 = crsb_rows[row_block - 1]; idx1 < crsb_rows[row_block];
         ++idx1) {
      if (product_storage_type == CompressedRowSparseMatrix::LOWER_TRIANGULAR) {
        for (int idx2 = crsb_rows[row_block - 1]; idx2 <= idx1; ++idx2) {
          product_terms.push_back(OuterProduct::ProductTerm(
              crsb_cols[idx1], crsb_cols[idx2], product_terms.size()));
        }
      } else {  // Upper triangular matrix.
        for (int idx2 = idx1; idx2 < crsb_rows[row_block]; ++idx2) {
          product_terms.push_back(OuterProduct::ProductTerm(
              crsb_cols[idx1], crsb_cols[idx2], product_terms.size()));
        }
      }
    }
  }

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

  const std::vector<int>& blocks = matrix_->col_blocks();

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
  int nnz = 0;
  int col_nnz = 0;
  program_.resize(product_terms.size());
  program_[product_terms[0].index] = 0;

  // Process first product term.
  for (int j = 0; j < blocks[product_terms[0].row]; ++j) {
    crsm_rows[block_offsets_[product_terms[0].row] + j + 1] =
        row_nnz[product_terms[0].row];
    for (int k = 0; k < blocks[product_terms[0].col]; ++k) {
      crsm_cols[row_nnz[product_terms[0].row] * j + k] =
          block_offsets_[product_terms[0].col] + k;
    }
  }

  // Process rest product terms.
  for (int i = 1; i < product_terms.size(); ++i) {
    const OuterProduct::ProductTerm& previous = product_terms[i - 1];
    const OuterProduct::ProductTerm& current = product_terms[i];

    // Sparsity structure is updated only if the term is not a repeat.
    if (previous.row != current.row || previous.col != current.col) {
      col_nnz += blocks[previous.col];
      if (previous.row != current.row) {
        nnz += col_nnz * blocks[previous.row];
        col_nnz = 0;

        for (int j = 0; j < blocks[current.row]; ++j) {
          crsm_rows[block_offsets_[current.row] + j + 1] = row_nnz[current.row];
        }
      }

      for (int j = 0; j < blocks[current.row]; ++j) {
        for (int k = 0; k < blocks[current.col]; ++k) {
          crsm_cols[nnz + row_nnz[current.row] * j + col_nnz + k] =
              block_offsets_[current.col] + k;
        }
      }
    }

    program_[current.index] = col_nnz;
  }

  for (int i = 1; i < num_cols + 1; ++i) {
    crsm_rows[i] += crsm_rows[i - 1];
  }
}

// Give input matrix m in Compressed Row Sparse Block format
//     (row_block, col_block)
// compute outer product m' * m as sum of block multiplications
//     (row_block, col_block1)' X (row_block, col_block2)
//
// Given row_block of the input matrix m, we use m_row_begin to represent
// the starting row of the row block and m_row_nnz to represent number of
// nonzero in each row of the row block, then the rows belonging to
// the row block can be represented as a dense matrix starting at
//     m.values() + m.rows()[m_row_begin]
// with dimension
//     <m.row_blocks()[row_block], m_row_nnz>
//
// Then each input matrix block (row_block, col_block) can be represented as
// a block of above dense matrix starting at position
//     (0, m_col_nnz)
// with size
//     <m.row_blocks()[row_block], m.col_blocks()[col_block]>
// where m_col_nnz is the number of nonzero before col_block in each row.
//
// The outer product block is represented similarly with m_row_begin,
// m_row_nnz, m_col_nnz, etc. replaced by row_begin, row_nnz, col_nnz,
// etc. The difference is, m_row_begin and m_col_nnz is counted
// during the traverse of block multiplication, while row_begin and
// col_nnz are got from pre-computed block_offsets and program.
//
// Due to the compression on rows, col_block is accessed through
// idx to crsb_col std::vector. So col_block is accessed as crsb_col[idx]
// in the code.
//
// Note this function produces a triangular matrix in block unit (i.e.
// diagonal block is a normal block) instead of standard triangular matrix.
// So there is no special handling for diagonal blocks.
//
// TODO(sameeragarwal): Multithreading support.
void OuterProduct::ComputeProduct() {
  matrix_->SetZero();
  double* values = matrix_->mutable_values();
  const int* rows = matrix_->rows();

  int cursor = 0;
  const double* m_values = m_.values();
  const int* m_rows = m_.rows();
  const std::vector<int>& row_blocks = m_.row_blocks();
  const std::vector<int>& col_blocks = m_.col_blocks();
  const std::vector<int>& crsb_rows = m_.crsb_rows();
  const std::vector<int>& crsb_cols = m_.crsb_cols();
  const CompressedRowSparseMatrix::StorageType storage_type =
      matrix_->storage_type();
#define COL_BLOCK1 (crsb_cols[idx1])
#define COL_BLOCK2 (crsb_cols[idx2])

  int m_row_begin = 0;
  for (int i = 0; i < start_row_block_; ++i) {
    m_row_begin += row_blocks[i];
  }

  // Iterate row blocks.
  for (int row_block = start_row_block_;
       row_block < end_row_block_;
       m_row_begin += row_blocks[row_block++]) {
    // Non zeros are not stored consecutively across rows in a block.
    // The gaps between rows is the number of nonzeros of the
    // input matrix compressed row.
    const int m_row_nnz = m_rows[m_row_begin + 1] - m_rows[m_row_begin];

    // Iterate (col_block1 x col_block2).
    for (int idx1 = crsb_rows[row_block], m_col_nnz1 = 0;
         idx1 < crsb_rows[row_block + 1];
         m_col_nnz1 += col_blocks[COL_BLOCK1], ++idx1) {
      // Non zeros are not stored consecutively across rows in a
      // block. The gaps between rows is the number of nonzeros of the
      // outer product matrix compressed row.
      const int row_begin = block_offsets_[COL_BLOCK1];
      const int row_nnz = rows[row_begin + 1] - rows[row_begin];
      if (storage_type == CompressedRowSparseMatrix::LOWER_TRIANGULAR) {
        for (int idx2 = crsb_rows[row_block], m_col_nnz2 = 0; idx2 <= idx1;
             m_col_nnz2 += col_blocks[COL_BLOCK2], ++idx2, ++cursor) {
          int col_nnz = program_[cursor];
          ComputeBlockMultiplication(row_blocks[row_block],
                                     col_blocks[COL_BLOCK1],
                                     col_blocks[COL_BLOCK2],
                                     m_col_nnz1,
                                     m_col_nnz2,
                                     m_row_nnz,
                                     m_values + m_rows[m_row_begin],
                                     row_nnz,
                                     values + rows[row_begin] + col_nnz);
        }
      } else {
        for (int idx2 = idx1, m_col_nnz2 = m_col_nnz1;
             idx2 < crsb_rows[row_block + 1];
             m_col_nnz2 += col_blocks[COL_BLOCK2], ++idx2, ++cursor) {
          int col_nnz = program_[cursor];
          ComputeBlockMultiplication(row_blocks[row_block],
                                     col_blocks[COL_BLOCK1],
                                     col_blocks[COL_BLOCK2],
                                     m_col_nnz1,
                                     m_col_nnz2,
                                     m_row_nnz,
                                     m_values + m_rows[m_row_begin],
                                     row_nnz,
                                     values + rows[row_begin] + col_nnz);
        }
      }
    }
  }

#undef COL_BLOCK1
#undef COL_BLOCK2

  CHECK_EQ(cursor, program_.size());
}

}  // namespace internal
}  // namespace ceres
