// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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

#include "ceres/compressed_row_sparse_matrix.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include "ceres/crs_matrix.h"
#include "ceres/internal/port.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::vector;

namespace {

// Helper functor used by the constructor for reordering the contents
// of a TripletSparseMatrix. This comparator assumes thay there are no
// duplicates in the pair of arrays rows and cols, i.e., there is no
// indices i and j (not equal to each other) s.t.
//
//  rows[i] == rows[j] && cols[i] == cols[j]
//
// If this is the case, this functor will not be a StrictWeakOrdering.
struct RowColLessThan {
  RowColLessThan(const int* rows, const int* cols)
      : rows(rows), cols(cols) {
  }

  bool operator()(const int x, const int y) const {
    if (rows[x] == rows[y]) {
      return (cols[x] < cols[y]);
    }
    return (rows[x] < rows[y]);
  }

  const int* rows;
  const int* cols;
};

}  // namespace

// This constructor gives you a semi-initialized CompressedRowSparseMatrix.
CompressedRowSparseMatrix::CompressedRowSparseMatrix(int num_rows,
                                                     int num_cols,
                                                     int max_num_nonzeros) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  rows_.resize(num_rows + 1, 0);
  cols_.resize(max_num_nonzeros, 0);
  values_.resize(max_num_nonzeros, 0.0);


  VLOG(1) << "# of rows: " << num_rows_
          << " # of columns: " << num_cols_
          << " max_num_nonzeros: " << cols_.size()
          << ". Allocating " << (num_rows_ + 1) * sizeof(int) +  // NOLINT
      cols_.size() * sizeof(int) +  // NOLINT
      cols_.size() * sizeof(double);  // NOLINT
}

CompressedRowSparseMatrix::CompressedRowSparseMatrix(
    const TripletSparseMatrix& m) {
  num_rows_ = m.num_rows();
  num_cols_ = m.num_cols();

  rows_.resize(num_rows_ + 1, 0);
  cols_.resize(m.num_nonzeros(), 0);
  values_.resize(m.max_num_nonzeros(), 0.0);

  // index is the list of indices into the TripletSparseMatrix m.
  vector<int> index(m.num_nonzeros(), 0);
  for (int i = 0; i < m.num_nonzeros(); ++i) {
    index[i] = i;
  }

  // Sort index such that the entries of m are ordered by row and ties
  // are broken by column.
  sort(index.begin(), index.end(), RowColLessThan(m.rows(), m.cols()));

  VLOG(1) << "# of rows: " << num_rows_
          << " # of columns: " << num_cols_
          << " max_num_nonzeros: " << cols_.size()
          << ". Allocating "
          << ((num_rows_ + 1) * sizeof(int) +  // NOLINT
              cols_.size() * sizeof(int) +     // NOLINT
              cols_.size() * sizeof(double));  // NOLINT

  // Copy the contents of the cols and values array in the order given
  // by index and count the number of entries in each row.
  for (int i = 0; i < m.num_nonzeros(); ++i) {
    const int idx = index[i];
    ++rows_[m.rows()[idx] + 1];
    cols_[i] = m.cols()[idx];
    values_[i] = m.values()[idx];
  }

  // Find the cumulative sum of the row counts.
  for (int i = 1; i < num_rows_ + 1; ++i) {
    rows_[i] += rows_[i - 1];
  }

  CHECK_EQ(num_nonzeros(), m.num_nonzeros());
}

CompressedRowSparseMatrix::CompressedRowSparseMatrix(const double* diagonal,
                                                     int num_rows) {
  CHECK_NOTNULL(diagonal);

  num_rows_ = num_rows;
  num_cols_ = num_rows;
  rows_.resize(num_rows + 1);
  cols_.resize(num_rows);
  values_.resize(num_rows);

  rows_[0] = 0;
  for (int i = 0; i < num_rows_; ++i) {
    cols_[i] = i;
    values_[i] = diagonal[i];
    rows_[i + 1] = i + 1;
  }

  CHECK_EQ(num_nonzeros(), num_rows);
}

CompressedRowSparseMatrix::~CompressedRowSparseMatrix() {
}

void CompressedRowSparseMatrix::SetZero() {
  std::fill(values_.begin(), values_.end(), 0);
}

void CompressedRowSparseMatrix::RightMultiply(const double* x,
                                              double* y) const {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(y);

  for (int r = 0; r < num_rows_; ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
      y[r] += values_[idx] * x[cols_[idx]];
    }
  }
}

void CompressedRowSparseMatrix::LeftMultiply(const double* x, double* y) const {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(y);

  for (int r = 0; r < num_rows_; ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
      y[cols_[idx]] += values_[idx] * x[r];
    }
  }
}

void CompressedRowSparseMatrix::SquaredColumnNorm(double* x) const {
  CHECK_NOTNULL(x);

  std::fill(x, x + num_cols_, 0.0);
  for (int idx = 0; idx < rows_[num_rows_]; ++idx) {
    x[cols_[idx]] += values_[idx] * values_[idx];
  }
}

void CompressedRowSparseMatrix::ScaleColumns(const double* scale) {
  CHECK_NOTNULL(scale);

  for (int idx = 0; idx < rows_[num_rows_]; ++idx) {
    values_[idx] *= scale[cols_[idx]];
  }
}

void CompressedRowSparseMatrix::ToDenseMatrix(Matrix* dense_matrix) const {
  CHECK_NOTNULL(dense_matrix);
  dense_matrix->resize(num_rows_, num_cols_);
  dense_matrix->setZero();

  for (int r = 0; r < num_rows_; ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
      (*dense_matrix)(r, cols_[idx]) = values_[idx];
    }
  }
}

void CompressedRowSparseMatrix::DeleteRows(int delta_rows) {
  CHECK_GE(delta_rows, 0);
  CHECK_LE(delta_rows, num_rows_);

  num_rows_ -= delta_rows;
  rows_.resize(num_rows_ + 1);

  // Walk the list of row blocks until we reach the new number of rows
  // and the drop the rest of the row blocks.
  int num_row_blocks = 0;
  int num_rows = 0;
  while (num_row_blocks < row_blocks_.size() && num_rows < num_rows_) {
    num_rows += row_blocks_[num_row_blocks];
    ++num_row_blocks;
  }

  row_blocks_.resize(num_row_blocks);
}

void CompressedRowSparseMatrix::AppendRows(const CompressedRowSparseMatrix& m) {
  CHECK_EQ(m.num_cols(), num_cols_);

  CHECK(row_blocks_.size() == 0 || m.row_blocks().size() !=0)
      << "Cannot append a matrix with row blocks to one without and vice versa."
      << "This matrix has : " << row_blocks_.size() << " row blocks."
      << "The matrix being appended has: " << m.row_blocks().size()
      << " row blocks.";

  if (m.num_rows() == 0) {
    return;
  }

  if (cols_.size() < num_nonzeros() + m.num_nonzeros()) {
    cols_.resize(num_nonzeros() + m.num_nonzeros());
    values_.resize(num_nonzeros() + m.num_nonzeros());
  }

  // Copy the contents of m into this matrix.
  DCHECK_LT(num_nonzeros(), cols_.size());
  if (m.num_nonzeros() > 0) {
    std::copy(m.cols(), m.cols() + m.num_nonzeros(), &cols_[num_nonzeros()]);
    std::copy(m.values(),
              m.values() + m.num_nonzeros(),
              &values_[num_nonzeros()]);
  }

  rows_.resize(num_rows_ + m.num_rows() + 1);
  // new_rows = [rows_, m.row() + rows_[num_rows_]]
  std::fill(rows_.begin() + num_rows_,
            rows_.begin() + num_rows_ + m.num_rows() + 1,
            rows_[num_rows_]);

  for (int r = 0; r < m.num_rows() + 1; ++r) {
    rows_[num_rows_ + r] += m.rows()[r];
  }

  num_rows_ += m.num_rows();
  row_blocks_.insert(row_blocks_.end(),
                     m.row_blocks().begin(),
                     m.row_blocks().end());

  // Generate crsb rows and cols.
  if (m.crsb_rows_.size() == 0) {
    return;
  }

  const int num_crsb_nonzeros = crsb_cols_.size();
  const int m_num_crsb_nonzeros = m.crsb_cols_.size();
  crsb_cols_.resize(num_crsb_nonzeros + m_num_crsb_nonzeros);
  std::copy(&m.crsb_cols()[0], &m.crsb_cols()[0] + m_num_crsb_nonzeros,
            &crsb_cols_[num_crsb_nonzeros]);

  const int num_crsb_rows = crsb_rows_.size() - 1;
  const int m_num_crsb_rows = m.crsb_rows_.size() - 1;
  crsb_rows_.resize(num_crsb_rows + m_num_crsb_rows + 1);
  std::fill(crsb_rows_.begin() + num_crsb_rows,
            crsb_rows_.begin() + num_crsb_rows + m_num_crsb_rows + 1,
            crsb_rows_[num_crsb_rows]);

  for (int r = 0; r < m_num_crsb_rows + 1; ++r) {
    crsb_rows_[num_crsb_rows + r] += m.crsb_rows()[r];
  }
}

void CompressedRowSparseMatrix::ToTextFile(FILE* file) const {
  CHECK_NOTNULL(file);
  for (int r = 0; r < num_rows_; ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
      fprintf(file,
              "% 10d % 10d %17f\n",
              r,
              cols_[idx],
              values_[idx]);
    }
  }
}

void CompressedRowSparseMatrix::ToCRSMatrix(CRSMatrix* matrix) const {
  matrix->num_rows = num_rows_;
  matrix->num_cols = num_cols_;
  matrix->rows = rows_;
  matrix->cols = cols_;
  matrix->values = values_;

  // Trim.
  matrix->rows.resize(matrix->num_rows + 1);
  matrix->cols.resize(matrix->rows[matrix->num_rows]);
  matrix->values.resize(matrix->rows[matrix->num_rows]);
}

void CompressedRowSparseMatrix::SetMaxNumNonZeros(int num_nonzeros) {
  CHECK_GE(num_nonzeros, 0);

  cols_.resize(num_nonzeros);
  values_.resize(num_nonzeros);
}

void CompressedRowSparseMatrix::SolveLowerTriangularInPlace(
    double* solution) const {
  for (int r = 0; r < num_rows_; ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1] - 1; ++idx) {
      solution[r] -= values_[idx] * solution[cols_[idx]];
    }
    solution[r] /=  values_[rows_[r + 1] - 1];
  }
}

void CompressedRowSparseMatrix::SolveLowerTriangularTransposeInPlace(
    double* solution) const {
  for (int r = num_rows_ - 1; r >= 0; --r) {
    solution[r] /= values_[rows_[r + 1] - 1];
    for (int idx = rows_[r + 1] - 2; idx >= rows_[r]; --idx) {
      solution[cols_[idx]] -= values_[idx] * solution[r];
    }
  }
}

CompressedRowSparseMatrix* CompressedRowSparseMatrix::CreateBlockDiagonalMatrix(
    const double* diagonal,
    const vector<int>& blocks) {
  int num_rows = 0;
  int num_nonzeros = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    num_rows += blocks[i];
    num_nonzeros += blocks[i] * blocks[i];
  }

  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(num_rows, num_rows, num_nonzeros);

  int* rows = matrix->mutable_rows();
  int* cols = matrix->mutable_cols();
  double* values = matrix->mutable_values();
  std::fill(values, values + num_nonzeros, 0.0);

  int idx_cursor = 0;
  int col_cursor = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    const int block_size = blocks[i];
    for (int r = 0; r < block_size; ++r) {
      *(rows++) = idx_cursor;
      values[idx_cursor + r] = diagonal[col_cursor + r];
      for (int c = 0; c < block_size; ++c, ++idx_cursor) {
        *(cols++) = col_cursor + c;
      }
    }
    col_cursor += block_size;
  }
  *rows = idx_cursor;

  *matrix->mutable_row_blocks() = blocks;
  *matrix->mutable_col_blocks() = blocks;

  // Generate crsb.
  vector<int>& crsb_rows = *matrix->mutable_crsb_rows();
  vector<int>& crsb_cols = *matrix->mutable_crsb_cols();
  for (int i = 0; i < blocks.size(); ++i) {
    crsb_rows.push_back(i);
    crsb_cols.push_back(i);
  }
  crsb_rows.push_back(blocks.size());

  CHECK_EQ(idx_cursor, num_nonzeros);
  CHECK_EQ(col_cursor, num_rows);
  return matrix;
}

CompressedRowSparseMatrix* CompressedRowSparseMatrix::Transpose() const {
  CompressedRowSparseMatrix* transpose =
      new CompressedRowSparseMatrix(num_cols_, num_rows_, num_nonzeros());

  int* transpose_rows = transpose->mutable_rows();
  int* transpose_cols = transpose->mutable_cols();
  double* transpose_values = transpose->mutable_values();

  for (int idx = 0; idx < num_nonzeros(); ++idx) {
    ++transpose_rows[cols_[idx] + 1];
  }

  for (int i = 1; i < transpose->num_rows() + 1; ++i) {
    transpose_rows[i] += transpose_rows[i - 1];
  }

  for (int r = 0; r < num_rows(); ++r) {
    for (int idx = rows_[r]; idx < rows_[r + 1]; ++idx) {
      const int c = cols_[idx];
      const int transpose_idx = transpose_rows[c]++;
      transpose_cols[transpose_idx] = r;
      transpose_values[transpose_idx] = values_[idx];
    }
  }

  for (int i = transpose->num_rows() - 1; i > 0 ; --i) {
    transpose_rows[i] = transpose_rows[i - 1];
  }
  transpose_rows[0] = 0;

  *(transpose->mutable_row_blocks()) = col_blocks_;
  *(transpose->mutable_col_blocks()) = row_blocks_;

  return transpose;
}

namespace {
// A ProductTerm is a term in the outer product of a matrix with
// itself.
struct ProductTerm {
  ProductTerm(const int row, const int col, const int index)
      : row(row), col(col), index(index) {
  }

  bool operator<(const ProductTerm& right) const {
    if (row == right.row) {
      if (col == right.col) {
        return index < right.index;
      }
      return col < right.col;
    }
    return row < right.row;
  }

  int row;
  int col;
  int index;
};

CompressedRowSparseMatrix*
CompressAndFillProgram(const int num_rows,
                       const int num_cols,
                       const vector<int>& blocks,
                       const vector<ProductTerm>& product,
                       vector<int>* program) {
  CHECK_GT(product.size(), 0);

  // Count the number of unique product term, which in turn is the
  // number of non-zeros in the outer product.
  // Also count the number of non-zeros in each row block.
  vector<int> row_nnz(blocks.size());
  std::fill(row_nnz.begin(), row_nnz.end(), 0);
  row_nnz[product[0].row] = blocks[product[0].col];
  int num_nonzeros = blocks[product[0].row] * blocks[product[0].col];
  for (int i = 1; i < product.size(); ++i) {
    if (product[i].row != product[i - 1].row ||
        product[i].col != product[i - 1].col) {
      row_nnz[product[i].row] += blocks[product[i].col];
      num_nonzeros += blocks[product[i].row] * blocks[product[i].col];
    }
  }

  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(num_rows, num_cols, num_nonzeros);

  // Compute block offsets.
  vector<int>& block_offsets = *(matrix->mutable_block_offsets());
  block_offsets.resize(blocks.size() + 1);
  block_offsets[0] = 0;
  for (int block = 0; block < blocks.size(); ++block) {
    block_offsets[block + 1] = block_offsets[block] + blocks[block];
  }

  int* crsm_rows = matrix->mutable_rows();
  std::fill(crsm_rows, crsm_rows + num_rows + 1, 0);
  int* crsm_cols = matrix->mutable_cols();
  std::fill(crsm_cols, crsm_cols + num_nonzeros, 0);

  CHECK_NOTNULL(program)->clear();
  program->resize(product.size());

  // Non zero elements are not stored consecutively across rows in a block.
  // We seperate nonzero into three categories:
  //   nonzeros in all previous row blocks counted in nnz
  //   nonzeros in current row block counted in row_nnz
  //   nonzeros in current row block and previous col blocks counted in col_nnz
  // Then the total number of nonzero at (row, col) of current block is
  //   nnz + row_nnz[current.row] * row + col_nnz + col
  int nnz = 0;
  int col_nnz = 0;

  // Process first product term.
  for (int row = 0; row < blocks[product[0].row]; ++row) {
    for (int col = 0; col < blocks[product[0].col]; ++col) {
      crsm_cols[row_nnz[product[0].row] * row + col]
          = block_offsets[product[0].col] + col;
    }
    crsm_rows[block_offsets[product[0].row] + row + 1] =
        row_nnz[product[0].row];
  }
  (*program)[product[0].index] = 0;

  // Process rest product terms.
  for (int i = 1; i < product.size(); ++i) {
    const ProductTerm& previous = product[i - 1];
    const ProductTerm& current = product[i];

    // Sparsity structure is updated only if the term is not a repeat.
    if (previous.row != current.row || previous.col != current.col) {
      col_nnz += blocks[previous.col];
      if (previous.row != current.row) {
        CHECK_EQ(col_nnz, row_nnz[previous.row]);
        nnz += col_nnz * blocks[previous.row];
        col_nnz = 0;

        for (int row = 0; row < blocks[current.row]; ++row) {
          crsm_rows[block_offsets[current.row] + row + 1] =
              row_nnz[current.row];
        }
      }

      for (int row = 0; row < blocks[current.row]; ++row) {
        for (int col = 0; col < blocks[current.col]; ++col) {
          crsm_cols[nnz + row_nnz[current.row] * row + col_nnz + col]
              = block_offsets[current.col] + col;
        }
      }
    }

    // Non zero elements are not stored consecutively across rows in a block.
    // Here we assign program the nonzero position of the first row of the block.
    // ComputeOuterProduct will consider the gaps between rows.
    (*program)[current.index] = nnz + col_nnz;
  }

  for (int i = 1; i < num_rows + 1; ++i) {
    crsm_rows[i] += crsm_rows[i - 1];
  }

  // Sanity checks for last product term.
  col_nnz += blocks[product.back().col];
  CHECK_EQ(col_nnz, row_nnz[product.back().row]);
  nnz += col_nnz * blocks[product.back().row];
  CHECK_EQ(num_nonzeros, nnz);
  CHECK_EQ(num_nonzeros, crsm_rows[num_rows]);

  return matrix;
}

// Implement block multiplication:
//
//   MatrixRef H(output, col_size1, output_cols)
//   ConstMatrixRef J(input, row_size, input_cols)
//
//   H(0, 0, col_size1, col_size2) +=
//      J(0, col_begin1, row_size, col_size1).transpose()
//    * J(0, col_begin2, row_size, col_size2)
void ComputerBlockOuterProduct(
  double *output, const int output_cols,
  const double *input, const int input_cols,
  const int col_begin1, const int col_begin2,
  const int row_size, const int col_size1, const int col_size2) {
  for (int r = 0; r < row_size; ++r) {
    for (int idx1 = 0; idx1 < col_size1; ++idx1) {
      for (int idx2 = 0; idx2 < col_size2; ++idx2) {
        output[output_cols * idx1 + idx2] +=
            input[input_cols * r + col_begin1 + idx1] *
            input[input_cols * r + col_begin2 + idx2];
      }
    }
  }
}
}  // namespace

CompressedRowSparseMatrix*
CompressedRowSparseMatrix::CreateOuterProductMatrixAndProgram(
      const CompressedRowSparseMatrix& m,
      const int stype,
      vector<int>* program) {
  CHECK_NOTNULL(program)->clear();
  CHECK_GT(m.num_nonzeros(), 0)
                << "Congratulations, "
                << "you found a bug in Ceres. Please report it.";

  vector<ProductTerm> product;
  const vector<int>& col_blocks = m.col_blocks();
  const vector<int>& crsb_rows = m.crsb_rows();
  const vector<int>& crsb_cols = m.crsb_cols();

  // Give input matrix m in Compressed Row Sparse Block format
  //     (row_block, idx)
  // represent each block multiplication
  //     (row_block, idx1)' X (row_block, idx2)
  // by its product term index and sort the product terms
  //     (crsb_cols[idx1], crsb_cols[idx2], index)
  for (int row_block = 1; row_block < crsb_rows.size(); ++row_block) {
    for (int idx1 = crsb_rows[row_block - 1]; idx1 < crsb_rows[row_block];
         ++idx1) {
      if (stype > 0) {
        for (int idx2 = crsb_rows[row_block - 1]; idx2 <= idx1; ++idx2) {
          product.push_back(ProductTerm(crsb_cols[idx1], crsb_cols[idx2],
                                        product.size()));
        }
      }
      else {
        for (int idx2 = idx1; idx2 < crsb_rows[row_block]; ++idx2) {
          product.push_back(ProductTerm(crsb_cols[idx1], crsb_cols[idx2],
                                        product.size()));
        }
      }
    }
  }

  sort(product.begin(), product.end());
  return CompressAndFillProgram(m.num_cols(), m.num_cols(),
                                col_blocks, product, program);
}

// Give input matrix m in Compressed Row Sparse Block format
//     (row_block, col_block_idx)
// compute outerproduct m' * m as sum of block multiplications
//     (row_block, col_block1_idx)' X (row_block, col_block2_idx)
//
// Note this function does not product a strict upper/lower matrix, but
// a uppper/lower BLOCK matrix (i.e. the diagnonals are not single elements,
// but blocks).
//
// input matrix block (row_block, col_block1_idx) is represented as
//     block(0, idx_begin,
//           row_blocks[row_block], col_blocks[crsb_cols[col_block1_idx]])
// of matrix at
//     m.values()[m.rows()[row_block_begin]]
// with dimensions
//     <row_blocks[row_block], row_block_nnz>
//
// Input program maps each block multiplication
//     (row_block, col_block1_idx)' X (row_block, col_block2_idx)
// represented by cursor to
//     block(0, 0,
//           col_blocks[crsb_cols[col_block1_idx]],
//           col_blocks[crsb_cols[col_block2_idx]])
// of outerproduct matrix at
//     result->values()[program[cursor]]
// with dimensions
//     <col_blocks[crsb_cols[col_block1_idx], col_block1_nnz>
//
void CompressedRowSparseMatrix::ComputeOuterProduct(
    const CompressedRowSparseMatrix& m,
    const int stype,
    const vector<int>& program,
    CompressedRowSparseMatrix* result) {
  result->SetZero();
  double* values = result->mutable_values();
  const int* rows = result->rows();
  const vector<int>& block_offsets = result->block_offsets();

  int cursor = 0;
  const double* m_values = m.values();
  const int* m_rows = m.rows();
  const vector<int>& row_blocks = m.row_blocks();
  const vector<int>& col_blocks = m.col_blocks();
  const vector<int>& crsb_rows = m.crsb_rows();
  const vector<int>& crsb_cols = m.crsb_cols();

  // Iterate row blocks.
  for (int row_block = 0, row_block_begin = 0;
       row_block < row_blocks.size();
       row_block_begin += row_blocks[row_block++]) {
    // Non zeros are not stored consecutively across rows in a block.
    // The gaps between rows is the number of nonzeros of the
    // input matrix compressed row.
    const int row_block_nnz =
        m_rows[row_block_begin + 1] - m_rows[row_block_begin];

    // Iterate (col_block1_idx x col_block2_idx).
    for (int col_block1_idx = crsb_rows[row_block], idx1_begin = 0;
         col_block1_idx < crsb_rows[row_block + 1];
         idx1_begin += col_blocks[crsb_cols[col_block1_idx++]]) {
      // Non zeros are not stored consecutively across rows in a block.
      // The gaps between rows is the number of nonzeros of the
      // outerproduct matrix compressed row.
      const int row_begin = block_offsets[crsb_cols[col_block1_idx]];
      const int col_block1_nnz = rows[row_begin + 1] - rows[row_begin];

      // Handle upper/lower triangular blocks.
      if (stype > 0) {
        for (int col_block2_idx = crsb_rows[row_block], idx2_begin = 0;
             col_block2_idx <= col_block1_idx; ++cursor,
             idx2_begin += col_blocks[crsb_cols[col_block2_idx++]]) {
          ComputerBlockOuterProduct(
            values + program[cursor], col_block1_nnz,
            m_values + m_rows[row_block_begin], row_block_nnz,
            idx1_begin, idx2_begin,
            row_blocks[row_block],
            col_blocks[crsb_cols[col_block1_idx]],
            col_blocks[crsb_cols[col_block2_idx]]);
        }
      }
      else {
        for (int col_block2_idx = col_block1_idx, idx2_begin = idx1_begin;
             col_block2_idx < crsb_rows[row_block + 1]; ++cursor,
             idx2_begin += col_blocks[crsb_cols[col_block2_idx++]]) {
          ComputerBlockOuterProduct(
            values + program[cursor], col_block1_nnz,
            m_values + m_rows[row_block_begin], row_block_nnz,
            idx1_begin, idx2_begin,
            row_blocks[row_block],
            col_blocks[crsb_cols[col_block1_idx]],
            col_blocks[crsb_cols[col_block2_idx]]);
        }
      }
    }
  }

  CHECK_EQ(cursor, program.size());
}

}  // namespace internal
}  // namespace ceres
