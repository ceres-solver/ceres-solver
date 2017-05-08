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

#include "ceres/compressed_row_sparse_matrix.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include "ceres/crs_matrix.h"
#include "ceres/internal/port.h"
#include "ceres/random.h"
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
  RowColLessThan(const int* rows, const int* cols) : rows(rows), cols(cols) {}

  bool operator()(const int x, const int y) const {
    if (rows[x] == rows[y]) {
      return (cols[x] < cols[y]);
    }
    return (rows[x] < rows[y]);
  }

  const int* rows;
  const int* cols;
};

void TransposeForCompressedRowSparseStructure(const int num_rows,
                                              const int num_cols,
                                              const int num_nonzeros,
                                              const int* rows,
                                              const int* cols,
                                              const double* values,
                                              int* transpose_rows,
                                              int* transpose_cols,
                                              double* transpose_values) {
  // Explicitly zero out transpose_rows.
  std::fill(transpose_rows, transpose_rows + num_cols + 1, 0);

  // Count the number of entries in each column of the original matrix
  // and assign to transpose_rows[col + 1].
  for (int idx = 0; idx < num_nonzeros; ++idx) {
    ++transpose_rows[cols[idx] + 1];
  }

  // Compute the starting position for each row in the transpose by
  // computing the cumulative sum of the entries of transpose_rows.
  for (int i = 1; i < num_cols + 1; ++i) {
    transpose_rows[i] += transpose_rows[i - 1];
  }

  // Populate transpose_cols and (optionally) transpose_values by
  // walking the entries of the source matrices. For each entry that
  // is added, the value of transpose_row is incremented allowing us
  // to keep track of where the next entry for that row should go.
  //
  // As a result transpose_row is shifted to the left by one entry.
  for (int r = 0; r < num_rows; ++r) {
    for (int idx = rows[r]; idx < rows[r + 1]; ++idx) {
      const int c = cols[idx];
      const int transpose_idx = transpose_rows[c]++;
      transpose_cols[transpose_idx] = r;
      if (values != NULL && transpose_values != NULL) {
        transpose_values[transpose_idx] = values[idx];
      }
    }
  }

  // This loop undoes the left shift to transpose_rows introduced by
  // the previous loop.
  for (int i = num_cols - 1; i > 0; --i) {
    transpose_rows[i] = transpose_rows[i - 1];
  }
  transpose_rows[0] = 0;
}

void AddRandomBlock(const int num_rows,
                    const int num_cols,
                    const int row_block_begin,
                    const int col_block_begin,
                    std::vector<int>* rows,
                    std::vector<int>* cols,
                    std::vector<double>* values) {
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      rows->push_back(row_block_begin + r);
      cols->push_back(col_block_begin + c);
      values->push_back(RandNormal());
    }
  }
}

}  // namespace

// This constructor gives you a semi-initialized CompressedRowSparseMatrix.
CompressedRowSparseMatrix::CompressedRowSparseMatrix(int num_rows,
                                                     int num_cols,
                                                     int max_num_nonzeros) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  storage_type_ = UNSYMMETRIC;
  rows_.resize(num_rows + 1, 0);
  cols_.resize(max_num_nonzeros, 0);
  values_.resize(max_num_nonzeros, 0.0);

  VLOG(1) << "# of rows: " << num_rows_ << " # of columns: " << num_cols_
          << " max_num_nonzeros: " << cols_.size() << ". Allocating "
          << (num_rows_ + 1) * sizeof(int) +     // NOLINT
                 cols_.size() * sizeof(int) +    // NOLINT
                 cols_.size() * sizeof(double);  // NOLINT
}

CompressedRowSparseMatrix* CompressedRowSparseMatrix::FromTripletSparseMatrix(
    const TripletSparseMatrix& input) {
  return CompressedRowSparseMatrix::FromTripletSparseMatrix(input, false);
}

CompressedRowSparseMatrix*
CompressedRowSparseMatrix::FromTripletSparseMatrixTransposed(
    const TripletSparseMatrix& input) {
  return CompressedRowSparseMatrix::FromTripletSparseMatrix(input, true);
}

CompressedRowSparseMatrix* CompressedRowSparseMatrix::FromTripletSparseMatrix(
    const TripletSparseMatrix& input, bool transpose) {
  int num_rows = input.num_rows();
  int num_cols = input.num_cols();
  const int* rows = input.rows();
  const int* cols = input.cols();
  const double* values = input.values();

  if (transpose) {
    std::swap(num_rows, num_cols);
    std::swap(rows, cols);
  }

  // index is the list of indices into the TripletSparseMatrix input.
  vector<int> index(input.num_nonzeros(), 0);
  for (int i = 0; i < input.num_nonzeros(); ++i) {
    index[i] = i;
  }

  // Sort index such that the entries of m are ordered by row and ties
  // are broken by column.
  std::sort(index.begin(), index.end(), RowColLessThan(rows, cols));

  VLOG(1) << "# of rows: " << num_rows << " # of columns: " << num_cols
          << " num_nonzeros: " << input.num_nonzeros() << ". Allocating "
          << ((num_rows + 1) * sizeof(int) +           // NOLINT
              input.num_nonzeros() * sizeof(int) +     // NOLINT
              input.num_nonzeros() * sizeof(double));  // NOLINT

  CompressedRowSparseMatrix* output =
      new CompressedRowSparseMatrix(num_rows, num_cols, input.num_nonzeros());

  // Copy the contents of the cols and values array in the order given
  // by index and count the number of entries in each row.
  int* output_rows = output->mutable_rows();
  int* output_cols = output->mutable_cols();
  double* output_values = output->mutable_values();

  output_rows[0] = 0;
  for (int i = 0; i < index.size(); ++i) {
    const int idx = index[i];
    ++output_rows[rows[idx] + 1];
    output_cols[i] = cols[idx];
    output_values[i] = values[idx];
  }

  // Find the cumulative sum of the row counts.
  for (int i = 1; i < num_rows + 1; ++i) {
    output_rows[i] += output_rows[i - 1];
  }

  CHECK_EQ(output->num_nonzeros(), input.num_nonzeros());
  return output;
}

CompressedRowSparseMatrix::CompressedRowSparseMatrix(const double* diagonal,
                                                     int num_rows) {
  CHECK_NOTNULL(diagonal);

  num_rows_ = num_rows;
  num_cols_ = num_rows;
  storage_type_ = UNSYMMETRIC;
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

CompressedRowSparseMatrix::~CompressedRowSparseMatrix() {}

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

  // The rest of the code updates the block information. Immediately
  // return in case of no block information.
  if (row_blocks_.empty()) {
    return;
  }

  // Sanity check for compressed row sparse block information
  CHECK_EQ(crsb_rows_.size(), row_blocks_.size() + 1);
  CHECK_EQ(crsb_rows_.back(), crsb_cols_.size());

  // Walk the list of row blocks until we reach the new number of rows
  // and the drop the rest of the row blocks.
  int num_row_blocks = 0;
  int num_rows = 0;
  while (num_row_blocks < row_blocks_.size() && num_rows < num_rows_) {
    num_rows += row_blocks_[num_row_blocks];
    ++num_row_blocks;
  }

  row_blocks_.resize(num_row_blocks);

  // Update compressed row sparse block (crsb) information.
  CHECK_EQ(num_rows, num_rows_);
  crsb_rows_.resize(num_row_blocks + 1);
  crsb_cols_.resize(crsb_rows_[num_row_blocks]);
}

void CompressedRowSparseMatrix::AppendRows(const CompressedRowSparseMatrix& m) {
  CHECK_EQ(m.num_cols(), num_cols_);

  CHECK((row_blocks_.empty() && m.row_blocks().empty()) ||
        (!row_blocks_.empty() && !m.row_blocks().empty()))
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
    std::copy(
        m.values(), m.values() + m.num_nonzeros(), &values_[num_nonzeros()]);
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

  // The rest of the code updates the block information. Immediately
  // return in case of no block information.
  if (row_blocks_.empty()) {
    return;
  }

  // Sanity check for compressed row sparse block information
  CHECK_EQ(crsb_rows_.size(), row_blocks_.size() + 1);
  CHECK_EQ(crsb_rows_.back(), crsb_cols_.size());

  row_blocks_.insert(
      row_blocks_.end(), m.row_blocks().begin(), m.row_blocks().end());

  // The rest of the code updates the compressed row sparse block
  // (crsb) information.
  const int num_crsb_nonzeros = crsb_cols_.size();
  const int m_num_crsb_nonzeros = m.crsb_cols_.size();
  crsb_cols_.resize(num_crsb_nonzeros + m_num_crsb_nonzeros);
  std::copy(&m.crsb_cols()[0],
            &m.crsb_cols()[0] + m_num_crsb_nonzeros,
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
      fprintf(file, "% 10d % 10d %17f\n", r, cols_[idx], values_[idx]);
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

CompressedRowSparseMatrix* CompressedRowSparseMatrix::CreateBlockDiagonalMatrix(
    const double* diagonal, const vector<int>& blocks) {
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

  // Fill compressed row sparse block (crsb) information.
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

  switch (storage_type_) {
    case UNSYMMETRIC:
      transpose->set_storage_type(UNSYMMETRIC);
      break;
    case LOWER_TRIANGULAR:
      transpose->set_storage_type(UPPER_TRIANGULAR);
      break;
    case UPPER_TRIANGULAR:
      transpose->set_storage_type(LOWER_TRIANGULAR);
      break;
    default:
      LOG(FATAL) << "Unknown storage type: " << storage_type_;
  };

  TransposeForCompressedRowSparseStructure(num_rows(),
                                           num_cols(),
                                           num_nonzeros(),
                                           rows(),
                                           cols(),
                                           values(),
                                           transpose->mutable_rows(),
                                           transpose->mutable_cols(),
                                           transpose->mutable_values());

  // The rest of the code updates the block information. Immediately
  // return in case of no block information.
  if (row_blocks_.empty()) {
    return transpose;
  }

  // Sanity check for compressed row sparse block information
  CHECK_EQ(crsb_rows_.size(), row_blocks_.size() + 1);
  CHECK_EQ(crsb_rows_.back(), crsb_cols_.size());

  *(transpose->mutable_row_blocks()) = col_blocks_;
  *(transpose->mutable_col_blocks()) = row_blocks_;

  // The rest of the code updates the compressed row sparse block
  // (crsb) information.
  vector<int>& transpose_crsb_rows = *transpose->mutable_crsb_rows();
  vector<int>& transpose_crsb_cols = *transpose->mutable_crsb_cols();

  transpose_crsb_rows.resize(col_blocks_.size() + 1);
  transpose_crsb_cols.resize(crsb_cols_.size());
  TransposeForCompressedRowSparseStructure(row_blocks().size(),
                                           col_blocks().size(),
                                           crsb_cols().size(),
                                           crsb_rows().data(),
                                           crsb_cols().data(),
                                           NULL,
                                           transpose_crsb_rows.data(),
                                           transpose_crsb_cols.data(),
                                           NULL);

  return transpose;
}

namespace {
// A ProductTerm is a term in the block outer product of a matrix with
// itself.
struct ProductTerm {
  ProductTerm(const int row, const int col, const int index)
      : row(row), col(col), index(index) {}

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

// Create outer product matrix based on the block product information.
// The input block product is already sorted. This function does not
// set the sparse rows/cols information. Instead, it only collects the
// nonzeros for each compressed row and puts in row_nnz. The caller of
// this function will traverse the block product in a second round to
// generate the sparse rows/cols information. This function also
// computes the block offset information for the outer product matrix,
// which is used in outer product computation.
CompressedRowSparseMatrix* CreateOuterProductMatrix(
    const int num_cols,
    const CompressedRowSparseMatrix::StorageType storage_type,
    const vector<int>& blocks,
    const vector<ProductTerm>& product,
    vector<int>* row_nnz) {
  // Count the number of unique product term, which in turn is the
  // number of non-zeros in the outer product. Also count the number
  // of non-zeros in each row.
  row_nnz->resize(blocks.size());
  std::fill(row_nnz->begin(), row_nnz->end(), 0);
  (*row_nnz)[product[0].row] = blocks[product[0].col];
  int num_nonzeros = blocks[product[0].row] * blocks[product[0].col];
  for (int i = 1; i < product.size(); ++i) {
    // Each (row, col) block counts only once.
    // This check depends on product sorted on (row, col).
    if (product[i].row != product[i - 1].row ||
        product[i].col != product[i - 1].col) {
      (*row_nnz)[product[i].row] += blocks[product[i].col];
      num_nonzeros += blocks[product[i].row] * blocks[product[i].col];
    }
  }

  CompressedRowSparseMatrix* matrix =
      new CompressedRowSparseMatrix(num_cols, num_cols, num_nonzeros);
  matrix->set_storage_type(storage_type);

  *(matrix->mutable_row_blocks()) = blocks;
  *(matrix->mutable_col_blocks()) = blocks;

  // Compute block offsets for outer product matrix, which is used in
  // ComputeOuterProduct.
  vector<int>* block_offsets = matrix->mutable_block_offsets();
  block_offsets->resize(blocks.size() + 1);
  (*block_offsets)[0] = 0;
  for (int i = 0; i < blocks.size(); ++i) {
    (*block_offsets)[i + 1] = (*block_offsets)[i] + blocks[i];
  }

  return matrix;
}

CompressedRowSparseMatrix* CompressAndFillProgram(
    const int num_cols,
    const CompressedRowSparseMatrix::StorageType storage_type,
    const vector<int>& blocks,
    const vector<ProductTerm>& product,
    vector<int>* program) {
  CHECK_GT(product.size(), 0);

  vector<int> row_nnz;
  CompressedRowSparseMatrix* matrix = CreateOuterProductMatrix(
      num_cols, storage_type, blocks, product, &row_nnz);

  const vector<int>& block_offsets = matrix->block_offsets();

  int* crsm_rows = matrix->mutable_rows();
  std::fill(crsm_rows, crsm_rows + num_cols + 1, 0);
  int* crsm_cols = matrix->mutable_cols();
  std::fill(crsm_cols, crsm_cols + matrix->num_nonzeros(), 0);

  CHECK_NOTNULL(program)->clear();
  program->resize(product.size());

  // Non zero elements are not stored consecutively across rows in a block.
  // We seperate nonzero into three categories:
  //   nonzeros in all previous row blocks counted in nnz
  //   nonzeros in current row counted in row_nnz
  //   nonzeros in previous col blocks of current row counted in col_nnz
  //
  // Give an element (j, k) within a block such that j and k
  // represent the relative position to the starting row and starting col of
  // the block, the row and col for the element is
  //   block_offsets[current.row] + j
  //   block_offsets[current.col] + k
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

  // Process first product term.
  for (int j = 0; j < blocks[product[0].row]; ++j) {
    crsm_rows[block_offsets[product[0].row] + j + 1] = row_nnz[product[0].row];
    for (int k = 0; k < blocks[product[0].col]; ++k) {
      crsm_cols[row_nnz[product[0].row] * j + k] =
          block_offsets[product[0].col] + k;
    }
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
        nnz += col_nnz * blocks[previous.row];
        col_nnz = 0;

        for (int j = 0; j < blocks[current.row]; ++j) {
          crsm_rows[block_offsets[current.row] + j + 1] = row_nnz[current.row];
        }
      }

      for (int j = 0; j < blocks[current.row]; ++j) {
        for (int k = 0; k < blocks[current.col]; ++k) {
          crsm_cols[nnz + row_nnz[current.row] * j + col_nnz + k] =
              block_offsets[current.col] + k;
        }
      }
    }

    (*program)[current.index] = col_nnz;
  }

  for (int i = 1; i < num_cols + 1; ++i) {
    crsm_rows[i] += crsm_rows[i - 1];
  }

  return matrix;
}

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

CompressedRowSparseMatrix*
CompressedRowSparseMatrix::CreateOuterProductMatrixAndProgram(
    const CompressedRowSparseMatrix& m,
    const CompressedRowSparseMatrix::StorageType storage_type,
    vector<int>* program) {
  CHECK(storage_type == LOWER_TRIANGULAR || storage_type == UPPER_TRIANGULAR);
  CHECK_NOTNULL(program)->clear();
  CHECK_GT(m.num_nonzeros(), 0)
      << "Congratulations, you found a bug in Ceres. Please report it.";

  vector<ProductTerm> product;
  const vector<int>& col_blocks = m.col_blocks();
  const vector<int>& crsb_rows = m.crsb_rows();
  const vector<int>& crsb_cols = m.crsb_cols();

  // Give input matrix m in Compressed Row Sparse Block format
  //     (row_block, col_block)
  // represent each block multiplication
  //     (row_block, col_block1)' X (row_block, col_block2)
  // by its product term index and sort the product terms
  //     (col_block1, col_block2, index)
  //
  // Due to the compression on rows, col_block is accessed through idx to
  // crsb_cols.  So col_block is accessed as crsb_cols[idx] in the code.
  for (int row_block = 1; row_block < crsb_rows.size(); ++row_block) {
    for (int idx1 = crsb_rows[row_block - 1]; idx1 < crsb_rows[row_block];
         ++idx1) {
      if (storage_type == LOWER_TRIANGULAR) {
        for (int idx2 = crsb_rows[row_block - 1]; idx2 <= idx1; ++idx2) {
          product.push_back(
              ProductTerm(crsb_cols[idx1], crsb_cols[idx2], product.size()));
        }
      } else {  // Upper triangular matrix.
        for (int idx2 = idx1; idx2 < crsb_rows[row_block]; ++idx2) {
          product.push_back(
              ProductTerm(crsb_cols[idx1], crsb_cols[idx2], product.size()));
        }
      }
    }
  }

  sort(product.begin(), product.end());
  return CompressAndFillProgram(
      m.num_cols(), storage_type, col_blocks, product, program);
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
// idx to crsb_col vector. So col_block is accessed as crsb_col[idx]
// in the code.
//
// Note this function produces a triangular matrix in block unit (i.e.
// diagonal block is a normal block) instead of standard triangular matrix.
// So there is no special handling for diagonal blocks.
void CompressedRowSparseMatrix::ComputeOuterProduct(
    const CompressedRowSparseMatrix& m,
    const vector<int>& program,
    CompressedRowSparseMatrix* result) {
  CHECK(result->storage_type() == LOWER_TRIANGULAR ||
        result->storage_type() == UPPER_TRIANGULAR);
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
  const StorageType storage_type = result->storage_type();
#define COL_BLOCK1 (crsb_cols[idx1])
#define COL_BLOCK2 (crsb_cols[idx2])

  // Iterate row blocks.
  for (int row_block = 0, m_row_begin = 0; row_block < row_blocks.size();
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
      const int row_begin = block_offsets[COL_BLOCK1];
      const int row_nnz = rows[row_begin + 1] - rows[row_begin];
      if (storage_type == LOWER_TRIANGULAR) {
        for (int idx2 = crsb_rows[row_block], m_col_nnz2 = 0; idx2 <= idx1;
             m_col_nnz2 += col_blocks[COL_BLOCK2], ++idx2, ++cursor) {
          int col_nnz = program[cursor];
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
          int col_nnz = program[cursor];
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

  CHECK_EQ(cursor, program.size());
}

CompressedRowSparseMatrix* CompressedRowSparseMatrix::CreateRandomMatrix(
    const CompressedRowSparseMatrix::RandomMatrixOptions& options) {
  CHECK_GT(options.num_row_blocks, 0);
  CHECK_GT(options.min_row_block_size, 0);
  CHECK_GT(options.max_row_block_size, 0);
  CHECK_LE(options.min_row_block_size, options.max_row_block_size);
  CHECK_GT(options.num_col_blocks, 0);
  CHECK_GT(options.min_col_block_size, 0);
  CHECK_GT(options.max_col_block_size, 0);
  CHECK_LE(options.min_col_block_size, options.max_col_block_size);
  CHECK_GT(options.block_density, 0.0);
  CHECK_LE(options.block_density, 1.0);

  vector<int> row_blocks;
  vector<int> col_blocks;

  // Generate the row block structure.
  for (int i = 0; i < options.num_row_blocks; ++i) {
    // Generate a random integer in [min_row_block_size, max_row_block_size]
    const int delta_block_size =
        Uniform(options.max_row_block_size - options.min_row_block_size);
    row_blocks.push_back(options.min_row_block_size + delta_block_size);
  }

  // Generate the col block structure.
  for (int i = 0; i < options.num_col_blocks; ++i) {
    // Generate a random integer in [min_col_block_size, max_col_block_size]
    const int delta_block_size =
        Uniform(options.max_col_block_size - options.min_col_block_size);
    col_blocks.push_back(options.min_col_block_size + delta_block_size);
  }

  vector<int> crsb_rows;
  vector<int> crsb_cols;
  vector<int> tsm_rows;
  vector<int> tsm_cols;
  vector<double> tsm_values;

  // For ease of construction, we are going to generate the
  // CompressedRowSparseMatrix by generating it as a
  // TripletSparseMatrix and then converting it to a
  // CompressedRowSparseMatrix.

  // It is possible that the random matrix is empty which is likely
  // not what the user wants, so do the matrix generation till we have
  // at least one non-zero entry.
  while (tsm_values.empty()) {
    crsb_rows.clear();
    crsb_cols.clear();
    tsm_rows.clear();
    tsm_cols.clear();
    tsm_values.clear();

    int row_block_begin = 0;
    for (int r = 0; r < options.num_row_blocks; ++r) {
      int col_block_begin = 0;
      crsb_rows.push_back(crsb_cols.size());
      for (int c = 0; c < options.num_col_blocks; ++c) {
        // Randomly determine if this block is present or not.
        if (RandDouble() <= options.block_density) {
          AddRandomBlock(row_blocks[r],
                         col_blocks[c],
                         row_block_begin,
                         col_block_begin,
                         &tsm_rows,
                         &tsm_cols,
                         &tsm_values);
          // Add the block to the block sparse structure.
          crsb_cols.push_back(c);
        }
        col_block_begin += col_blocks[c];
      }
      row_block_begin += row_blocks[r];
    }
    crsb_rows.push_back(crsb_cols.size());
  }

  const int num_rows = std::accumulate(row_blocks.begin(), row_blocks.end(), 0);
  const int num_cols = std::accumulate(col_blocks.begin(), col_blocks.end(), 0);
  const bool kDoNotTranspose = false;
  CompressedRowSparseMatrix* matrix =
      CompressedRowSparseMatrix::FromTripletSparseMatrix(
          TripletSparseMatrix(
              num_rows, num_cols, tsm_rows, tsm_cols, tsm_values),
          kDoNotTranspose);
  (*matrix->mutable_row_blocks()) = row_blocks;
  (*matrix->mutable_col_blocks()) = col_blocks;
  (*matrix->mutable_crsb_rows()) = crsb_rows;
  (*matrix->mutable_crsb_cols()) = crsb_cols;
  matrix->set_storage_type(CompressedRowSparseMatrix::UNSYMMETRIC);
  return matrix;
}

}  // namespace internal
}  // namespace ceres
