// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
#include <vector>
#include "ceres/crs_matrix.h"
#include "ceres/internal/port.h"
#include "ceres/sparse_matrix_storage.h"
#include "ceres/triplet_sparse_matrix.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {
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
  storage_.num_rows = num_rows;
  storage_.num_cols = num_cols;
  storage_.rows.resize(num_rows + 1, 0);
  storage_.cols.resize(max_num_nonzeros, 0);
  storage_.values.resize(max_num_nonzeros, 0.0);


  VLOG(1) << "# of rows: " << storage_.num_rows
          << " # of columns: " << storage_.num_cols
          << " max_num_nonzeros: " << storage_.cols.size()
          << ". Allocating " << (storage_.num_rows + 1) * sizeof(int) +  // NOLINT
      storage_.cols.size() * sizeof(int) +  // NOLINT
      storage_.cols.size() * sizeof(double);  // NOLINT
}

CompressedRowSparseMatrix::CompressedRowSparseMatrix(
    const TripletSparseMatrix& m) {
  storage_.num_rows = m.num_rows();
  storage_.num_cols = m.num_cols();

  storage_.rows.resize(storage_.num_rows + 1, 0);
  storage_.cols.resize(m.num_nonzeros(), 0);
  storage_.values.resize(m.max_num_nonzeros(), 0.0);

  // index is the list of indices into the TripletSparseMatrix m.
  vector<int> index(m.num_nonzeros(), 0);
  for (int i = 0; i < m.num_nonzeros(); ++i) {
    index[i] = i;
  }

  // Sort index such that the entries of m are ordered by row and ties
  // are broken by column.
  sort(index.begin(), index.end(), RowColLessThan(m.rows(), m.cols()));

  VLOG(1) << "# of rows: " << storage_.num_rows
          << " # of columns: " << storage_.num_cols
          << " max_num_nonzeros: " << storage_.cols.size()
          << ". Allocating "
          << ((storage_.num_rows + 1) * sizeof(int) +  // NOLINT
              storage_.cols.size() * sizeof(int) +     // NOLINT
              storage_.cols.size() * sizeof(double));  // NOLINT

  // Copy the contents of the cols and values array in the order given
  // by index and count the number of entries in each row.
  for (int i = 0; i < m.num_nonzeros(); ++i) {
    const int idx = index[i];
    ++storage_.rows[m.rows()[idx] + 1];
    storage_.cols[i] = m.cols()[idx];
    storage_.values[i] = m.values()[idx];
  }

  // Find the cumulative sum of the row counts.
  for (int i = 1; i < storage_.num_rows + 1; ++i) {
    storage_.rows[i] += storage_.rows[i-1];
  }

  CHECK_EQ(num_nonzeros(), m.num_nonzeros());
}

CompressedRowSparseMatrix::CompressedRowSparseMatrix(const double* diagonal,
                                                     int num_rows) {
  CHECK_NOTNULL(diagonal);

  storage_.num_rows = num_rows;
  storage_.num_cols = num_rows;
  storage_.rows.resize(num_rows + 1);
  storage_.cols.resize(num_rows);
  storage_.values.resize(num_rows);

  storage_.rows[0] = 0;
  for (int i = 0; i < storage_.num_rows; ++i) {
    storage_.cols[i] = i;
    storage_.values[i] = diagonal[i];
    storage_.rows[i + 1] = i + 1;
  }

  CHECK_EQ(num_nonzeros(), num_rows);
}

CompressedRowSparseMatrix::~CompressedRowSparseMatrix() {
}

void CompressedRowSparseMatrix::SetZero() {
  fill(storage_.values.begin(), storage_.values.end(), 0);
}

void CompressedRowSparseMatrix::RightMultiply(const double* x,
                                              double* y) const {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(y);

  for (int r = 0; r < storage_.num_rows; ++r) {
    for (int idx = storage_.rows[r]; idx < storage_.rows[r + 1]; ++idx) {
      y[r] += storage_.values[idx] * x[storage_.cols[idx]];
    }
  }
}

void CompressedRowSparseMatrix::LeftMultiply(const double* x, double* y) const {
  CHECK_NOTNULL(x);
  CHECK_NOTNULL(y);

  for (int r = 0; r < storage_.num_rows; ++r) {
    for (int idx = storage_.rows[r]; idx < storage_.rows[r + 1]; ++idx) {
      y[storage_.cols[idx]] += storage_.values[idx] * x[r];
    }
  }
}

void CompressedRowSparseMatrix::SquaredColumnNorm(double* x) const {
  CHECK_NOTNULL(x);

  fill(x, x + storage_.num_cols, 0.0);
  for (int idx = 0; idx < storage_.rows[storage_.num_rows]; ++idx) {
    x[storage_.cols[idx]] += storage_.values[idx] * storage_.values[idx];
  }
}

void CompressedRowSparseMatrix::ScaleColumns(const double* scale) {
  CHECK_NOTNULL(scale);

  for (int idx = 0; idx < storage_.rows[storage_.num_rows]; ++idx) {
    storage_.values[idx] *= scale[storage_.cols[idx]];
  }
}

void CompressedRowSparseMatrix::ToDenseMatrix(Matrix* dense_matrix) const {
  CHECK_NOTNULL(dense_matrix);
  dense_matrix->resize(storage_.num_rows, storage_.num_cols);
  dense_matrix->setZero();

  for (int r = 0; r < storage_.num_rows; ++r) {
    for (int idx = storage_.rows[r]; idx < storage_.rows[r + 1]; ++idx) {
      (*dense_matrix)(r, storage_.cols[idx]) = storage_.values[idx];
    }
  }
}

void CompressedRowSparseMatrix::DeleteRows(int delta_rows) {
  CHECK_GE(delta_rows, 0);
  CHECK_LE(delta_rows, storage_.num_rows);

  int new_num_rows = storage_.num_rows - delta_rows;

  storage_.num_rows = new_num_rows;
  storage_.rows.resize(storage_.num_rows + 1);
}

void CompressedRowSparseMatrix::AppendRows(const CompressedRowSparseMatrix& m) {
  CHECK_EQ(m.num_cols(), storage_.num_cols);

  if (storage_.cols.size() < num_nonzeros() + m.num_nonzeros()) {
    storage_.cols.resize(num_nonzeros() + m.num_nonzeros());
    storage_.values.resize(num_nonzeros() + m.num_nonzeros());
  }

  // Copy the contents of m into this matrix.
  copy(m.cols(), m.cols() + m.num_nonzeros(), &storage_.cols[num_nonzeros()]);
  copy(m.values(),
       m.values() + m.num_nonzeros(),
       &storage_.values[num_nonzeros()]);

  storage_.rows.resize(storage_.num_rows + m.num_rows() + 1);
  // new_rows = [storage_.rows, m.row() + storage_.rows[storage_.num_rows]]
  fill(storage_.rows.begin() +  storage_.num_rows,
       storage_.rows.begin() + storage_.num_rows + m.num_rows() + 1,
       storage_.rows[storage_.num_rows]);

  for (int r = 0; r < m.num_rows() + 1; ++r) {
    storage_.rows[storage_.num_rows + r] += m.rows()[r];
  }

  storage_.num_rows += m.num_rows();
}

void CompressedRowSparseMatrix::ToTextFile(FILE* file) const {
  CHECK_NOTNULL(file);
  for (int r = 0; r < storage_.num_rows; ++r) {
    for (int idx = storage_.rows[r]; idx < storage_.rows[r + 1]; ++idx) {
      fprintf(file,
              "% 10d % 10d %17f\n",
              r,
              storage_.cols[idx],
              storage_.values[idx]);
    }
  }
}

void CompressedRowSparseMatrix::ToCRSMatrix(CRSMatrix* matrix) const {
  matrix->num_rows = storage_.num_rows;
  matrix->num_cols = storage_.num_cols;
  matrix->rows = storage_.rows;
  matrix->cols = storage_.cols;
  matrix->values = storage_.values;

  // Trim.
  matrix->rows.resize(matrix->num_rows + 1);
  matrix->cols.resize(matrix->rows[matrix->num_rows]);
  matrix->values.resize(matrix->rows[matrix->num_rows]);
}

}  // namespace internal
}  // namespace ceres
