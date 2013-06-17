// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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


#include "ceres/incomplete_lq_factorization.h"

#include <vector>
#include <utility>
#include <cmath>
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/port.h"
#include "ceres/internal/eigen.h"
#include "glog/logging.h"
#include <iostream>

namespace ceres {
namespace internal {

double NormalizeRow(const int row, CompressedRowSparseMatrix* matrix) {
  double norm = 0.0;
  const int* rows = matrix->rows();
  double* values = matrix->mutable_values();

  for (int idx =  rows[row]; idx < rows[row + 1]; ++idx) {
    norm += values[idx] * values[idx];
  }

  norm = sqrt(norm);
  const double inverse_norm = 1.0 / norm;
  for (int idx =  rows[row]; idx < rows[row + 1]; ++idx) {
    values[idx] *= inverse_norm;
  }

  return norm;
}

double RowDotProduct(const CompressedRowSparseMatrix& a,
                     const int row_a,
                     const CompressedRowSparseMatrix& b,
                     const int row_b) {
  const int* a_rows = a.rows();
  const int* a_cols = a.cols();
  const double* a_values = a.values();

  const int* b_rows = b.rows();
  const int* b_cols = b.cols();
  const double* b_values = b.values();

  const int row_a_end = a_rows[row_a + 1];
  const int row_b_end = b_rows[row_b + 1];

  int idx_a = a_rows[row_a];
  int idx_b = b_rows[row_b];
  double dot_product = 0.0;
  while (idx_a < row_a_end && idx_b < row_b_end) {
    if (a_cols[idx_a] == b_cols[idx_b]) {
      dot_product += a_values[idx_a++] * b_values[idx_b++];
    }

    while (a_cols[idx_a] < b_cols[idx_b] && idx_a < row_a_end) {
      ++idx_a;
    }

    while (a_cols[idx_a] > b_cols[idx_b] && idx_b < row_b_end) {
      ++idx_b;
    }
  }

  return dot_product;
}

struct SecondGreaterThan {
 public:
  bool operator()(const pair<int, double>& lhs,
                  const pair<int, double>& rhs) const {
    return (fabs(lhs.second) > fabs(rhs.second));
  }
};

void DropAndCompress(const Vector& dense_row,
                     const int num_cols,
                     const int level_of_fill,
                     const double drop_tolerance,
                     vector<pair<int, double> >* scratch,
                     CompressedRowSparseMatrix* sparse_row) {
  SparseMatrixStorage* storage = sparse_row->mutable_storage();
  storage->rows[1] = 0;

  if (num_cols == 0) {
    return;
  }

  const double max_value = dense_row.head(num_cols).cwiseAbs().maxCoeff();
  const double threshold = drop_tolerance * max_value;

  int scratch_count = 0;
  for (int i = 0; i < num_cols; ++i) {
    if (fabs(dense_row[i]) > threshold) {
      pair<int,double>& entry = (*scratch)[scratch_count];
      entry.first = i;
      entry.second = dense_row[i];
      ++scratch_count;
    }
  }

  if (scratch_count > level_of_fill) {
    nth_element(scratch->begin(),
                scratch->begin() + level_of_fill,
                scratch->begin() + scratch_count,
                SecondGreaterThan());
    scratch_count = level_of_fill;
    sort(scratch->begin(), scratch->begin() + scratch_count);
  }

  for (int i = 0; i < scratch_count; ++i) {
    const pair<int,double>& entry = (*scratch)[i];
    storage->cols[i] = entry.first;
    storage->values[i] = entry.second;
  }

  storage->rows[1] = scratch_count;
}

CompressedRowSparseMatrix* IncompleteLQFactorization(
    const CompressedRowSparseMatrix& matrix,
    const int l_level_of_fill,
    const double l_drop_tolerance,
    const int q_level_of_fill,
    const double q_drop_tolerance) {
  const int num_rows = matrix.num_rows();
  const int num_cols = matrix.num_cols();

  CompressedRowSparseMatrix* l =
      new CompressedRowSparseMatrix(0, num_rows, l_level_of_fill * num_rows);

  CompressedRowSparseMatrix q(0, num_cols, q_level_of_fill * num_rows);

  CompressedRowSparseMatrix li(1, num_rows, num_rows);
  Vector li_dense(num_rows);
  CompressedRowSparseMatrix qi(1, num_cols, num_cols);
  Vector qi_dense(num_cols);

  vector<pair<int, double> > scratch(num_cols);
  for (int i = 0; i < num_rows; ++i) {
    li_dense.setZero();
    for (int j = 0; j < i; ++j) {
      li_dense(j) = RowDotProduct(matrix, i, q, j);
    }
    DropAndCompress(li_dense, i, l_level_of_fill, l_drop_tolerance, &scratch, &li);

    qi_dense.setZero();
    for (int idx = matrix.rows()[i]; idx < matrix.rows()[i + 1]; ++idx) {
      qi_dense(matrix.cols()[idx]) = matrix.values()[idx];
    }

    for (int j = 0; j < li.num_nonzeros(); ++j) {
      const int r = li.cols()[j];
      const double lij = li.values()[j];
      for (int idx = matrix.rows()[r]; idx < matrix.rows()[r + 1]; ++idx) {
        qi_dense(matrix.cols()[idx]) -= lij * q.values()[idx];
      }
    }
    DropAndCompress(qi_dense, num_cols, q_level_of_fill, q_drop_tolerance, &scratch, &qi);

    li.mutable_cols()[li.num_nonzeros()] = i;
    li.mutable_values()[li.num_nonzeros()] = NormalizeRow(0, &qi);
    ++li.mutable_rows()[1];
    l->AppendRows(li);
    q.AppendRows(qi);
  };

  return l;
}


}  // namespace internal
}  // namespace ceres
