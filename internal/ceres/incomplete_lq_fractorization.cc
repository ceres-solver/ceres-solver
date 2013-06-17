#include <vector>
#include <utility>
#include <cmath>
#include "ceres/crs_matrix.h"
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
  const int* b_cols = b_cols;
  const double* b_values = b.values();


  const int row_a_end = a_rows[row_a + 1];
  const int row_b_end = b_rows[row_b + 1];

  int idx_a = a_rows[row_a];
  int idx_b = b_rows[row_b];
  double dot_product = 0.0;
  while ((idx_a < row_a_end) && (idx_b < row_b_end)) {
    if (a_cols[idx_a] == b_cols[idx_b]) {
      dot_product += a.values()[idx_a] * b.values()[idx_b];
      ++idx_a;
      ++idx_b;
    }

    while (a_cols[idx_a] < b_cols[idx_b] && (idx_a < row_a_end)) ++idx_a;
    while (a_cols[idx_a] > b_cols[idx_b] && (idx_b < row_b_end)) ++idx_b;
  }
  return dot_product;
}

void DropAndCompress(const Vector& dense_row,
                     const int num_cols,
                     const int level_of_fill,
                     const double drop_tolerance,
                     vector<pair<int, double> >* scratch,
                     CompressedRowSparseMatrix* sparse_row) {
  const double max_value = dense_row.cwiseAbs().maxCoeff();
  const double threshold = drop_tolerance * max_value;
  SparseMatrixStorage* storage = sparse_row->storage;

  storage->cols.resize(0);
  storage->values.resize(0);
  scratch->resize(num_cols);

  int scratch_count = 0;
  for (int i = 0; i < num_cols; ++i) {
    if (fabs(dense_row[i]) > threshold) {
      pair<int,double>& entry = (*scratch)[scratch_count];
      entry.first = i;
      entry.second = dense_row[i];
      ++scratch_count;
    }
  }

  if (storage->cols.size() > level_of_fill) {
    sort(scratch->begin(), scratch->begin() + scratch_count, SecondGreaterThan());
    scratch->resize(level_of_fill);
    scratch_count = level_of_fill;
  }

  sort(scratch->begin(), scratch->begin() + scratch_count);
  for (int i = 0; i < scratch_count; ++i) {
    const pair<int,double>& entry = (*scratch)[i];
    storage->cols.push_back(entry.first);
    storage->values.push_back(entry.second);
  }

  storage->num_rows = 1;
  storage->rows[0] = 0;
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

  CompressedRowSparseMatrix qi(1, num_cols, num_cols);
  CompressedRowSparseMatrix li(1, num_rows, num_rows);

  Vector li_dense(num_rows);
  Vector qi_dense(num_cols);
  vector<pair<int, double> > scratch(num_cols);

  qi_dense.setZero();
  for (int idx = matrix.rows[0]; idx < matrix.rows[1]; ++idx) {
    qi_dense(matrix.cols[idx]) = matrix.values[idx];
  }

  DropAndCompress(qi_dense,
                  num_cols,
                  q_level_of_fill,
                  q_drop_tolerance,
                  &scratch,
                  &qi);

  li.cols()[0] = 0;
  li.values()[0] = NormalizeRow(0, &qi);

  q.AppendRows(qi);
  l->AppendRows(li);

  for (int i = 1; i < num_rows; ++i) {
    li_dense.setZero();

    // Dot product
    for (int j = 0; j < i; ++j) {
      li_dense(j) = RowDotProduct(a, i, q, j);
    }

    DropAndCompress(li_dense, i, l_level_of_fill, l_drop_tolerance, &scratch, &li);

    qi_dense.setZero();
    for (int idx = matrix.rows[i]; idx < matrix.rows[i + 1]; ++idx) {
      qi_dense(matrix.cols[idx]) = matrix.values[idx];
    }

    for (int j = 0; j < li.cols.size(); ++j) {
      const int r = li.cols[j];
      const double lij = li.values[j];

      for (int idx = matrix.rows[r]; idx < matrix.rows[r + 1]; ++idx) {
        qi_dense(matrix.cols[idx]) -= lij * q->values[idx];
      }
    }

    // Update qi;
    DropAndCompress(qi_dense, num_cols, q_level_of_fill, q_drop_tolerance, &scratch, &qi);

    li.cols.push_back(i);
    li.values.push_back(NormalizeRow(0, &qi));
    q.AppendRows(qi);
    li->AppendRows(li);
  };

  return l;
}


}  // namespace internal
}  // namespace ceres
