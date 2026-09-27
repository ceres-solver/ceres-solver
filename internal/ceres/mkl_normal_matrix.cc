// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#include "ceres/mkl_normal_matrix.h"

#ifndef CERES_NO_MKL

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/mkl_diagnostics.h"
#include "ceres/mkl_sparse_matrix.h"
#include "ceres/mkl_utils.h"
#include "mkl.h"

namespace ceres::internal {

bool ComputeAtAUsingMkl(const sparse_matrix_t matrix,
                        MklSparseHandle* result,
                        std::string* message) {
  sparse_matrix_t raw_product = nullptr;
  if (!CheckMklStatus(
          mkl_sparse_syrk(SPARSE_OPERATION_TRANSPOSE, matrix, &raw_product),
          "sparse symmetric product",
          message)) {
    return false;
  }
  MklSparseHandle product(raw_product);
  if (!CheckMklStatus(
          mkl_sparse_order(product.get()), "sparse matrix ordering", message)) {
    return false;
  }
  *result = std::move(product);
  return true;
}

bool ComputeAtAUsingMkl(const CompressedRowSparseMatrix& matrix,
                        const int max_num_threads,
                        std::unique_ptr<CompressedRowSparseMatrix>* result,
                        std::string* message) {
  const MklThreadScope thread_scope(max_num_threads);
  MklCsrMatrix operand;
  MklSparseHandle product;
  if (!operand.Create(matrix, message) ||
      !ComputeAtAUsingMkl(operand.get(), &product, message)) {
    return false;
  }
  return ExportMklCsr(product.get(), result, message);
}

bool ExportMklCsr(const sparse_matrix_t handle,
                  std::unique_ptr<CompressedRowSparseMatrix>* result,
                  std::string* message) {
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT num_rows = 0;
  MKL_INT num_cols = 0;
  MKL_INT* row_start = nullptr;
  MKL_INT* row_end = nullptr;
  MKL_INT* columns = nullptr;
  double* values = nullptr;
  if (!CheckMklStatus(mkl_sparse_d_export_csr(handle,
                                              &indexing,
                                              &num_rows,
                                              &num_cols,
                                              &row_start,
                                              &row_end,
                                              &columns,
                                              &values),
                      "sparse matrix export",
                      message)) {
    return false;
  }

  if (indexing != SPARSE_INDEX_BASE_ZERO ||
      num_rows > std::numeric_limits<int>::max() ||
      num_cols > std::numeric_limits<int>::max()) {
    *message = absl::StrFormat(
        "MKL sparse matrix export returned dimensions %d x %d with index base "
        "%d, expected dimensions representable by int and index base 0.",
        num_rows,
        num_cols,
        static_cast<int>(indexing));
    return false;
  }

  // Ceres requires compact row bounds starting at zero, unlike MKL's
  // four-array CSR format.
  if (num_rows > 0 && row_start[0] != 0) {
    *message = absl::StrFormat(
        "MKL sparse matrix export returned a four-array CSR structure whose "
        "first row starts at %d, expected 0.",
        row_start[0]);
    return false;
  }
  for (MKL_INT row = 0; row + 1 < num_rows; ++row) {
    if (row_end[row] != row_start[row + 1]) {
      *message = absl::StrFormat(
          "MKL sparse matrix export returned a non-compact four-array CSR "
          "structure: row %d ends at %d, but row %d starts at %d, expected "
          "them to be equal.",
          row,
          row_end[row],
          row + 1,
          row_start[row + 1]);
      return false;
    }
  }

  const MKL_INT num_nonzeros = num_rows == 0 ? 0 : row_end[num_rows - 1];
  if (num_nonzeros > std::numeric_limits<int>::max()) {
    *message = absl::StrFormat(
        "MKL sparse matrix export returned %d nonzeros, expected at most %d.",
        num_nonzeros,
        std::numeric_limits<int>::max());
    return false;
  }

  *result = std::make_unique<CompressedRowSparseMatrix>(
      static_cast<int>(num_rows),
      static_cast<int>(num_cols),
      static_cast<int>(num_nonzeros));
  CompressedRowSparseMatrix& exported = **result;
  std::copy_n(row_start, num_rows, exported.mutable_rows());
  exported.mutable_rows()[num_rows] = static_cast<int>(num_nonzeros);
  std::copy_n(columns, num_nonzeros, exported.mutable_cols());
  std::copy_n(values, num_nonzeros, exported.mutable_values());
  exported.set_storage_type(
      CompressedRowSparseMatrix::StorageType::UPPER_TRIANGULAR);
  return true;
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
