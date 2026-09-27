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

#include "ceres/mkl_sparse_matrix.h"

#ifndef CERES_NO_MKL

#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/mkl_diagnostics.h"
#include "mkl.h"

namespace ceres::internal {

void MklSparseHandleDeleter::operator()(const pointer handle) const noexcept {
  const sparse_status_t status = mkl_sparse_destroy(handle);
  if (status != SPARSE_STATUS_SUCCESS) {
    LOG(ERROR) << "MKL sparse handle destruction returned "
               << MklStatusToString(status) << " (" << static_cast<int>(status)
               << "), expected " << MklStatusToString(SPARSE_STATUS_SUCCESS)
               << " (" << static_cast<int>(SPARSE_STATUS_SUCCESS) << ").";
  }
}

bool MklCsrMatrix::Create(const CompressedRowSparseMatrix& matrix,
                          std::string* message) {
  const int num_rows = matrix.num_rows();
  const int num_nonzeros = matrix.num_nonzeros();
  return Create(
      num_rows,
      matrix.num_cols(),
      std::vector<MKL_INT>(matrix.rows(), matrix.rows() + num_rows + 1),
      std::vector<MKL_INT>(matrix.cols(), matrix.cols() + num_nonzeros),
      std::vector<double>(matrix.values(), matrix.values() + num_nonzeros),
      message);
}

bool MklCsrMatrix::Create(const int num_rows,
                          const int num_cols,
                          std::vector<MKL_INT> rows,
                          std::vector<MKL_INT> columns,
                          std::vector<double> values,
                          std::string* message) {
  handle_.reset();
  rows_ = std::move(rows);
  columns_ = std::move(columns);
  values_ = std::move(values);

  sparse_matrix_t raw_handle = nullptr;
  const sparse_status_t status =
      mkl_sparse_d_create_csr(&raw_handle,
                              SPARSE_INDEX_BASE_ZERO,
                              static_cast<MKL_INT>(num_rows),
                              static_cast<MKL_INT>(num_cols),
                              rows_.data(),
                              rows_.data() + 1,
                              columns_.data(),
                              values_.data());
  MklSparseHandle handle(raw_handle);
  if (!CheckMklStatus(status, "sparse matrix creation", message)) {
    return false;
  }
  // The handle refers to the arrays owned by this object, so sorting them
  // leaves the caller's data untouched.
  if (!CheckMklStatus(
          mkl_sparse_order(handle.get()), "sparse matrix ordering", message)) {
    return false;
  }
  handle_ = std::move(handle);
  return true;
}

}  // namespace ceres::internal

#endif  // CERES_NO_MKL
