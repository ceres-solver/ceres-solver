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

#ifndef CERES_INTERNAL_MKL_SPARSE_MATRIX_H_
#define CERES_INTERNAL_MKL_SPARSE_MATRIX_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_MKL

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/export.h"
#include "mkl.h"

namespace ceres::internal {

// Describes a general matrix whose stored entries oneMKL uses as they are.
inline constexpr matrix_descr kGeneralMatrixDescriptor{
    SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

struct CERES_NO_EXPORT MklSparseHandleDeleter {
  using pointer = sparse_matrix_t;

  void operator()(pointer handle) const noexcept;
};

using MklSparseHandle = std::unique_ptr<std::remove_pointer_t<sparse_matrix_t>,
                                        MklSparseHandleDeleter>;

// Owns an MKL CSR view of a Ceres matrix. Ceres does not guarantee that column
// indices are sorted within a row, which some oneMKL sparse routines require,
// so the converted copy is sorted.
class CERES_NO_EXPORT MklCsrMatrix final {
 public:
  MklCsrMatrix() = default;
  MklCsrMatrix(const MklCsrMatrix&) = delete;
  MklCsrMatrix& operator=(const MklCsrMatrix&) = delete;

  // Convert both the structure and the values of matrix.
  bool Create(const CompressedRowSparseMatrix& matrix, std::string* message);

  // Take ownership of a matrix already in the representation of oneMKL. rows
  // holds num_rows + 1 row offsets.
  bool Create(int num_rows,
              int num_cols,
              std::vector<MKL_INT> rows,
              std::vector<MKL_INT> columns,
              std::vector<double> values,
              std::string* message);

  sparse_matrix_t get() const noexcept { return handle_.get(); }
  int num_rows() const noexcept {
    return rows_.empty() ? 0 : static_cast<int>(rows_.size()) - 1;
  }
  // Row offsets and column indices, sorted within each row.
  const MKL_INT* rows() const noexcept { return rows_.data(); }
  const MKL_INT* columns() const noexcept { return columns_.data(); }
  double* mutable_values() noexcept { return values_.data(); }

 private:
  MklSparseHandle handle_;
  std::vector<MKL_INT> rows_;
  std::vector<MKL_INT> columns_;
  std::vector<double> values_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_MKL

#endif  // CERES_INTERNAL_MKL_SPARSE_MATRIX_H_
