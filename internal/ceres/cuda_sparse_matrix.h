// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: joydeepb@cs.utexas.edu (Joydeep Biswas)
//
// A CUDA sparse matrix linear operator.

#ifndef CERES_INTERNAL_CUDA_SPARSE_MATRIX_H_
#define CERES_INTERNAL_CUDA_SPARSE_MATRIX_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <cstdint>
#include <memory>
#include <string>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/context_impl.h"
#include "ceres/internal/export.h"
#include "ceres/types.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_buffer.h"
#include "ceres/cuda_vector.h"
#include "cusparse.h"

namespace ceres::internal {

// A sparse matrix hosted on the GPU in compressed row sparse format, with
// CUDA-accelerated operations.
class CERES_NO_EXPORT CudaSparseMatrix {
 public:
  // Create a GPU copy of the matrix provided. The caller must ensure that
  // InitCuda() has already been successfully called on context before calling
  // this constructor.
  CudaSparseMatrix(ContextImpl* context,
                   const CompressedRowSparseMatrix& crs_matrix);

  ~CudaSparseMatrix();

  // y = y + Ax;
  void RightMultiplyAndAccumulate(const CudaVector& x, CudaVector* y);
  // y = y + A'x;
  void LeftMultiplyAndAccumulate(const CudaVector& x, CudaVector* y);

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }
  int num_nonzeros() const { return num_nonzeros_; }

  // If subsequent uses of this matrix involve only numerical changes and no
  // structural changes, then this method can be used to copy the updated
  // non-zero values -- the row and column index arrays are kept  the same. It
  // is the caller's responsibility to ensure that the sparsity structure of the
  // matrix is unchanged.
  void CopyValuesFromCpu(const CompressedRowSparseMatrix& crs_matrix);

  const cusparseSpMatDescr_t& descr() const { return descr_; }

 private:
  // Disable copy and assignment.
  CudaSparseMatrix(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;

  // y = y + op(M)x. op must be either CUSPARSE_OPERATION_NON_TRANSPOSE or
  // CUSPARSE_OPERATION_TRANSPOSE.
  void SpMv(cusparseOperation_t op, const CudaVector& x, CudaVector* y);

  int num_rows_ = 0;
  int num_cols_ = 0;
  int num_nonzeros_ = 0;

  ContextImpl* context_ = nullptr;
  // CSR row indices.
  CudaBuffer<int32_t> rows_;
  // CSR column indices.
  CudaBuffer<int32_t> cols_;
  // CSR values.
  CudaBuffer<double> values_;

  // CuSparse object that describes this matrix.
  cusparseSpMatDescr_t descr_ = nullptr;

  CudaBuffer<uint8_t> spmv_buffer_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_SPARSE_MATRIX_H_
