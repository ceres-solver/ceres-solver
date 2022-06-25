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
#include <string>

#include "ceres/block_sparse_matrix.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/export.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "ceres/context_impl.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_buffer.h"
#include "ceres/cuda_linear_operator.h"
#include "ceres/cuda_vector.h"
#include "cusparse.h"

namespace ceres::internal {

class CERES_NO_EXPORT CudaSparseMatrix : public CudaLinearOperator {
 public:
  CudaSparseMatrix() {};
  // ~CudaSparseMatrix() = default;

  bool Init(ContextImpl* context, std::string* message);

  // y = y + Ax;
  void RightMultiply(const CudaVector& x, CudaVector* y) override;
  // y = y + A'x;
  void LeftMultiply(const CudaVector& x, CudaVector* y) override;

  int num_rows() const override { return num_rows_; }
  int num_cols() const override { return num_cols_; }

  void CopyFrom(const CRSMatrix& crs_matrix);
  void CopyFrom(const BlockSparseMatrix& bs_matrix);
  void CopyFrom(const TripletSparseMatrix& ts_matrix);

  const cusparseSpMatDescr_t& descr() const { return csr_descr_; }

 private:
  // Disable copy and assignment.
  CudaSparseMatrix(const CudaSparseMatrix&) = delete;
  CudaSparseMatrix& operator=(const CudaSparseMatrix&) = delete;

  // Convenience wrapper around cusparseSpMV to compute y = y + op(A)x.
  void SpMv(cusparseOperation_t op, const CudaVector& x, CudaVector* y);

  int num_rows_ = 0;
  int num_cols_ = 0;

  // CSR row indices.
  CudaBuffer<int32_t> csr_row_indices_;
  // CSR column indices.
  CudaBuffer<int32_t> csr_col_indices_;
  // CSR values.
  CudaBuffer<double> csr_values_;

  ContextImpl* context_ = nullptr;

  // CuSparse object that describes this matrix.
  cusparseSpMatDescr_t csr_descr_ = nullptr;

  CudaBuffer<uint8_t> buffer_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_SPARSE_MATRIX_H_
