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

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include "ceres/cuda_sparse_matrix.h"

#include <math.h>

#include "ceres/internal/export.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/crs_matrix.h"
#include "ceres/types.h"
#include "ceres/context_impl.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_buffer.h"
#include "ceres/cuda_vector.h"
#include "ceres/ceres_cuda_kernels.h"
#include "cusparse.h"


namespace ceres::internal {

bool CudaSparseMatrix::Init(ContextImpl* context, std::string* message) {
  CHECK(message != nullptr);
  if (context == nullptr) {
    *message = "CudaVector::Init: context is nullptr";
    return false;
  }
  if (!context->InitCUDA(message)) {
    *message = "CudaVector::Init: context->InitCUDA() failed";
    return false;
  }
  context_ = context;
  return true;
}

void CudaSparseMatrix::CopyFrom(const CRSMatrix& crs_matrix) {
  csr_row_indices_.CopyFromCpuAsync(crs_matrix.rows, context_->stream_);
  csr_col_indices_.CopyFromCpuAsync(crs_matrix.cols, context_->stream_);
  csr_values_.CopyFromCpuAsync(crs_matrix.values, context_->stream_);
  num_rows_ = crs_matrix.num_rows;
  num_cols_ = crs_matrix.num_cols;
  cusparseCreateCsr(&csr_descr_,
                    num_rows_,
                    num_cols_,
                    crs_matrix.values.size(),
                    csr_row_indices_.data(),
                    csr_col_indices_.data(),
                    csr_values_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
}

void CudaSparseMatrix::CopyFrom(const BlockSparseMatrix& bs_matrix) {
  CRSMatrix crs_matrix;
  bs_matrix.ToCRSMatrix(&crs_matrix);
  CopyFrom(crs_matrix);
}

void CudaSparseMatrix::CopyFrom(const TripletSparseMatrix& ts_matrix) {
  CRSMatrix crs_matrix;
  ts_matrix.ToCRSMatrix(&crs_matrix);
  CopyFrom(crs_matrix);
}

void CudaSparseMatrix::SpMv(cusparseOperation_t op,
                            const CudaVector& x,
                            CudaVector* y) {
  size_t buffer_size = 0;
  y->resize(num_rows_);
  const double alpha = 1.0;
  const double beta = 1.0;

  CHECK_EQ(cusparseSpMV_bufferSize(context_->cusparse_handle_,
                                   op,
                                   &alpha,
                                   csr_descr_,
                                   x.descr(),
                                   &beta,
                                   y->descr(),
                                   CUDA_R_64F,
                                   CUSPARSE_SPMV_ALG_DEFAULT,
                                   &buffer_size),
           CUSPARSE_STATUS_SUCCESS);
  buffer_.Reserve(buffer_size);
  CHECK_EQ(cusparseSpMV(context_->cusparse_handle_,
                        op,
                        &alpha,
                        csr_descr_,
                        x.descr(),
                        &beta,
                        y->descr(),
                        CUDA_R_64F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        buffer_.data()),
           CUSPARSE_STATUS_SUCCESS);
}

void CudaSparseMatrix::RightMultiply(const CudaVector& x, CudaVector* y) {
  SpMv(CUSPARSE_OPERATION_NON_TRANSPOSE, x, y);
}

void CudaSparseMatrix::LeftMultiply(const CudaVector& x, CudaVector* y) {
  // TODO(Joydeep Biswas): If this operation is to be done frequently, we should
  // store a CSC format of the matrix, which is incidentally the CSR format of
  // the matrix transpose, and call cusparseSpMV with
  // CUSPARSE_OPERATION_NON_TRANSPOSE. From the cuSPARSE documentation:
  // "In general, opA == CUSPARSE_OPERATION_NON_TRANSPOSE is 3x faster than opA
  // != CUSPARSE_OPERATION_NON_TRANSPOSE"
  SpMv(CUSPARSE_OPERATION_TRANSPOSE, x, y);
}

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA