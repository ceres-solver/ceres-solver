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
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/crs_matrix.h"
#include "ceres/types.h"
#include "ceres/context_impl.h"
#include "ceres/wall_time.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_buffer.h"
#include "ceres/cuda_vector.h"
#include "ceres/ceres_cuda_kernels.h"
#include "cusparse.h"


namespace ceres::internal {

bool CudaSparseMatrix::Init(ContextImpl* context, std::string* message) {
  if (context == nullptr) {
    if (message) *message = "CudaVector::Init: context is nullptr";
    return false;
  }
  if (!context->InitCUDA(message)) {
    if (message) *message = "CudaVector::Init: context->InitCUDA() failed";
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
  num_nonzeros_ = crs_matrix.values.size();
  DestroyDescriptor();
  cusparseCreateCsr(&csr_descr_,
                    num_rows_,
                    num_cols_,
                    num_nonzeros_,
                    csr_row_indices_.data(),
                    csr_col_indices_.data(),
                    csr_values_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
}

void CudaSparseMatrix::CopyFrom(const CompressedRowSparseMatrix& crs_matrix) {
  num_rows_ = crs_matrix.num_rows();
  num_cols_ = crs_matrix.num_cols();
  num_nonzeros_ = crs_matrix.num_nonzeros();
  csr_row_indices_.CopyFromCpuAsync(
      crs_matrix.rows(), num_rows_ + 1, context_->stream_);
  csr_col_indices_.CopyFromCpuAsync(
      crs_matrix.cols(), num_nonzeros_, context_->stream_);
  csr_values_.CopyFromCpuAsync(
      crs_matrix.values(), num_nonzeros_, context_->stream_);
  DestroyDescriptor();
  cusparseCreateCsr(&csr_descr_,
                    num_rows_,
                    num_cols_,
                    num_nonzeros_,
                    csr_row_indices_.data(),
                    csr_col_indices_.data(),
                    csr_values_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
}

void CudaSparseMatrix::Resize(int num_rows, int num_cols, int num_nnz) {
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  csr_row_indices_.Reserve(num_rows + 1);
  csr_col_indices_.Reserve(num_nnz);
  csr_values_.Reserve(num_nnz);
  num_nonzeros_ = num_nnz;
  DestroyDescriptor();
  cusparseCreateCsr(&csr_descr_,
                    num_rows_,
                    num_cols_,
                    num_nonzeros_,
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

void CudaSparseMatrix::DestroyDescriptor() {
  if (csr_descr_) {
    CHECK_EQ(cusparseDestroySpMat(csr_descr_), CUSPARSE_STATUS_SUCCESS);
    csr_descr_ = nullptr;
  }
}

void CudaSparseMatrix::CopyFromTranspose(const CudaSparseMatrix& other) {
  num_rows_ = other.num_cols_;
  num_cols_ = other.num_rows_;
  csr_values_.Reserve(other.csr_values_.size());
  csr_row_indices_.Reserve(num_rows_ + 1);
  csr_col_indices_.Reserve(csr_values_.size());
  num_nonzeros_ = other.num_nonzeros_;
  DestroyDescriptor();
  cusparseCreateCsr(&csr_descr_,
                    num_rows_,
                    num_cols_,
                    num_nonzeros_,
                    csr_row_indices_.data(),
                    csr_col_indices_.data(),
                    csr_values_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
  size_t buffer_size = 0;
  CHECK_EQ(cusparseCsr2cscEx2_bufferSize(
      context_->cusparse_handle_,
      other.num_rows_,
      other.num_cols_,
      other.num_nonzeros_,
      other.csr_values_.data(),
      other.csr_row_indices_.data(),
      other.csr_col_indices_.data(),
      csr_values_.data(),
      csr_row_indices_.data(),
      csr_col_indices_.data(),
      CUDA_R_64F,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,
      &buffer_size), CUSPARSE_STATUS_SUCCESS);
  CudaBuffer<uint8_t> buffer;
  buffer.Reserve(buffer_size);
  CHECK_EQ(cusparseCsr2cscEx2(
      context_->cusparse_handle_,
      other.num_rows_,
      other.num_cols_,
      other.num_nonzeros_,
      other.csr_values_.data(),
      other.csr_row_indices_.data(),
      other.csr_col_indices_.data(),
      csr_values_.data(),
      csr_row_indices_.data(),
      csr_col_indices_.data(),
      CUDA_R_64F,
      CUSPARSE_ACTION_NUMERIC,
      CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1,
      buffer.data()), CUSPARSE_STATUS_SUCCESS);
  cudaDeviceSynchronize();
}

void CudaSparseMatrix::Multiply(const CudaSparseMatrix& A,
                                const CudaSparseMatrix& B) {
  printf("A has %d nonzeros\n", A.num_nonzeros());
  printf("B has %d nonzeros\n", B.num_nonzeros());
  // Create a temporary descriptor for the matrix M -- we don't yet know the
  // number of nonzeros, so we just set it to 1 to allocate some non-zero
  // memory.
  num_rows_ = A.num_rows_;
  num_cols_ = B.num_cols_;
  printf("Resizing M to %d x %d\n", num_rows_, num_cols_);
  DestroyDescriptor();
  CHECK_EQ(cusparseCreateCsr(
      &csr_descr_,
      num_rows_,
      num_cols_,
      0,
      nullptr,
      nullptr,
      nullptr,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F), CUSPARSE_STATUS_SUCCESS);
  const double alpha = 1.0;
  const double beta = 0.0;
  size_t buffer1_size = 0;
  cusparseSpGEMMDescr_t spgemm_desc;
  CHECK_EQ(cusparseSpGEMM_createDescr(&spgemm_desc), CUSPARSE_STATUS_SUCCESS);
  CHECK_EQ(cusparseSpGEMM_workEstimation(
      context_->cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr(),
      B.descr(),
      &beta,
      csr_descr_,
      CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT,
      spgemm_desc,
      &buffer1_size,
      nullptr), CUSPARSE_STATUS_SUCCESS);
  printf("buffer1_size = %lu\n", buffer1_size);

  CudaBuffer<uint8_t> buffer1;
  buffer1.Reserve(buffer1_size);
  CHECK_EQ(cusparseSpGEMM_workEstimation(
      context_->cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr(),
      B.descr(),
      &beta,
      csr_descr_,
      CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT,
      spgemm_desc,
      &buffer1_size,
      buffer1.data()), CUSPARSE_STATUS_SUCCESS);

  size_t buffer2_size = 0;
  CHECK_EQ(cusparseSpGEMM_compute(
      context_->cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr(),
      B.descr(),
      &beta,
      csr_descr_,
      CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT,
      spgemm_desc,
      &buffer2_size,
      nullptr), CUSPARSE_STATUS_SUCCESS);
  printf("buffer2_size = %lu\n", buffer2_size);

  // Allocate at most 1 GB of memory for the operation.
  buffer2_size = std::min<size_t>(buffer2_size, 1lu << 34);
  static CudaBuffer<uint8_t> buffer2;
  buffer2.Reserve(buffer2_size);
  CHECK_EQ(cusparseSpGEMM_compute(
      context_->cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr(),
      B.descr(),
      &beta,
      csr_descr_,
      CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT,
      spgemm_desc,
      &buffer2_size,
      buffer2.data()), CUSPARSE_STATUS_SUCCESS);

  int64_t new_num_rows, new_num_cols, new_nnz;
  CHECK_EQ(cusparseSpMatGetSize(
      csr_descr_,
      &new_num_rows,
      &new_num_cols,
      &new_nnz), CUSPARSE_STATUS_SUCCESS);
  printf("new_num_rows = %d\n", int(new_num_rows));
  printf("new_num_cols = %d\n", int(new_num_cols));
  printf("new_nnz = %d\n", int(new_nnz));
  // CHECK_NE(new_nnz, 0);
  Resize(new_num_rows, new_num_cols, new_nnz);
  CHECK_EQ(cusparseSpGEMM_copy(
      context_->cusparse_handle_,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      A.descr(),
      B.descr(),
      &beta,
      csr_descr_,
      CUDA_R_64F,
      CUSPARSE_SPGEMM_DEFAULT,
      spgemm_desc), CUSPARSE_STATUS_SUCCESS);
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