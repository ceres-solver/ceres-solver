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

#include <memory>

#include "ceres/block_sparse_matrix.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/context_impl.h"
#include "ceres/crs_matrix.h"
#include "ceres/internal/export.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_buffer.h"
#include "ceres/cuda_kernels.h"
#include "ceres/cuda_vector.h"
#include "cuda_runtime_api.h"
#include "cusparse.h"

namespace ceres::internal {

CudaSparseMatrix::CudaSparseMatrix(ContextImpl* context,
                                   const CompressedRowSparseMatrix& crs_matrix)
    : context_(context),
      rows_{context},
      cols_{context},
      values_{context},
      spmv_buffer_{context} {
  DCHECK_NE(context, nullptr);
  CHECK(context->IsCudaInitialized());
  num_rows_ = crs_matrix.num_rows();
  num_cols_ = crs_matrix.num_cols();
  num_nonzeros_ = crs_matrix.num_nonzeros();
  rows_.CopyFromCpu(crs_matrix.rows(), num_rows_ + 1);
  cols_.CopyFromCpu(crs_matrix.cols(), num_nonzeros_);
  values_.CopyFromCpu(crs_matrix.values(), num_nonzeros_);
  cusparseCreateCsr(&descr_,
                    num_rows_,
                    num_cols_,
                    num_nonzeros_,
                    rows_.data(),
                    cols_.data(),
                    values_.data(),
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_64F);
}

CudaSparseMatrix::~CudaSparseMatrix() {
  CHECK_EQ(cusparseDestroySpMat(descr_), CUSPARSE_STATUS_SUCCESS);
  descr_ = nullptr;
}

void CudaSparseMatrix::CopyValuesFromCpu(
    const CompressedRowSparseMatrix& crs_matrix) {
  // There is no quick and easy way to verify that the structure is unchanged,
  // but at least we can check that the size of the matrix and the number of
  // nonzeros is unchanged.
  CHECK_EQ(num_rows_, crs_matrix.num_rows());
  CHECK_EQ(num_cols_, crs_matrix.num_cols());
  CHECK_EQ(num_nonzeros_, crs_matrix.num_nonzeros());
  values_.CopyFromCpu(crs_matrix.values(), num_nonzeros_);
}

void CudaSparseMatrix::SpMv(cusparseOperation_t op,
                            const CudaVector& x,
                            CudaVector* y) {
  size_t buffer_size = 0;
  const double alpha = 1.0;
  const double beta = 1.0;

  // Starting in CUDA 11.2.1, CUSPARSE_MV_ALG_DEFAULT was deprecated in favor of
  // CUSPARSE_SPMV_ALG_DEFAULT.
#if CUDART_VERSION >= 11021
  const auto algorithm = CUSPARSE_SPMV_ALG_DEFAULT;
#else   // CUDART_VERSION >= 11021
  const auto algorithm = CUSPARSE_MV_ALG_DEFAULT;
#endif  // CUDART_VERSION >= 11021

  CHECK_EQ(cusparseSpMV_bufferSize(context_->cusparse_handle_,
                                   op,
                                   &alpha,
                                   descr_,
                                   x.descr(),
                                   &beta,
                                   y->descr(),
                                   CUDA_R_64F,
                                   algorithm,
                                   &buffer_size),
           CUSPARSE_STATUS_SUCCESS);
  spmv_buffer_.Reserve(buffer_size);
  CHECK_EQ(cusparseSpMV(context_->cusparse_handle_,
                        op,
                        &alpha,
                        descr_,
                        x.descr(),
                        &beta,
                        y->descr(),
                        CUDA_R_64F,
                        algorithm,
                        spmv_buffer_.data()),
           CUSPARSE_STATUS_SUCCESS);
}

void CudaSparseMatrix::RightMultiplyAndAccumulate(const CudaVector& x,
                                                  CudaVector* y) {
  SpMv(CUSPARSE_OPERATION_NON_TRANSPOSE, x, y);
}

void CudaSparseMatrix::LeftMultiplyAndAccumulate(const CudaVector& x,
                                                 CudaVector* y) {
  // TODO(Joydeep Biswas): We should consider storing a transposed copy of the
  // matrix by converting CSR to CSC. From the cuSPARSE documentation:
  // "In general, opA == CUSPARSE_OPERATION_NON_TRANSPOSE is 3x faster than opA
  // != CUSPARSE_OPERATION_NON_TRANSPOSE"
  SpMv(CUSPARSE_OPERATION_TRANSPOSE, x, y);
}

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA
