// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2033 Google Inc. All rights reserved.
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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)
//
// A wrapper for using cpu preconditioners with gpu vectors

#ifndef CERES_INTERNAL_CUDA_PRECONDITIONER_WRAPPER_H_
#define CERES_INTERNAL_CUDA_PRECONDITIONER_WRAPPER_H_
#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "ceres/preconditioner.h"
#include "ceres/schur_jacobi_preconditioner.h"

namespace ceres::internal {

class CERES_NO_EXPORT CudaPreconditionerWrapper : public Preconditioner {
 public:
  CudaPreconditionerWrapper(std::unique_ptr<Preconditioner> preconditioner,
                            ContextImpl* context);

  // RightMultiplyAndAccumulate uses temporary storage on cpu-side and is not
  // thread-safe
  void RightMultiplyAndAccumulate(const double* x, double* y) const override;
  int num_rows() const override;

  bool Update(const LinearOperator& A, const double* D) override;

 private:
  std::unique_ptr<Preconditioner> preconditioner_;
  mutable Vector x_;
  mutable Vector y_;
  ContextImpl* context_;
};

class CERES_NO_EXPORT CudaIdentityPreconditionerBlockSparse
    : public Preconditioner {
 public:
  CudaIdentityPreconditionerBlockSparse(const int num_rows,
                                        ContextImpl* context)
      : num_rows_(num_rows), context_(context) {}

  void RightMultiplyAndAccumulate(const double* x, double* y) const {
    const double a = 1.;
    CHECK_EQ(CUBLAS_STATUS_SUCCESS,
             cublasDaxpy(context_->cublas_handle_, num_rows_, &a, x, 1, y, 1));
  }
  int num_rows() const { return num_rows_; };

  bool Update(const LinearOperator& /*A*/, const double* /* D*/) {
    return true;
  };

 private:
  const int num_rows_;
  ContextImpl* context_;
};

class CERES_NO_EXPORT CudaSchurJacobiPreconditioner
    : public BlockSparseMatrixPreconditioner {
 public:
  CudaSchurJacobiPreconditioner(
      std::unique_ptr<SchurJacobiPreconditioner> preconditioner,
      ContextImpl* context);

  void RightMultiplyAndAccumulate(const double* x, double* y) const;
  int num_rows() const;

 private:
  bool UpdateImpl(const BlockSparseMatrix& A, const double* D);
  std::unique_ptr<SchurJacobiPreconditioner> preconditioner_;
  std::unique_ptr<CudaSparseMatrix> m_;
  ContextImpl* context_;
};

}  // namespace ceres::internal

#endif
#endif
