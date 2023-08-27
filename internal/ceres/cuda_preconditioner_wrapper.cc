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

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_preconditioner_wrapper.h"

namespace ceres::internal {

CudaPreconditionerWrapper::CudaPreconditionerWrapper(
    std::unique_ptr<Preconditioner> preconditioner, ContextImpl* context)
    : preconditioner_(std::move(preconditioner)),
      x_(preconditioner_->num_rows()),
      y_(preconditioner_->num_rows()),
      context_(context) {}

void CudaPreconditionerWrapper::RightMultiplyAndAccumulate(const double* x,
                                                           double* y) const {
  CHECK_EQ(cudaSuccess,
           cudaMemcpyAsync(x_.data(),
                           x,
                           sizeof(double) * num_rows(),
                           cudaMemcpyDeviceToHost,
                           context_->DefaultStream()));
  CHECK_EQ(cudaSuccess,
           cudaMemcpyAsync(y_.data(),
                           y,
                           sizeof(double) * num_rows(),
                           cudaMemcpyDeviceToHost,
                           context_->DefaultStream()));
  CHECK_EQ(cudaSuccess, cudaStreamSynchronize(context_->DefaultStream()));
  preconditioner_->RightMultiplyAndAccumulate(x_.data(), y_.data());
  CHECK_EQ(cudaSuccess,
           cudaMemcpyAsync(y,
                           y_.data(),
                           sizeof(double) * num_rows(),
                           cudaMemcpyHostToDevice,
                           context_->DefaultStream()));
}

int CudaPreconditionerWrapper::num_rows() const {
  return preconditioner_->num_rows();
}

bool CudaPreconditionerWrapper::Update(const LinearOperator& A,
                                       const double* D) {
  return preconditioner_->Update(A, D);
}

}  // namespace ceres::internal

#endif
