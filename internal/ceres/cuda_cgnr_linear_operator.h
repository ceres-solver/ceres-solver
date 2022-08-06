// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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

#ifndef CERES_INTERNAL_CUDA_CGNR_LINEAR_OPERATOR_H_
#define CERES_INTERNAL_CUDA_CGNR_LINEAR_OPERATOR_H_

#include <algorithm>
#include <memory>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"

#ifndef CERES_NO_CUDA

#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"

namespace ceres::internal {

// A linear operator which takes a matrix A and a diagonal vector D and
// performs products of the form
//
//   (A^T A + D^T D)x
//
// This is used to implement iterative general sparse linear solving with
// conjugate gradients, where A is the Jacobian and D is a regularizing
// parameter. A brief proof is included in cgnr_linear_operator.h.
class CERES_NO_EXPORT CudaCgnrLinearOperator final : public CudaLinearOperator {
 public:
  CudaCgnrLinearOperator() = default;

  bool Init(CudaLinearOperator* A,
            CudaVector* D,
            ContextImpl* context,
            std::string* message) {
    CHECK(message != nullptr);
    A_ = A;
    D_ = D;
    return z_.Init(context, message);
  }

  void RightMultiply(const CudaVector& x, CudaVector* y) final {
    CHECK(A_ != nullptr);
    CHECK(y != nullptr);
    CHECK_EQ(y->num_rows(), A_->num_cols());
    // z = Ax
    printf("Resizing z to %d\n", A_->num_rows());
    z_.resize(A_->num_rows());
    printf("Setting z to zero\n");
    z_.setZero();
    printf("Right multiplying A\n");
    A_->RightMultiply(x, &z_);

    // y = y + Atz
    //   = y + AtAx
    printf("Right multiplying A^T\n");
    A_->LeftMultiply(z_, y);

    // y = y + DtDx
    if (D_ != nullptr) {
      printf("Right multiplying D^T\n");
      y->DtDxpy(*D_, x);
    }
  }

  void LeftMultiply(const CudaVector& x, CudaVector* y) {
    RightMultiply(x, y);
  }

  int num_rows() const final { return A_->num_cols(); }
  int num_cols() const final { return A_->num_cols(); }

 private:
  CudaLinearOperator* A_ = nullptr;
  CudaVector* D_ = nullptr;
  CudaVector z_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDA

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_CUDA_CGNR_LINEAR_OPERATOR_H_
