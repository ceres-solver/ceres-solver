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
//
// Interface to CUDA preconditioners.

#ifndef CERES_INTERNAL_CUDA_PRECONDITIONER_H_
#define CERES_INTERNAL_CUDA_PRECONDITIONER_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <memory>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"

namespace ceres::internal {

class CERES_NO_EXPORT CudaPreconditioner {
 public:

  bool Init(ContextImpl* context, std::string* message) {
    if (context == nullptr) {
      *message = "Context is nullptr.";
      return false;
    }
    if (!context->InitCUDA(message)) {
      return false;
    }
    context_ = context;
    return true;
  }

  // Update the numerical value of the preconditioner for the linear
  // system:
  //
  //  |   A   | x = |b|
  //  |diag(D)|     |0|
  //
  // for some vector b. It is important that the matrix A have the
  // same block structure as the one used to construct this object.
  //
  // D can be nullptr, in which case its interpreted as a diagonal matrix
  // of size zero.
  // Returns true iff the preconditioner was successfully updated. A failure
  // might be due to numerical problems (e.g. A is singular).
  virtual bool Update(const CudaSparseMatrix& A, const CudaVector& D) = 0;

  // Given a preconditioner matrix M, this returns y = M^-1 * x.
  // In general, this could perform in-place solving using factorized
  // preconditioners, for example, given an incomplete LU preconditioner
  // M = LU, this computes y = (LU)\x.
  virtual void Apply(const CudaVector& x, CudaVector* y) = 0;

  ContextImpl* context_ = nullptr;
};

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_PRECONDITIONER_H_
