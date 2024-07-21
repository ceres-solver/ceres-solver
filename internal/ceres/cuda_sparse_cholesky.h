// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2024 Google Inc. All rights reserved.
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
// Author: markshachkov@gmail.com (Mark Shachkov)
//
// A C++ interface to cuDSS.

#ifndef CERES_INTERNAL_CUDA_SPARSE_CHOLESKY_H_
#define CERES_INTERNAL_CUDA_SPARSE_CHOLESKY_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDSS

#include <memory>

#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"

namespace ceres::internal {

// This class is a factory for implementation of sparse cholesky that uses cuDSS
// on CUDA capable GPU's to solve sparse linear systems. Scalar controls the
// precision used during computations, currently float and double are supported.
// Details of implementation are encapsulated into cuda_sparse_cholesky.cc
template <typename Scalar = double>
class CERES_NO_EXPORT CudaSparseCholesky : public SparseCholesky {
 public:
  static constexpr bool IsNestedDissectionAvailable() noexcept { return false; }

  static std::unique_ptr<SparseCholesky> Create(
      ContextImpl* context, const OrderingType ordering_type);
};

}  // namespace ceres::internal

#endif  // CERES_NO_CUDSS

#endif  // CERES_INTERNAL_CUDA_SPARSE_CHOLESKY_H_
