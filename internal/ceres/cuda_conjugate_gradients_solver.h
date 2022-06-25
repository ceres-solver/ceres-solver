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
// CUDA-Accelerated Conjugate Gradients based solver for positive
// semidefinite linear systems.

#ifndef CERES_INTERNAL_CUDA_CONJUGATE_GRADIENTS_SOLVER_H_
#define CERES_INTERNAL_CUDA_CONJUGATE_GRADIENTS_SOLVER_H_

// This include must come before any #ifndef check on Ceres compile options.
// clang-format off
#include "ceres/internal/config.h"
// clang-format on

#include <memory>

#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_linear_operator.h"
#include "ceres/cuda_vector.h"

namespace ceres::internal {

class CERES_NO_EXPORT CudaConjugateGradientsSolver {
 public:
  static std::unique_ptr<CudaConjugateGradientsSolver> Create(
      const LinearSolver::Options& options);

  bool Init(ContextImpl* context, std::string* message);

  LinearSolver::Summary Solve(
      CudaLinearOperator* A,
      CudaLinearOperator* preconditioner,
      const CudaVector& b,
      const LinearSolver::PerSolveOptions& per_solve_options,
      CudaVector* x);

 private:
  explicit CudaConjugateGradientsSolver(LinearSolver::Options options) :
      options_(options) { }
  const LinearSolver::Options options_;
  ContextImpl* context_ = nullptr;

  CudaVector r_;
  CudaVector p_;
  CudaVector z_;
  CudaVector tmp_;
};

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_NO_CUDA
#endif  // CERES_INTERNAL_CUDA_CONJUGATE_GRADIENTS_SOLVER_H_
