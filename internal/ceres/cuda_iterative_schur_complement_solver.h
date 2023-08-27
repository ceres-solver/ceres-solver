// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#ifndef CERES_INTERNAL_CUDA_ITERATIVE_SCHUR_COMPLEMENT_SOLVER_H_
#define CERES_INTERNAL_CUDA_ITERATIVE_SCHUR_COMPLEMENT_SOLVER_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_CUDA
#include "ceres/cuda_vector.h"
#include "ceres/iterative_schur_complement_solver.h"

namespace ceres::internal {

class CERES_NO_EXPORT CudaIterativeSchurComplementSolver
    : public IterativeSchurComplementSolverBase {
 public:
  explicit CudaIterativeSchurComplementSolver(LinearSolver::Options options);

 private:
  double* reduced_linear_system_solution() override;
  void CreatePreconditioner(const BlockSparseMatrix* A) override;
  void CreatePreSolver(const int max_num_spse_iterations,
                       const double spse_tolerance) override;
  void Initialize() override;
  void BackSubstitute(const double* reduced_system_solution,
                      double* x) override;
  LinearSolver::Summary ReducedSolve(
      const ConjugateGradientsSolverOptions& cg_options) override;

  CudaVector reduced_linear_system_solution_;
  CudaVector x_;
  std::unique_ptr<CudaVector> scratch_[4];
};

}  // namespace ceres::internal

#endif
#endif
