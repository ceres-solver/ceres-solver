// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: richie.stebbing@gmail.com (Richard Stebbing)
//
// A solver for sparse linear least squares problem based on solving
// the normal equations via a sparse cholesky factorization.

#ifndef CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_
#define CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_

#if !defined(CERES_NO_SUITESPARSE)

#include "ceres/internal/macros.h"
#include "ceres/linear_solver.h"
#include "ceres/suitesparse.h"

namespace ceres {
namespace internal {

class CompressedRowSparseMatrix;

// Solves the normal equations (A'A + D'D) x = A'b, using the CHOLMOD sparse
// cholesky solver but with sparsity analysis at each iteration.
class DynamicSparseNormalCholeskySolver :
  public CompressedRowSparseMatrixSolver {
 public:
  explicit DynamicSparseNormalCholeskySolver(
    const LinearSolver::Options& options);
  virtual ~DynamicSparseNormalCholeskySolver();

 private:
  virtual LinearSolver::Summary SolveImpl(
      CompressedRowSparseMatrix* A,
      const double* b,
      const LinearSolver::PerSolveOptions& options,
      double* x);

  LinearSolver::Summary SolveImplUsingSuiteSparse(
      CompressedRowSparseMatrix* A,
      const LinearSolver::PerSolveOptions& options,
      double* rhs_and_solution);

  SuiteSparse ss_;

  const LinearSolver::Options options_;
  CERES_DISALLOW_COPY_AND_ASSIGN(DynamicSparseNormalCholeskySolver);
};

}  // namespace internal
}  // namespace ceres

#endif  // !defined(CERES_NO_SUITESPARSE)
#endif  // CERES_INTERNAL_DYNAMIC_SPARSE_NORMAL_CHOLESKY_SOLVER_H_
