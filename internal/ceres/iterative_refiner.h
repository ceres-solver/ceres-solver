// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_ITERATIVE_REFINER_H_
#define CERES_INTERNAL_ITERATIVE_REFINER_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

class SparseMatrix;
class SparseCholesky;

// Iterative refinement
// (https://en.wikipedia.org/wiki/Iterative_refinement) is the process
// of improving the solution to a linear system, by using the
// following iteration.
//
// r_i = b - Ax_i
// Ad_i = r_i
// x_{i+1} = x_i + d_i
//
// IterativeRefiner implements this process for Symmetric Positive
// Definite linear systems.
//
// The above iterative loop is till max_num_iterations is reached or
// the following convergence criterion is satisfied:
//
//    |b - Ax|
// ------------- < 5e-15
// |A| |x| + |b|
//
// All norms in the above expression are max-norms. The above
// expression is what is recommended and used by Hogg & Scott in "A
// fast and robust mixed-precision solver for the solution of sparse
// symmetric linear systems".
//
// For example usage, please see sparse_normal_cholesky_solver.cc
class IterativeRefiner {
 public:
  struct Summary {
    bool converged = false;
    int num_iterations = -1;
    double lhs_max_norm = -1;
    double rhs_max_norm = -1;
    double solution_max_norm = -1;
    double residual_max_norm = -1;
  };

  // num_cols is the number of rows & columns in the linear system
  // being solved.
  //
  // max_num_iterations is the maximum number of refinement iterations
  // to perform.
  IterativeRefiner(int num_cols, int max_num_iterations);

  // sparse_cholesky is assumed to contain an already computed
  // factorization (or approximation thereof) of lhs.
  //
  // solution is expected to contain a approximation to the solution
  // to lhs * x = rhs. It can be zero.
  Summary Refine(const SparseMatrix& lhs,
                 const double* rhs,
                 SparseCholesky* sparse_cholesky,
                 double* solution);

 private:
  int num_cols_;
  int max_num_iterations_;
  Vector residual_;
  Vector correction_;
  Vector lhs_x_solution_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_ITERATIVE_REFINER_H_
