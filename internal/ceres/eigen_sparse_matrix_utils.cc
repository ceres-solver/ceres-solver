// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: mike@hidof.com (Michael Vitus)
// Adapted from the compressed_col_sparse_matrix_utils.h

#include "ceres/eigen_sparse_matrix_utils.h"

namespace ceres {
namespace internal {

void SolveUpperTriangularInPlace(const Eigen::SparseMatrix<double>& R,
                                 Eigen::VectorXd* rhs_and_solution_ptr) {
  Eigen::VectorXd& rhs_and_solution = *rhs_and_solution_ptr;
  for (int col = R.cols() - 1; col >= 0; --col) {
    Eigen::SparseMatrix<double>::ReverseInnerIterator diag(R, col);
    rhs_and_solution[col] /= diag.value();
    for (Eigen::SparseMatrix<double>::InnerIterator it(R, col); it; ++it) {
      if (it.row() == col) {
        continue;
      }
      rhs_and_solution[it.row()] -= it.value() * rhs_and_solution[col];
    }
  }
}

void SolveRTRWithSparseRHS(const Eigen::SparseMatrix<double>& R,
                           const int rhs_nonzero_index,
                           Eigen::VectorXd* solution_ptr) {
  Eigen::VectorXd& solution = *solution_ptr;
  solution.setZero();

  // The following loop solves for the solution to the linear system
  //
  //  R' x = e_i
  //
  // where e_i is a vector with e(i) = 1 and all other entries zero.
  Eigen::SparseMatrix<double>::ReverseInnerIterator
      it_non_zero_index(R, rhs_nonzero_index);
  solution(rhs_nonzero_index) = 1.0 / it_non_zero_index.value();

  for (int col = rhs_nonzero_index + 1; col < R.outerSize(); ++col) {
    double diagonal_value = 1.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(R, col); it; ++it) {
      if (it.row() < rhs_nonzero_index)
        continue;
      if (it.row() == col) {
        diagonal_value = it.value();
        continue;
      }
      solution(col) -= it.value() * solution(it.row());
    }
    solution(col) /= diagonal_value;
  }
  SolveUpperTriangularInPlace(R, solution_ptr);
}

}  // namespace internal
}  // namespace ceres
