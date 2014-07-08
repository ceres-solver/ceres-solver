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

#ifndef CERES_INTERNAL_EIGEN_SPARSE_MATRIX_UTILS_H_
#define CERES_INTERNAL_EIGEN_SPARSE_MATRIX_UTILS_H_

#include "Eigen/SparseCore"

namespace ceres {
namespace internal {

// Solve the linear system
//
//   R * solution = rhs
//
// Where R is an upper triangular compressed column sparse matrix.
void SolveUpperTriangularInPlace(const Eigen::SparseMatrix<double>& R,
                                 Eigen::VectorXd* rhs_and_solution_ptr);

// Given a upper triangular matrix R in compressed column form, solve
// the linear system,
//
//  R'R x = b
//
// Where b is all zeros except for rhs_nonzero_index, where it is
// equal to one.
//
// The function exploits this knowledge to reduce the number of
// floating point operations.
void SolveRTRWithSparseRHS(const Eigen::SparseMatrix<double>& R,
                           const int rhs_nonzero_index,
                           Eigen::VectorXd* solution_ptr);

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_EIGEN_SPARSE_MATRIX_UTILS_H_
