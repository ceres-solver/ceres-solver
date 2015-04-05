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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Wrappers around Eigen's dense Cholesky factorization
// routines. Normally, if CERES_USE_LDLT_FOR_EIGEN_CHOLESKY is not
// defined Eigen's LLT factorization is used, which is faster than the
// LDLT factorization, however on ARM the LLT factorization seems to
// give significantly worse results than LDLT and we are forced to use
// LDLT instead.
//
// These wrapper functions provide a level of indirection to deal with
// this switching and hide it from the rest of the code base.

#ifndef CERES_INTERNAL_ARRAY_UTILS_H_
#define CERES_INTERNAL_ARRAY_UTILS_H_

#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

// Invert a matrix using Eigen's dense Cholesky factorization. values
// and inverse_values can point to the same array.
Eigen::ComputationInfo
InvertUpperTriangularUsingCholesky(int size,
                                   const double* values,
                                   double* inverse_values);

// Solve a linear system using Eigen's dense Cholesky
// factorization. rhs_values and solution can point to the same array.
Eigen::ComputationInfo
SolveUpperTriangularUsingCholesky(int size,
                                  const double* lhs_values,
                                  const double* rhs_values,
                                  double* solution);

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_ARRAY_UTILS_H_
