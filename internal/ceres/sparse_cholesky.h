// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2017 Google Inc. All rights reserved.
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

#ifndef CERES_INTERNAL_SPARSE_CHOLESKY_H_
#define CERES_INTERNAL_SPARSE_CHOLESKY_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#include "ceres/linear_solver.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

// TODO(sameeragarwal): It is likely the case that some sparse linear
// algebra libraries do not like upper/lower triangular matrices, this
// should be sorted out and dealt with in the API.
//
// Document class.
class SparseCholesky {
 public:
  static SparseCholesky* Create(
      SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type,
      OrderingType ordering_type);
  virtual ~SparseCholesky();
  virtual LinearSolverTerminationType Factorize(
      CompressedRowSparseMatrix* lhs, std::string* message) = 0;
  virtual LinearSolverTerminationType Solve(const double* rhs,
                                            double* solution,
                                            std::string* message) = 0;
  virtual LinearSolverTerminationType FactorAndSolve(
      CompressedRowSparseMatrix* lhs,
      const double* rhs,
      double* solution,
      std::string* message);

};

// Different sparse linear algebra libraries prefer different storage
// orders for the input matrix. This trait class helps choose the
// ordering based on the sparse linear algebra backend being used.
//
// The storage order is lower-triangular by default. It is only
// SuiteSparse which prefers an upper triangular matrix. Saves a whole
// matrix copy in the process.
//
// Note that this is the storage order for a compressed row sparse
// matrix. All the sparse linear algebra libraries take compressed
// column sparse matrices as input. We map these matrices to into
// compressed column sparse matrices before calling them and in the
// process, transpose them.
//
// TODO(sameeragarwal): For SuiteSparse, this does not account for
// post ordering, where the optimal storage order maybe different.
CompressedRowSparseMatrix::StorageType StorageTypeForSparseLinearAlgebraLibrary(
    SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type);

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SPARSE_CHOLESKY_H_
