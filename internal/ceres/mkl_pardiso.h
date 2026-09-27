// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#ifndef CERES_INTERNAL_MKL_PARDISO_H_
#define CERES_INTERNAL_MKL_PARDISO_H_

#include "ceres/internal/config.h"

#ifndef CERES_NO_MKL

#include <memory>
#include <string>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"

namespace ceres::internal {

class PardisoSolver;

// Computes a fill reducing ordering of matrix using the PARDISO reordering
// phase. matrix is expected to use triangular storage. Exposed instead of
// PardisoSolver itself so that the ordering code does not depend on the
// solver's internals.
CERES_NO_EXPORT bool ComputePardisoOrdering(
    const CompressedRowSparseMatrix& matrix,
    LinearSolverOrderingType ordering_type,
    int max_num_threads,
    int* ordering,
    std::string* message);

class CERES_NO_EXPORT MklSparseCholesky final : public SparseCholesky {
 public:
  static std::unique_ptr<MklSparseCholesky> Create(
      OrderingType ordering_type,
      int max_num_threads,
      bool use_two_level_factorization);

  ~MklSparseCholesky() final;

  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  MklSparseCholesky(OrderingType ordering_type,
                    int max_num_threads,
                    bool use_two_level_factorization);

  std::unique_ptr<PardisoSolver> solver_;
};

}  // namespace ceres::internal

#endif  // CERES_NO_MKL

#endif  // CERES_INTERNAL_MKL_PARDISO_H_
