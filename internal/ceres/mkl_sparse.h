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
// Author: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)
//
// A simple C++ interface to Intel MKL PARDISO solver

#ifndef CERES_INTERNAL_MKL_SPARSE_H_
#define CERES_INTERNAL_MKL_SPARSE_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"

#ifdef CERES_USE_MKL

namespace ceres::internal {

namespace MKLUtils {
// Compute AtA product
CompressedRowSparseMatrix AtA(const CompressedRowSparseMatrix& m);
}  // namespace MKLUtils

// Compute column ordering for sparse normal Cholesky
void MKLComputeOrdering(CompressedRowSparseMatrix& A,
                        const LinearSolverOrderingType ordering_type,
                        int* ordering);
// Compute column ordering for Schur-complement solver
void MKLComputeOrderingSchurComplement(
    CompressedRowSparseMatrix& E,
    CompressedRowSparseMatrix& F,
    const LinearSolverOrderingType ordering_type,
    int* ordering);

class MKLPardiso;
class CERES_NO_EXPORT MKLSparseCholesky final : public SparseCholesky {
 public:
  static std::unique_ptr<MKLSparseCholesky> Create(
      const OrderingType ordering_type);

  ~MKLSparseCholesky() override;
  CompressedRowSparseMatrix::StorageType StorageType() const final;
  LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                        std::string* message) final;
  LinearSolverTerminationType Solve(const double* rhs,
                                    double* solution,
                                    std::string* message) final;

 private:
  explicit MKLSparseCholesky(const OrderingType ordering_type);

  const OrderingType ordering_type_;
  std::unique_ptr<MKLPardiso> mkl_;
  bool analyzed_ = false;
};

}  // namespace ceres::internal

#endif  // CERES_USE_MKL
#endif  // CERES_INTERNAL_MKL_SPARSE_H_
