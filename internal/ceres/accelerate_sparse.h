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

#ifndef CERES_INTERNAL_ACCELERATE_SPARSE_H_
#define CERES_INTERNAL_ACCELERATE_SPARSE_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifndef CERES_NO_ACCELERATE_SPARSE

#include <memory>
#include <string>
#include <vector>

#include "ceres/linear_solver.h"
#include "ceres/sparse_cholesky.h"
#include "Accelerate.h"

namespace ceres {
namespace internal {

class CompressedRowSparseMatrix;
class TripletSparseMatrix;

// An implementation of SparseCholesky interface using Apple's Accelerate
// framework.
class AppleAccelerateCholesky : public SparseCholesky {
 public:
  // Factory
  static std::unique_ptr<SparseCholesky> Create(OrderingType ordering_type);

  // SparseCholesky interface.
  virtual ~AppleAccelerateCholesky();
  virtual CompressedRowSparseMatrix::StorageType StorageType() const;
  virtual LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                                std::string* message);
  virtual LinearSolverTerminationType Solve(const double* rhs,
                                            double* solution,
                                            std::string* message);

 private:
  AppleAccelerateCholesky(const OrderingType ordering_type);

  const OrderingType ordering_type_;
};

class FloatAppleAccelerateCholesky : public SparseCholesky {
 public:
  // Factory
  static std::unique_ptr<SparseCholesky> Create(OrderingType ordering_type);

  // SparseCholesky interface.
  virtual ~FloatAppleAccelerateCholesky();
  virtual CompressedRowSparseMatrix::StorageType StorageType() const;
  virtual LinearSolverTerminationType Factorize(CompressedRowSparseMatrix* lhs,
                                                std::string* message);
  virtual LinearSolverTerminationType Solve(const double* rhs,
                                            double* solution,
                                            std::string* message);

 private:
  FloatAppleAccelerateCholesky(const OrderingType ordering_type);

  const OrderingType ordering_type_;
};

}
}

#endif  // CERES_NO_ACCELERATE_SPARSE

#endif  // CERES_INTERNAL_ACCELERATE_SPARSE_H_
