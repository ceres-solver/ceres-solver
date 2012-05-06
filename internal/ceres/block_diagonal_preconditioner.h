// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)

#ifndef CERES_INTERNAL_BLOCK_DIAGONAL_PRECONDITIONER_H_
#define CERES_INTERNAL_BLOCK_DIAGONAL_PRECONDITIONER_H_

#include <vector>
#include "ceres/linear_operator.h"

namespace ceres {
namespace internal {

class CompressedRowBlockStructure;
class LinearOperator;
class SparseMatrix;

// A block diagonal preconditioner; also known as block-Jacobi.
class BlockDiagonalPreconditioner : public LinearOperator {
 public:
  BlockDiagonalPreconditioner(const LinearOperator& A);
  virtual ~BlockDiagonalPreconditioner();

  void Update(const LinearOperator& matrix);

  virtual void RightMultiply(const double* x, double* y) const;
  virtual void LeftMultiply(const double* x, double* y) const;

  virtual int num_rows() const { return size_; }
  virtual int num_cols() const { return size_; }

 private:
  std::vector<double*> blocks_;
  std::vector<double> block_storage_;
  int size_;
  const CompressedRowBlockStructure& block_structure_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_LINEAR_OPERATOR_H_
