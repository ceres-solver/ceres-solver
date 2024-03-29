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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// Base classes for access to an linear operator.

#ifndef CERES_INTERNAL_LINEAR_OPERATOR_H_
#define CERES_INTERNAL_LINEAR_OPERATOR_H_

#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "ceres/types.h"

namespace ceres::internal {

class ContextImpl;

// This is an abstract base class for linear operators. It supports
// access to size information and left and right multiply operators.
class CERES_NO_EXPORT LinearOperator {
 public:
  virtual ~LinearOperator();

  // y = y + Ax;
  virtual void RightMultiplyAndAccumulate(const double* x, double* y) const = 0;
  virtual void RightMultiplyAndAccumulate(const double* x,
                                          double* y,
                                          ContextImpl* context,
                                          int num_threads) const;
  // y = y + A'x;
  virtual void LeftMultiplyAndAccumulate(const double* x, double* y) const = 0;
  virtual void LeftMultiplyAndAccumulate(const double* x,
                                         double* y,
                                         ContextImpl* context,
                                         int num_threads) const;

  virtual void RightMultiplyAndAccumulate(const Vector& x, Vector& y) const {
    RightMultiplyAndAccumulate(x.data(), y.data());
  }

  virtual void LeftMultiplyAndAccumulate(const Vector& x, Vector& y) const {
    LeftMultiplyAndAccumulate(x.data(), y.data());
  }

  virtual void RightMultiplyAndAccumulate(const Vector& x,
                                          Vector& y,
                                          ContextImpl* context,
                                          int num_threads) const {
    RightMultiplyAndAccumulate(x.data(), y.data(), context, num_threads);
  }

  virtual void LeftMultiplyAndAccumulate(const Vector& x,
                                         Vector& y,
                                         ContextImpl* context,
                                         int num_threads) const {
    LeftMultiplyAndAccumulate(x.data(), y.data(), context, num_threads);
  }

  virtual int num_rows() const = 0;
  virtual int num_cols() const = 0;
};

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_LINEAR_OPERATOR_H_
