// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// This file includes a few synthetic cost functors to test the functionality of
// the Expression logic and code generation.

#ifndef CERES_CODE_GENERATION_FUNCTORS_H_
#define CERES_CODE_GENERATION_FUNCTORS_H_

#include "ceres/internal/expression_arithmetic.h"

namespace ceres {
namespace internal {

struct TestAllExpressions {
  TestAllExpressions(double local_variable) {}
  template <typename T>
  bool operator()(const T* const _x, T* residuals) const {
    auto& x = *_x;

    T tmp;
    tmp = x;

    // Arith. Operators
    tmp = tmp * x;
    tmp = tmp + x;
    tmp = x - tmp;
    tmp = tmp / x;
    tmp = -tmp;

    // Compount Operators
    tmp += x;
    tmp -= x;
    tmp *= x;
    tmp /= x;

    // Functions
    tmp += sin(x);
    tmp += cos(x);
    tmp += sqrt(x);
    tmp += exp(x);
    tmp += log(x);
    tmp += floor(x);

    // Comparison
    auto c1 = tmp < x;
    auto c2 = tmp > x;
    auto c3 = tmp <= x;
    auto c4 = tmp >= x;
    auto c5 = tmp == x;
    auto c6 = tmp != x;

    // Logical
    auto c7 = c1 && c2 && c3 && c4 && c5 && c6;
    auto c8 = c1 || c2 || c3 || c4 || c5 || c6;

    // Ternary  ?-operator
    tmp = ternary(c7, tmp, T(2) * tmp);
    tmp = ternary(c8, tmp, T(2) * tmp);

    // local variables
    tmp *= CERES_EXPRESSION_EXTERNAL_CONSTANT(local_variable_);

    residuals[0] = tmp;
    return true;
  }

  double local_variable_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_CODE_GENERATION_FUNCTORS_H_
