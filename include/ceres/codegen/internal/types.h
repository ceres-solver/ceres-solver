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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_TYPES_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_TYPES_H_

#include "ceres/codegen/macros.h"

namespace ceres {
// The return type of a comparison, for example from <, &&, ==.
//
// In the context of traditional Ceres Jet operations, this would
// always be a bool. However, in the autodiff code generation context,
// the return is always an expression, and so a different type must be
// used as a return from comparisons.
//
// In the autodiff codegen context, this function is overloaded so that 'type'
// is one of the autodiff code generation expression types.
template <typename T>
struct ComparisonReturnType {
  using type = bool;
};

namespace internal {
// The InputAssignment struct is used to implement the CERES_LOCAL_VARIABLE
// macro defined in macros.h. The input is a double variable and the
// corresponding name as a string. During execution mode (T==double or
// T==Jet<double>) the variable name is unused and this function only returns
// the value. During code generation (T==ExpressionRef) the value is unused and
// an assignment expression is created. For example:
//   v_0 = observed_x;
template <typename T>
struct InputAssignment {
  static inline T Get(double v, const char* /* unused */) { return v; }
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_TYPES_H_
