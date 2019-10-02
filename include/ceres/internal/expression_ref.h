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
//
// This file contains the basic expression type, which is used during code
// creation. Only assignment expressions of the following form are supported:
//
// result = [constant|binary_expr|functioncall]
//
// Examples:
// v_78 = v_28 / v_62;
// v_97 = exp(v_20);
// v_89 = 3.000000;
//
//
#ifndef CERES_PUBLIC_EXPRESSION_REF_H_
#define CERES_PUBLIC_EXPRESSION_REF_H_

#include <string>
#include "ceres/jet.h"

namespace ceres {
namespace internal {

using ExpressionId = int;
static constexpr ExpressionId kInvalidExpressionId = -1;

// @brief A type-safe reference to 'Expression'.
//
// This class represents a scalar value that creates new expressions during
// evaluation. ExpressionRef can be used as template parameter for cost functors
// and Jets.
//
// ExpressionRef should be passed by value.
struct ExpressionRef {
  ExpressionRef() = default;

  // Create a constant expression directly from a double value.
  // v_0 = 123;
  ExpressionRef(double constant);

  // Returns v_id
  std::string ToString();

  // Compound operators (defined in expression_arithmetic.h)
  ExpressionRef& operator+=(ExpressionRef y);
  ExpressionRef& operator-=(ExpressionRef y);
  ExpressionRef& operator*=(ExpressionRef y);
  ExpressionRef& operator/=(ExpressionRef y);

  ExpressionId id = kInvalidExpressionId;
};

// Arith. Operators
ExpressionRef operator-(ExpressionRef f);
ExpressionRef operator+(ExpressionRef f);
ExpressionRef operator+(ExpressionRef f, ExpressionRef g);
ExpressionRef operator-(ExpressionRef f, ExpressionRef g);
ExpressionRef operator*(ExpressionRef f, ExpressionRef g);
ExpressionRef operator/(ExpressionRef f, ExpressionRef g);

// Functions
// TODO: Add all function supported by Jet.
ExpressionRef sin(ExpressionRef f);
ExpressionRef cos(ExpressionRef f);
ExpressionRef sqrt(ExpressionRef f);
ExpressionRef exp(ExpressionRef f);
ExpressionRef log(ExpressionRef f);
ExpressionRef pow(ExpressionRef f, ExpressionRef g);
ExpressionRef floor(ExpressionRef f);

// @brief A reference to a comparison expressions.
//
// This additonal type is required, so that we can detect invalid conditions
// during compile time. For example, the following should create a compile time
// error:
//
// ExpressionRef a(5);
// CERES_IF(a){           // Error: Invalid conversion
// ...
//
// Aollowing will work:
//
// ExpressionRef a(5), b(7);
// ComparisonExpressionRef c = a < b;
// CERES_IF(c){
// ...
struct ComparisonExpressionRef {
  ExpressionRef id;
  explicit ComparisonExpressionRef(ExpressionRef id) : id(id) {}
};

ExpressionRef Ternary(ComparisonExpressionRef c,
                      ExpressionRef a,
                      ExpressionRef b);

// Comparison operators
ComparisonExpressionRef operator<(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator<=(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator>(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator>=(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator==(ExpressionRef a, ExpressionRef b);
ComparisonExpressionRef operator!=(ExpressionRef a, ExpressionRef b);

// Logical Operators
ComparisonExpressionRef operator&&(ComparisonExpressionRef a,
                                   ComparisonExpressionRef b);
ComparisonExpressionRef operator||(ComparisonExpressionRef a,
                                   ComparisonExpressionRef b);

// This struct is used to mark numbers which are constant over
// multiple invocations but can differ between instances.
template <typename T>
struct RuntimeConstant {
  using ReturnType = T;
  static inline ReturnType Get(double v, const char* name) { return v; }
};

template <typename G, int N>
struct RuntimeConstant<Jet<G, N>> {
  using ReturnType = Jet<G, N>;
  static inline Jet<G, N> Get(double v, const char* name) {
    return Jet<G, N>(v);
  }
};

// This wrapper functions is here solve the cyclic dependency by not having to
// include "Expression.h" in this file.
ExpressionRef GetRuntimeConstant(const char* name);

template <int N>
struct RuntimeConstant<Jet<ExpressionRef, N>> {
  using ReturnType = Jet<ExpressionRef, N>;
  static inline ReturnType Get(double v, const char* name) {
    // Note: The scalar value of v will be thrown away, because we don't need it
    // during code generation.
    (void)v;
    return Jet<ExpressionRef, N>(GetRuntimeConstant(name));
  }
};

template <typename T>
inline typename RuntimeConstant<T>::ReturnType MakeRuntimeConstant(
    double v, const char* name) {
  return RuntimeConstant<T>::Get(v, name);
}

#define CERES_EXPRESSION_RUNTIME_CONSTANT(_v) \
  ceres::internal::MakeRuntimeConstant<T>(_v, #_v)
}  // namespace internal

// See jet.h for more info on this type.
template <>
struct ComparisonReturnType<internal::ExpressionRef> {
  using type = internal::ComparisonExpressionRef;
};

}  // namespace ceres
#endif
