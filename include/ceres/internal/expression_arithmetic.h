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
// This file contains all the required operators and function to use
// 'ExpressionRef' in mathematical functions. For example:
//
// ExpressionRef a(5);
// ExpressionRef b(7);
// ExpressionRef result = a * b + sin(a) - ternary(a < b, a, b);
//
// Since all "standard" operators are defined, we can use ExpressionRef also in
// ceres::Jet to generate expressions for the derivatives.
//
// Jet<ExpressionRef,2> J1(2.1);
// Jet<ExpressionRef,2> J2(2.1);
// auto tmp = J1 + J2;

#ifndef CERES_PUBLIC_EXPRESSION_ARITHMETIC_H_
#define CERES_PUBLIC_EXPRESSION_ARITHMETIC_H_

#include "Eigen/Core"
#include "ceres/jet.h"
#include "expression.h"

namespace ceres {

template <>
struct ComparisonReturnType<internal::ExpressionRef> {
  using type = internal::ComparisonExpressionRef;
};

namespace internal {

inline ExpressionRef::ExpressionRef(double constant) {
  (*this) = Expression::MakeConstant(constant);
}

inline ExpressionRef operator-(ExpressionRef f) {
  return Expression::MakeUnaryMinus(f);
}

inline ExpressionRef operator+(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::PLUS, f, g);
}

inline ExpressionRef operator-(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::MINUS, f, g);
}

inline ExpressionRef operator*(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::MULTIPLICATION, f, g);
}

inline ExpressionRef operator/(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::DIVISION, f, g);
}

// Compound operators
inline ExpressionRef& ExpressionRef::operator+=(ExpressionRef y) {
  *this = *this + y;
  return *this;
}

inline ExpressionRef& ExpressionRef::operator-=(ExpressionRef y) {
  *this = *this - y;
  return *this;
}

inline ExpressionRef& ExpressionRef::operator*=(ExpressionRef y) {
  *this = *this * y;
  return *this;
}

inline ExpressionRef& ExpressionRef::operator/=(ExpressionRef y) {
  *this = *this / y;
  return *this;
}

// Functions
inline ExpressionRef sin(ExpressionRef f) {
  return Expression::MakeFunction("sin", {f});
}

inline ExpressionRef cos(ExpressionRef f) {
  return Expression::MakeFunction("cos", {f});
}

inline ExpressionRef sqrt(ExpressionRef f) {
  return Expression::MakeFunction("sqrt", {f});
}

inline ExpressionRef exp(ExpressionRef f) {
  return Expression::MakeFunction("exp", {f});
}

inline ExpressionRef log(ExpressionRef f) {
  return Expression::MakeFunction("log", {f});
}

inline ExpressionRef pow(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeFunction("pow", {f, g});
}

inline ExpressionRef floor(ExpressionRef f) {
  return Expression::MakeFunction("floor", {f});
}

inline ExpressionRef ternary(ComparisonExpressionRef c,
                             ExpressionRef a,
                             ExpressionRef b) {
  return Expression::MakeTernary(c, a, b);
}

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(op)                       \
  inline ComparisonExpressionRef operator op(ExpressionRef a,                 \
                                             ExpressionRef b) {               \
    return ComparisonExpressionRef(Expression::MakeBinaryCompare(#op, a, b)); \
  }

#define CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(op)                      \
  inline ComparisonExpressionRef operator op(ComparisonExpressionRef a,   \
                                             ComparisonExpressionRef b) { \
    return ComparisonExpressionRef(                                       \
        Expression::MakeBinaryCompare(#op, a.id, b.id));                  \
  }

CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<=)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>=)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(==)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(!=)  // NOLINT
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(&&)     // NOLINT
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(||)     // NOLINT
#undef CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR
#undef CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR

// This struct is used to mark numbers which are constant over
// multiple invocations but can differ between instances.
template <typename T>
struct ExternalConstant {
  using ReturnType = T;
  static inline ReturnType get(double v, const char* name) { return v; }
};

template <typename G, int N>
struct ExternalConstant<Jet<G, N>> {
  using ReturnType = Jet<G, N>;
  static inline Jet<G, N> get(double v, const char* name) {
    return Jet<G, N>(v);
  }
};

template <int N>
struct ExternalConstant<Jet<ExpressionRef, N>> {
  using ReturnType = Jet<ExpressionRef, N>;
  static inline ReturnType get(double v, const char* name) {
    // Note: The scalar value of v will be thrown away, because we don't need it
    // during code generation.
    (void)v;
    return Jet<ExpressionRef, N>(Expression::MakeExternalConstant(name));
  }
};

template <typename T>
inline typename ExternalConstant<T>::ReturnType make_externalConstant(
    double v, const char* name) {
  return ExternalConstant<T>::get(v, name);
}

#define CERES_EXPRESSION_EXTERNAL_CONSTANT(_v) \
  ceres::internal::make_externalConstant<T>(_v, #_v)
}  // namespace internal
}  // namespace ceres

#endif
