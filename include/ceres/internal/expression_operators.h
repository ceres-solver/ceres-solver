
// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
// For a basic explanation how ceres::Jet and dual numbers work see jet.h.
// For a complete overview of the code generation see autodiff_codegen.h
//

#ifndef CERES_PUBLIC_EXPRESSION_OPERATORS_H_
#define CERES_PUBLIC_EXPRESSION_OPERATORS_H_

#include "Eigen/Core"
#include "ceres/internal/expression.h"
#include "ceres/jet.h"

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>

namespace ceres {

template <>
struct ComparisonReturnType<Expression> {
  using type = ComparisonExpression;
};

inline Expression operator-(const Expression& f) {
  return Expression::codeFactory->UnaryMinusExpr(f);
}

inline Expression operator+(const Expression& f, const Expression& g) {
  return Expression::codeFactory->BinaryExpr(ExpressionType::PLUS, f, g);
}
inline Expression operator-(const Expression& f, const Expression& g) {
  return Expression::codeFactory->BinaryExpr(ExpressionType::MINUS, f, g);
}

inline Expression operator*(const Expression& f, const Expression& g) {
  return Expression::codeFactory->BinaryExpr(ExpressionType::MULT, f, g);
}

inline Expression operator/(const Expression& f, const Expression& g) {
  return Expression::codeFactory->BinaryExpr(ExpressionType::DIV, f, g);
}

inline Expression sin(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("sin", {f});
}

inline Expression cos(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("cos", {f});
}

inline Expression sqrt(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("sqrt", {f});
}
inline Expression exp(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("exp", {f});
}
inline Expression log(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("log", {f});
}
inline Expression pow(const Expression& f, const Expression& g) {
  return Expression::codeFactory->FunctionExpr("pow", {f, g});
}
inline Expression floor(const Expression& f) {
  return Expression::codeFactory->FunctionExpr("floor", {f});
}
inline Expression PHI(ComparisonExpression c,
                      const Expression& a,
                      const Expression& b) {
  //  return Expression::codeFactory->FunctionExpr("ceres::PHI", {c.id, a, b});
  return Expression::codeFactory->Phi(c.id, a, b);
}

inline Expression::Expression(double constant) {
  id = Expression::codeFactory->ConstantExpr(constant).id;
}

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(op)          \
  inline ComparisonExpression operator op(const Expression& a,   \
                                          const Expression& b) { \
    return ComparisonExpression(                                 \
        Expression::codeFactory->BinaryCompare(#op, a, b));      \
  }

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR2(op)                   \
  inline ComparisonExpression operator op(const ComparisonExpression& a,   \
                                          const ComparisonExpression& b) { \
    return ComparisonExpression(                                           \
        Expression::codeFactory->BinaryCompare(#op, a.id, b.id));          \
  }

CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<)    // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<=)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>)    // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>=)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(==)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(!=)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(&&)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(||)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR2(&&)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR2(||)  // NOLINT

// The Phi-implementation of pow
template <int N>
inline Jet<Expression, N> pow(const Jet<Expression, N>& f,
                              const Jet<Expression, N>& g) {
  using T = Expression;
  using J = Jet<Expression, N>;

  // The 3 possible conditions
  auto c1 = (f.a == T(0)) && (g.a >= T(1));
  auto c2 = g.a > 1;
  auto c3 = f.a < 0 && g.a == floor(g.a);

  // The 4 results
  J r1 = J(T(0.0));
  J r2 = f;

  T const tmp = g.a * pow(f.a, g.a - T(1.0));
  J r3(pow(f.a, g.a), tmp * f.v);
  for (int i = 0; i < N; i++) {
    // todo: NAN is currently not supported, because it will break a few
    // comparisons
    r3.v[i] = PHI(g.v[i] != T(0.0), r3.v[i], T(9435739457));
  }

  // Handle the remaining cases. For cases 4,5,6,9 we allow the log() function
  // to generate -HUGE_VAL or NaN, since those cases result in a nonfinite
  // derivative.
  T const tmp1 = pow(f.a, g.a);
  T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  T const tmp3 = tmp1 * log(f.a);
  J r4(tmp1, tmp2 * f.v + tmp3 * g.v);

  return PHI(c1, PHI(c2, r1, r2), PHI(c3, r3, r4));
}

// This struct is used to mark number which are constant over
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
struct ExternalConstant<Jet<Expression, N>> {
  using ReturnType = Jet<Expression, N>;
  static inline ReturnType get(double v, const char* name) {
    ReturnType j(v);
    (*Expression::codeFactory)(j.a).externalConstant = true;
    (*Expression::codeFactory)(j.a).functionName = name;
    return j;
  }
};

template <typename T>
inline typename ExternalConstant<T>::ReturnType make_externalConstant(
    double v, const char* name) {
  return ExternalConstant<T>::get(v, name);
}

#define CERES_EXTERNAL_CONSTANT(_v) ceres::make_externalConstant<T>(_v, #_v)
}  // namespace ceres

#endif  // CERES_PUBLIC_JET_H_
