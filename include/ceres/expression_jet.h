
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

#ifndef CERES_PUBLIC_EXPRESSIONJET_H_
#define CERES_PUBLIC_EXPRESSIONJET_H_

#include "Eigen/Core"
#include "ceres/jet.h"
#include "expressions.h"

#include <cmath>
#include <iosfwd>
#include <iostream>
#include <string>

namespace ceres {

template <typename T, int N>
struct ExpressionJet {
  using Base = ExpressionJet<T, N>;
  // we need it as a global variable, because we cannot pass it as parameters to
  // the operators
  static CodeFactory factory;

  enum { DIMENSION = N };

  ExpressionJet() : ExpressionJet(T(0)) {}

  // Constructor from scalar: a + 0.
  explicit ExpressionJet(const T& value) {
    a = factory.ConstantExpr(value);
    for (int i = 0; i < N; ++i) {
      v[i] = factory.ConstantExpr(0);
    }
  }

  // Constructor from scalar plus variable: a + t_i.
  ExpressionJet(const T& value, int k) {
    a = factory.ConstantExpr(value);
    for (int i = 0; i < N; ++i) {
      if (i == k)
        v[i] = factory.ConstantExpr(1);
      else
        v[i] = factory.ConstantExpr(0);
    }
  }

  ExpressionJet<T, N>& operator+=(const ExpressionJet<T, N>& y) {
    ExpressionJet<T, N> tmp = *this + y;
    *this = tmp;
    return *this;
  }

  ExpressionJet<T, N>& operator-=(const ExpressionJet<T, N>& y) {
    ExpressionJet<T, N> tmp = *this - y;
    *this = tmp;
    return *this;
  }
  ExpressionJet<T, N>& operator/=(const ExpressionJet<T, N>& y) {
    ExpressionJet<T, N> tmp = *this / y;
    *this = tmp;
    return *this;
  }
  ExpressionJet<T, N>& operator*=(const ExpressionJet<T, N>& y) {
    ExpressionJet<T, N> tmp = *this * y;
    *this = tmp;
    return *this;
  }

  // Instead of the values of the derivatives, we keep track of the expression
  // that produce the values.
  ExpressionId a;
  std::array<ExpressionId, N> v;

  // Sub expressions that are required to compute the expressions in a and v.
  std::vector<ExpressionId> tmp;

  void addTmp(ExpressionId expr) { tmp.push_back(expr); }
  template <unsigned long K>
  void addTmp(const std::array<ExpressionId, K>& l) {
    for (int i = 0; i < K; ++i) {
      tmp.push_back(l[i]);
    }
  }
};

template <typename T, int N>
CodeFactory ExpressionJet<T, N>::factory;

// Binary -
template <typename T, int N>
inline ExpressionJet<T, N> operator-(const ExpressionJet<T, N>& f,
                                     const ExpressionJet<T, N>& g) {
  ExpressionJet<T, N> h;
  h.a = f.factory.BinaryExpr(BINARY_MINUS, f.a, g.a);
  h.v = f.factory.BinaryExpr(BINARY_MINUS, f.v, g.v);
  return h;
}

// Binary +
template <typename T, int N>
inline ExpressionJet<T, N> operator+(const ExpressionJet<T, N>& f,
                                     const ExpressionJet<T, N>& g) {
  ExpressionJet<T, N> h;
  h.a = f.factory.BinaryExpr(BINARY_PLUS, f.a, g.a);
  h.v = f.factory.BinaryExpr(BINARY_PLUS, f.v, g.v);
  return h;
}
// Binary *
template <typename T, int N>
inline ExpressionJet<T, N> operator*(const ExpressionJet<T, N>& f,
                                     const ExpressionJet<T, N>& g) {
  ExpressionJet<T, N> h;
  h.a = f.factory.BinaryExpr(BINARY_MULT, f.a, g.a);
  auto t1 = f.factory.BinaryExpr(BINARY_MULT, f.a, g.v);
  auto t2 = f.factory.BinaryExpr(BINARY_MULT, f.v, g.a);
  h.v = f.factory.BinaryExpr(BINARY_PLUS, t1, t2);
  h.addTmp(t1);
  h.addTmp(t2);
  return h;
}

// Binary /
template <typename T, int N>
inline ExpressionJet<T, N> operator/(const ExpressionJet<T, N>& f,
                                     const ExpressionJet<T, N>& g) {
  ExpressionJet<T, N> h;

  // Generate a single division 1/g.a and then multiply with the fraction
  // instead.

  auto t3 = f.factory.ConstantExpr(1);
  auto t4 = f.factory.BinaryExpr(BINARY_DIV, t3, g.a);

  //  h.a = f.factory.BinaryExpr(BINARY_DIV, f.a, g.a);
  h.a = f.factory.BinaryExpr(BINARY_MULT, f.a, t4);

  auto t1 = f.factory.BinaryExpr(BINARY_MULT, h.a, g.v);
  auto t2 = f.factory.BinaryExpr(BINARY_MINUS, f.v, t1);

  //  h.v = f.factory.BinaryExpr(BINARY_DIV, t2, g.a);
  h.v = f.factory.BinaryExpr(BINARY_MULT, t2, t4);
  h.addTmp(t1);
  h.addTmp(t2);
  h.addTmp(t3);
  h.addTmp(t4);
  return h;
}

// TODO: the operators between scalars and jets

// Unary +
template <typename T, int N>
inline ExpressionJet<T, N> const& operator+(const ExpressionJet<T, N>& f) {
  return f;
}

// Unary -
template <typename T, int N>
inline ExpressionJet<T, N> operator-(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.UnaryMinusExpr(f.a);
  for (int i = 0; i < N; ++i) {
    g.v[i] = f.factory.UnaryMinusExpr(f.v[i]);
  }
  return g;
}

template <typename T, int N>
inline ExpressionJet<T, N> PHI(ComparisonExpressionId c,
                               const ExpressionJet<T, N>& a,
                               const ExpressionJet<T, N>& b) {
  ExpressionJet<T, N> g;
  g.a = a.factory.FunctionExpr("ceres::PHI", {c.id, a.a, b.a});
  for (int i = 0; i < N; ++i) {
    g.v[i] = a.factory.FunctionExpr("ceres::PHI", {c.id, a.v[i], b.v[i]});
  }
  return g;
}

#define CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(op)                  \
  template <typename T, int N>                                              \
  inline ComparisonExpressionId operator op(const ExpressionJet<T, N>& f,   \
                                            const ExpressionJet<T, N>& g) { \
    auto ex = f.factory.BinaryCompare(#op, f.a, g.a);                       \
    ComparisonExpressionId ret(ex);                                         \
    ret.factory = &f.factory;                                               \
    return ret;                                                             \
  }

CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(<)   // NOLINT
CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(<=)  // NOLINT
CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(>)   // NOLINT
CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(>=)  // NOLINT
CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(==)  // NOLINT
CERES_DEFINE_EXPRESSIONJET_COMPARISON_OPERATOR(!=)  // NOLINT

#define CERES_DEFINE_EXPRESSIONJET_EXPRESSION_COMPARISON_OPERATOR(op)   \
  inline ComparisonExpressionId operator op(ComparisonExpressionId a,   \
                                            ComparisonExpressionId b) { \
    auto ex = a.factory->BinaryCompare(#op, a.id, b.id);                \
    ComparisonExpressionId ret(ex);                                     \
    ret.factory = a.factory;                                            \
    return ret;                                                         \
  }

CERES_DEFINE_EXPRESSIONJET_EXPRESSION_COMPARISON_OPERATOR(&&)  // NOLINT
CERES_DEFINE_EXPRESSIONJET_EXPRESSION_COMPARISON_OPERATOR(||)  // NOLINT

// exp(a + h) ~= exp(a) + exp(a) h
template <typename T, int N>
inline ExpressionJet<T, N> exp(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("exp", {f.a});
  g.v = f.factory.BinaryExpr(BINARY_MULT, g.a, f.v);
  return g;
}

// log(a + h) ~= log(a) + h / a
template <typename T, int N>
inline ExpressionJet<T, N> log(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("log", {f.a});
  g.v = f.factory.BinaryExpr(BINARY_DIV, f.v, f.a);
  return g;
}

// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
template <typename T, int N>
inline ExpressionJet<T, N> sqrt(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("sqrt", {f.a});
  auto t1 = f.factory.ConstantExpr(2);
  auto t2 = f.factory.BinaryExpr(BINARY_MULT, t1, g.a);
  g.v = f.factory.BinaryExpr(BINARY_DIV, f.v, t2);
  g.addTmp(t1);
  g.addTmp(t2);
  return g;
}

// cos(a + h) ~= cos(a) - sin(a) h
template <typename T, int N>
inline ExpressionJet<T, N> cos(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("cos", {f.a});
  auto t1 = f.factory.FunctionExpr("sin", {f.a});
  auto t2 = f.factory.UnaryMinusExpr(t1);
  g.v = f.factory.BinaryExpr(BINARY_MULT, t2, f.v);
  g.addTmp(t1);
  g.addTmp(t2);
  return g;
}

// acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h
template <typename T, int N>
inline ExpressionJet<T, N> acos(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("acos", {f.a});
  auto t1 = f.factory.ConstantExpr(1);
  auto t2 = f.factory.BinaryExpr(BINARY_MULT, f.a, f.a);
  auto t3 = f.factory.BinaryExpr(BINARY_MINUS, t1, t2);
  auto t4 = f.factory.FunctionExpr("sqrt", {t3});
  auto t5 = f.factory.ConstantExpr(-1);
  auto t6 = f.factory.BinaryExpr(BINARY_DIV, t5, t4);
  g.v = f.factory.BinaryExpr(BINARY_MULT, t6, f.v);
  g.addTmp(t1);
  g.addTmp(t2);
  g.addTmp(t3);
  g.addTmp(t4);
  g.addTmp(t5);
  g.addTmp(t6);
  return g;
}

// sin(a + h) ~= sin(a) + cos(a) h
template <typename T, int N>
inline ExpressionJet<T, N> sin(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("sin", {f.a});
  auto t1 = f.factory.FunctionExpr("cos", {f.a});
  g.v = f.factory.BinaryExpr(BINARY_MULT, t1, f.v);
  g.addTmp(t1);
  return g;
}

// asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h
template <typename T, int N>
inline ExpressionJet<T, N> asin(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  g.a = f.factory.FunctionExpr("asin", {f.a});
  auto t1 = f.factory.ConstantExpr(1);
  auto t2 = f.factory.BinaryExpr(BINARY_MULT, f.a, f.a);
  auto t3 = f.factory.BinaryExpr(BINARY_MINUS, t1, t2);
  auto t4 = f.factory.FunctionExpr("sqrt", {t3});
  auto t5 = f.factory.ConstantExpr(1);
  auto t6 = f.factory.BinaryExpr(BINARY_DIV, t5, t4);
  g.v = f.factory.BinaryExpr(BINARY_MULT, t6, f.v);
  g.addTmp(t1);
  g.addTmp(t2);
  g.addTmp(t3);
  g.addTmp(t4);
  g.addTmp(t5);
  g.addTmp(t6);
  return g;
}

template <typename T, int N>
inline ExpressionJet<T, N> pow(const ExpressionJet<T, N>& f, double e) {
  ExpressionJet<T, N> g;
  auto t1 = f.factory.ConstantExpr(-1);
  auto t2 = f.factory.ConstantExpr(e);
  auto t3 = f.factory.BinaryExpr(BINARY_PLUS, t1, t2);
  auto t4 = f.factory.FunctionExpr("pow", {f.a, t3});
  auto t5 = f.factory.BinaryExpr(BINARY_MULT, t2, t4);
  g.a = f.factory.FunctionExpr("pow", {f.a, t2});
  g.v = f.factory.BinaryExpr(BINARY_MULT, t5, f.v);
  g.addTmp(t1);
  g.addTmp(t2);
  g.addTmp(t3);
  g.addTmp(t4);
  g.addTmp(t5);
  return g;
}

template <typename T, int N>
inline ExpressionJet<T, N> pow(const ExpressionJet<T, N>& f,
                               const ExpressionJet<T, N>& g) {
  //  T const tmp1 = pow(f.a, g.a);
  //  T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  //  T const tmp3 = tmp1 * log(f.a);
  //  return Jet<T, N>(tmp1, tmp2 * f.v + tmp3 * g.v);

  ExpressionJet<T, N> r;

  //  T const tmp1 = pow(f.a, g.a);
  r.a = f.factory.FunctionExpr("pow", {f.a, g.a});

  // T const tmp2 = g.a * pow(f.a, g.a - T(1.0));
  auto t1 = f.factory.ConstantExpr(-1);
  auto t2 = f.factory.BinaryExpr(BINARY_PLUS, g.a, t1);
  auto t3 = f.factory.FunctionExpr("pow", {f.a, t2});
  auto t4 = f.factory.BinaryExpr(BINARY_MULT, g.a, t3);

  //  T const tmp3 = tmp1 * log(f.a);
  auto t5 = f.factory.FunctionExpr("log", {f.a});
  auto t6 = f.factory.BinaryExpr(BINARY_MULT, r.a, t5);

  //  return Jet<T, N>(tmp1, tmp2 * f.v + tmp3 * g.v);
  auto t7 = f.factory.BinaryExpr(BINARY_MULT, t4, f.v);
  auto t8 = f.factory.BinaryExpr(BINARY_MULT, t6, g.v);
  r.v = f.factory.BinaryExpr(BINARY_PLUS, t7, t8);

  return r;
}
// TODO: the remaining common functions from jet.h

// This struct is used to mark number which are constant over multiple
// invocations but can differ between instances.
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

template <typename G, int N>
struct ExternalConstant<ExpressionJet<G, N>> {
  using ReturnType = ExpressionJet<G, N>;
  static inline ExpressionJet<G, N> get(double v, const char* name) {
    ExpressionJet<G, N> j(v);

    j.factory(j.a).externalConstant = true;
    j.factory(j.a).functionName = name;

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
