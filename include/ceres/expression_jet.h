
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

  // Default-construct "a" because otherwise this can lead to false errors about
  // uninitialized uses when other classes relying on default constructed T
  // (where T is a Jet<T, N>). This usually only happens in opt mode. Note that
  // the C++ standard mandates that e.g. default constructed doubles are
  // initialized to 0.0; see sections 8.5 of the C++03 standard.
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

  // Compound operators
  // make all current expressions into temporaries
  // and add new ones referencing the olds
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

  ExpressionId a;
  std::array<ExpressionId, N> v;
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
  h.addTmp(t1);
  h.addTmp(t2);

  h.v = f.factory.BinaryExpr(BINARY_PLUS, t1, t2);

  //    h.v = f.a * g.v + f.v * g.a;
  return h;
}

// Binary /
template <typename T, int N>
inline ExpressionJet<T, N> operator/(const ExpressionJet<T, N>& f,
                                     const ExpressionJet<T, N>& g) {
  ExpressionJet<T, N> h;
  // This uses:
  //
  //   a + u   (a + u)(b - v)   (a + u)(b - v)
  //   ----- = -------------- = --------------
  //   b + v   (b + v)(b - v)        b^2
  //
  // which holds because v*v = 0.
  //    h.a = f.a / g.a;
  h.a = f.factory.BinaryExpr(BINARY_DIV, f.a, g.a);

  auto t1 = f.factory.BinaryExpr(BINARY_MULT, h.a, g.v);
  auto t2 = f.factory.BinaryExpr(BINARY_MINUS, f.v, t1);
  h.v = f.factory.BinaryExpr(BINARY_DIV, t2, g.a);
  //    h.v = (f.v - f.a / g.a * g.v) / g.a;

  return h;
}

// Unary +
template <typename T, int N>
inline ExpressionJet<T, N> const& operator+(const ExpressionJet<T, N>& f) {
  return f;
}

// TODO(keir): Try adding __attribute__((always_inline)) to these functions to
// see if it causes a performance increase.

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

inline double PHI(bool c, double a, double b) { return c ? a : b; }

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

template <typename T, int N>
inline Jet<T, N> PHI(bool c, const Jet<T, N>& f, const Jet<T, N>& g) {
  return c ? f : g;
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

  //    g.v = g.a * f.v;
  g.v = f.factory.BinaryExpr(BINARY_MULT, g.a, f.v);
  return g;
}

// log(a + h) ~= log(a) + h / a
template <typename T, int N>
inline ExpressionJet<T, N> log(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  //    g.a = log(f.a);
  g.a = f.factory.FunctionExpr("log", {f.a});
  //    g.v = f.v / f.a;
  g.v = f.factory.BinaryExpr(BINARY_DIV, f.v, f.a);
  return g;
}

// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
template <typename T, int N>
inline ExpressionJet<T, N> sqrt(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  //    g.a = sqrt(f.a);
  g.a = f.factory.FunctionExpr("sqrt", {f.a});
  //    g.v = f.v / (T(2.0) * g.a);
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
  //    g.a     = cos(f.a);
  g.a = f.factory.FunctionExpr("cos", {f.a});

  //    T sin_a = sin(f.a);
  auto t1 = f.factory.FunctionExpr("sin", {f.a});

  //    g.v = -sin_a * f.v;
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
  //    g.a = acos(f.a);
  g.a = f.factory.FunctionExpr("acos", {f.a});

  // g.v = -T(1.0) / sqrt(T(1.0) - f.a * f.a) * f.v;

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
  //    g.a     = sin(f.a);

  g.a = f.factory.FunctionExpr("sin", {f.a});
  //    T cos_a = cos(f.a);
  auto t1 = f.factory.FunctionExpr("cos", {f.a});
  g.v = f.factory.BinaryExpr(BINARY_MULT, t1, f.v);

  g.addTmp(t1);
  //    g.v     = cos_a * f.v;
  return g;
}

// asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h
template <typename T, int N>
inline ExpressionJet<T, N> asin(const ExpressionJet<T, N>& f) {
  ExpressionJet<T, N> g;
  //    g.a = asin(f.a);
  g.a = f.factory.FunctionExpr("asin", {f.a});

  //    g.v = T(1.0) / sqrt(T(1.0) - f.a * f.a) * f.v;
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

}  // namespace ceres

#endif  // CERES_PUBLIC_JET_H_
