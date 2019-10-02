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

#include "ceres/internal/expression_ref.h"
#include "assert.h"
#include "ceres/internal/expression.h"

namespace ceres {
namespace internal {

std::string ExpressionRef::ToString() { return std::to_string(id); }

ExpressionRef::ExpressionRef(double constant) {
  (*this) = Expression::MakeConstant(constant);
}

// Compound operators
ExpressionRef& ExpressionRef::operator+=(ExpressionRef y) {
  *this = *this + y;
  return *this;
}

ExpressionRef& ExpressionRef::operator-=(ExpressionRef y) {
  *this = *this - y;
  return *this;
}

ExpressionRef& ExpressionRef::operator*=(ExpressionRef y) {
  *this = *this * y;
  return *this;
}

ExpressionRef& ExpressionRef::operator/=(ExpressionRef y) {
  *this = *this / y;
  return *this;
}

// Arith. Operators
ExpressionRef operator-(ExpressionRef f) {
  return Expression::MakeUnaryArithmetic(f, ExpressionType::UNARY_MINUS);
}

ExpressionRef operator+(ExpressionRef f) {
  return Expression::MakeUnaryArithmetic(f, ExpressionType::UNARY_PLUS);
}

ExpressionRef operator+(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::PLUS, f, g);
}

ExpressionRef operator-(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::MINUS, f, g);
}

ExpressionRef operator*(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::MULTIPLICATION, f, g);
}

ExpressionRef operator/(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeBinaryArithmetic(ExpressionType::DIVISION, f, g);
}

// Functions
ExpressionRef sin(ExpressionRef f) {
  return Expression::MakeFunctionCall("sin", {f});
}

ExpressionRef cos(ExpressionRef f) {
  return Expression::MakeFunctionCall("cos", {f});
}

ExpressionRef sqrt(ExpressionRef f) {
  return Expression::MakeFunctionCall("sqrt", {f});
}

ExpressionRef exp(ExpressionRef f) {
  return Expression::MakeFunctionCall("exp", {f});
}

ExpressionRef log(ExpressionRef f) {
  return Expression::MakeFunctionCall("log", {f});
}

ExpressionRef pow(ExpressionRef f, ExpressionRef g) {
  return Expression::MakeFunctionCall("pow", {f, g});
}

ExpressionRef floor(ExpressionRef f) {
  return Expression::MakeFunctionCall("floor", {f});
}

ExpressionRef Ternary(ComparisonExpressionRef c,
                      ExpressionRef a,
                      ExpressionRef b) {
  return Expression::MakeTernary(c, a, b);
}

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(op)                       \
  ComparisonExpressionRef operator op(ExpressionRef a, ExpressionRef b) {     \
    return ComparisonExpressionRef(Expression::MakeBinaryCompare(#op, a, b)); \
  }

#define CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(op)               \
  ComparisonExpressionRef operator op(ComparisonExpressionRef a,   \
                                      ComparisonExpressionRef b) { \
    return ComparisonExpressionRef(                                \
        Expression::MakeBinaryCompare(#op, a.id, b.id));           \
  }

CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<=)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>)   // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>=)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(==)  // NOLINT
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(!=)  // NOLINT
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(&&)     // NOLINT
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(||)

ExpressionRef GetRuntimeConstant(const char* name) {
  return Expression::MakeRuntimeConstant(name);
}

// NOLINT
#undef CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR
#undef CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR

}  // namespace internal
}  // namespace ceres
