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
#include "glog/logging.h"

namespace ceres {
namespace internal {

ExpressionRef ExpressionRef::Create(ExpressionId id) {
  ExpressionRef ref;
  ref.id = id;
  return ref;
}

ExpressionRef::ExpressionRef(double compile_time_constant) {
  id = Expression::CreateCompileTimeConstant(compile_time_constant);
}

ExpressionRef::ExpressionRef(const ExpressionRef& other) { *this = other; }

ExpressionRef& ExpressionRef::operator=(const ExpressionRef& other) {
  // Assigning an uninitialized variable to another variable is an error.
  CHECK(other.IsInitialized()) << "Uninitialized Assignment.";

  if (IsInitialized()) {
    // Create assignment from other -> this
    Expression::CreateAssignment(id, other.id);
  } else {
    // Special case: "this" expressionref is invalid
    //    -> Skip assignment
    //    -> Let "this" point to the same variable as other
    //    id = Expression::CreateAssignment(id, other.id);
    id = other.id;
  }
  return *this;
}

// Compound operators
ExpressionRef& ExpressionRef::operator+=(const ExpressionRef& x) {
  *this = *this + x;
  return *this;
}

ExpressionRef& ExpressionRef::operator-=(const ExpressionRef& x) {
  *this = *this - x;
  return *this;
}

ExpressionRef& ExpressionRef::operator*=(const ExpressionRef& x) {
  *this = *this * x;
  return *this;
}

ExpressionRef& ExpressionRef::operator/=(const ExpressionRef& x) {
  *this = *this / x;
  return *this;
}

// Arith. Operators
ExpressionRef operator-(const ExpressionRef& x) {
  return ExpressionRef::Create(Expression::CreateUnaryArithmetic("-", x.id));
}

ExpressionRef operator+(const ExpressionRef& x) {
  return ExpressionRef::Create(Expression::CreateUnaryArithmetic("+", x.id));
}

ExpressionRef operator+(const ExpressionRef& x, const ExpressionRef& y) {
  return ExpressionRef::Create(
      Expression::CreateBinaryArithmetic("+", x.id, y.id));
}

ExpressionRef operator-(const ExpressionRef& x, const ExpressionRef& y) {
  return ExpressionRef::Create(
      Expression::CreateBinaryArithmetic("-", x.id, y.id));
}

ExpressionRef operator/(const ExpressionRef& x, const ExpressionRef& y) {
  return ExpressionRef::Create(
      Expression::CreateBinaryArithmetic("/", x.id, y.id));
}

ExpressionRef operator*(const ExpressionRef& x, const ExpressionRef& y) {
  return ExpressionRef::Create(
      Expression::CreateBinaryArithmetic("*", x.id, y.id));
}

ExpressionRef Ternary(const ComparisonExpressionRef& c,
                      const ExpressionRef& x,
                      const ExpressionRef& y) {
  return ExpressionRef::Create(
      Expression::CreateFunctionCall("Ternary", {c.id, x.id, y.id}));
}

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(op)         \
  ComparisonExpressionRef operator op(const ExpressionRef& x,   \
                                      const ExpressionRef& y) { \
    return ComparisonExpressionRef(ExpressionRef::Create(       \
        Expression::CreateBinaryCompare(#op, x.id, y.id)));     \
  }

#define CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(op)                      \
  ComparisonExpressionRef operator op(const ComparisonExpressionRef& x,   \
                                      const ComparisonExpressionRef& y) { \
    return ComparisonExpressionRef(ExpressionRef::Create(                 \
        Expression::CreateBinaryCompare(#op, x.id, y.id)));               \
  }

CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<)
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(<=)
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>)
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(>=)
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(==)
CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(!=)
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(&&)
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(||)
#undef CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR
#undef CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR

ComparisonExpressionRef operator!(const ComparisonExpressionRef& x) {
  return ComparisonExpressionRef(
      ExpressionRef::Create(Expression::CreateLogicalNegation(x.id)));
}

}  // namespace internal
}  // namespace ceres
