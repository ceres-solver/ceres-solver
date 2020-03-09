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

#include "ceres/codegen/internal/expression_ref.h"

#include "ceres/codegen/internal/expression_graph.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

ExpressionRef AddExpressionToGraph(const Expression& expression) {
  return ExpressionRef(expression);
}

ExpressionRef::ExpressionRef(double compile_time_constant)
    : ExpressionRef(
          Expression::CreateCompileTimeConstant(compile_time_constant)) {}

ExpressionRef::ExpressionRef(const Expression& expression) {
  ExpressionGraph* graph = GetCurrentExpressionGraph();
  CHECK(graph)
      << "The ExpressionGraph has to be created before using Expressions. This "
         "is achieved by calling ceres::StartRecordingExpressions.";
  id = graph->InsertBack(expression);
}

ExpressionRef::ExpressionRef(const ExpressionRef& other) { *this = other; }

ExpressionRef& ExpressionRef::operator=(const ExpressionRef& other) {
  // Assigning an uninitialized variable to another variable is an error.
  CHECK(other.IsInitialized()) << "Uninitialized Assignment.";
  if (IsInitialized()) {
    // Create assignment from other -> this
    AddExpressionToGraph(Expression::CreateAssignment(this->id, other.id));
  } else {
    // Create a new variable and
    // Create assignment from other -> this
    // Passing kInvalidExpressionId to CreateAssignment generates a new
    // variable name which we store in the id.
    id = AddExpressionToGraph(
             Expression::CreateAssignment(kInvalidExpressionId, other.id))
             .id;
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
  return AddExpressionToGraph(Expression::CreateUnaryArithmetic("-", x.id));
}

ExpressionRef operator+(const ExpressionRef& x) {
  return AddExpressionToGraph(Expression::CreateUnaryArithmetic("+", x.id));
}

ExpressionRef operator+(const ExpressionRef& x, const ExpressionRef& y) {
  return AddExpressionToGraph(
      Expression::CreateBinaryArithmetic("+", x.id, y.id));
}

ExpressionRef operator-(const ExpressionRef& x, const ExpressionRef& y) {
  return AddExpressionToGraph(
      Expression::CreateBinaryArithmetic("-", x.id, y.id));
}

ExpressionRef operator/(const ExpressionRef& x, const ExpressionRef& y) {
  return AddExpressionToGraph(
      Expression::CreateBinaryArithmetic("/", x.id, y.id));
}

ExpressionRef operator*(const ExpressionRef& x, const ExpressionRef& y) {
  return AddExpressionToGraph(
      Expression::CreateBinaryArithmetic("*", x.id, y.id));
}

ExpressionRef Ternary(const ComparisonExpressionRef& c,
                      const ExpressionRef& x,
                      const ExpressionRef& y) {
  return AddExpressionToGraph(Expression::CreateScalarFunctionCall(
      "ceres::Ternary", {c.id, x.id, y.id}));
}

#define CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR(op)         \
  ComparisonExpressionRef operator op(const ExpressionRef& x,   \
                                      const ExpressionRef& y) { \
    return ComparisonExpressionRef(AddExpressionToGraph(        \
        Expression::CreateBinaryCompare(#op, x.id, y.id)));     \
  }

#define CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(op)                      \
  ComparisonExpressionRef operator op(const ComparisonExpressionRef& x,   \
                                      const ComparisonExpressionRef& y) { \
    return ComparisonExpressionRef(AddExpressionToGraph(                  \
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
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(&)
CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR(|)
#undef CERES_DEFINE_EXPRESSION_COMPARISON_OPERATOR
#undef CERES_DEFINE_EXPRESSION_LOGICAL_OPERATOR

ComparisonExpressionRef operator!(const ComparisonExpressionRef& x) {
  return ComparisonExpressionRef(
      AddExpressionToGraph(Expression::CreateLogicalNegation(x.id)));
}

}  // namespace internal
}  // namespace ceres
