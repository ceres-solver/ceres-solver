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

#include "ceres/internal/expression.h"
#include <algorithm>

#include "ceres/internal/expression_graph.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

// Wrapper for ExpressionGraph::CreateArithmeticExpression, which checks if a
// graph is currently active. See that function for an explanation.
static Expression& MakeArithmeticExpression(
    ExpressionType type, ExpressionId lhs_id = kInvalidExpressionId) {
  auto pool = GetCurrentExpressionGraph();
  CHECK(pool)
      << "The ExpressionGraph has to be created before using Expressions. This "
         "is achieved by calling ceres::StartRecordingExpressions.";
  return pool->CreateArithmeticExpression(type, lhs_id);
}

// Wrapper for ExpressionGraph::CreateControlExpression.
static Expression& MakeControlExpression(ExpressionType type) {
  auto pool = GetCurrentExpressionGraph();
  CHECK(pool)
      << "The ExpressionGraph has to be created before using Expressions. This "
         "is achieved by calling ceres::StartRecordingExpressions.";
  return pool->CreateControlExpression(type);
}

ExpressionId Expression::CreateCompileTimeConstant(double v) {
  auto& expr = MakeArithmeticExpression(ExpressionType::COMPILE_TIME_CONSTANT);
  expr.value_ = v;
  return expr.lhs_id_;
}

ExpressionId Expression::CreateInputAssignment(const std::string& name) {
  auto& expr = MakeArithmeticExpression(ExpressionType::INPUT_ASSIGNMENT);
  expr.name_ = name;
  return expr.lhs_id_;
}

ExpressionId Expression::CreateOutputAssignment(ExpressionId v,
                                                const std::string& name) {
  auto& expr = MakeArithmeticExpression(ExpressionType::OUTPUT_ASSIGNMENT);
  expr.arguments_.push_back(v);
  expr.name_ = name;
  return expr.lhs_id_;
}

ExpressionId Expression::CreateAssignment(ExpressionId dst, ExpressionId src) {
  auto& expr = MakeArithmeticExpression(ExpressionType::ASSIGNMENT, dst);

  expr.arguments_.push_back(src);
  return expr.lhs_id_;
}

ExpressionId Expression::CreateBinaryArithmetic(const std::string& op,
                                                ExpressionId l,
                                                ExpressionId r) {
  auto& expr = MakeArithmeticExpression(ExpressionType::BINARY_ARITHMETIC);
  expr.name_ = op;
  expr.arguments_.push_back(l);
  expr.arguments_.push_back(r);
  return expr.lhs_id_;
}

ExpressionId Expression::CreateUnaryArithmetic(const std::string& op,
                                               ExpressionId v) {
  auto& expr = MakeArithmeticExpression(ExpressionType::UNARY_ARITHMETIC);
  expr.name_ = op;
  expr.arguments_.push_back(v);
  return expr.lhs_id_;
}

ExpressionId Expression::CreateBinaryCompare(const std::string& name,
                                             ExpressionId l,
                                             ExpressionId r) {
  auto& expr = MakeArithmeticExpression(ExpressionType::BINARY_COMPARISON);
  expr.arguments_.push_back(l);
  expr.arguments_.push_back(r);
  expr.name_ = name;
  return expr.lhs_id_;
}

ExpressionId Expression::CreateLogicalNegation(ExpressionId v) {
  auto& expr = MakeArithmeticExpression(ExpressionType::LOGICAL_NEGATION);
  expr.arguments_.push_back(v);
  return expr.lhs_id_;
}

ExpressionId Expression::CreateFunctionCall(
    const std::string& name, const std::vector<ExpressionId>& params) {
  auto& expr = MakeArithmeticExpression(ExpressionType::FUNCTION_CALL);
  expr.arguments_ = params;
  expr.name_ = name;
  return expr.lhs_id_;
}

void Expression::CreateIf(ExpressionId condition) {
  auto& expr = MakeControlExpression(ExpressionType::IF);
  expr.arguments_.push_back(condition);
}

void Expression::CreateElse() { MakeControlExpression(ExpressionType::ELSE); }

void Expression::CreateEndIf() { MakeControlExpression(ExpressionType::ENDIF); }

Expression::Expression(ExpressionType type, ExpressionId id)
    : type_(type), lhs_id_(id) {}

bool Expression::IsArithmeticExpression() const { return HasValidLhs(); }

bool Expression::IsControlExpression() const { return !HasValidLhs(); }

bool Expression::IsReplaceableBy(const Expression& other) const {
  // Check everything except the id.
  return (type_ == other.type_ && name_ == other.name_ &&
          value_ == other.value_ && arguments_ == other.arguments_);
}

void Expression::Replace(const Expression& other) {
  if (other.lhs_id_ == lhs_id_) {
    return;
  }

  type_ = other.type_;
  arguments_ = other.arguments_;
  name_ = other.name_;
  value_ = other.value_;
}

bool Expression::DirectlyDependsOn(ExpressionId other) const {
  return (std::find(arguments_.begin(), arguments_.end(), other) !=
          arguments_.end());
}

bool Expression::IsCompileTimeConstantAndEqualTo(double constant) const {
  return type_ == ExpressionType::COMPILE_TIME_CONSTANT && value_ == constant;
}

void Expression::MakeNop() {
  type_ = ExpressionType::NOP;
  arguments_.clear();
}

bool Expression::operator==(const Expression& other) const {
  return type() == other.type() && name() == other.name() &&
         value() == other.value() && lhs_id() == other.lhs_id() &&
         arguments() == other.arguments();
}

bool Expression::IsSemanticallyEquivalentTo(const Expression& other) const {
  return type() == other.type() && name() == other.name() &&
         value() == other.value() &&
         arguments().size() == other.arguments().size();
}

}  // namespace internal
}  // namespace ceres
