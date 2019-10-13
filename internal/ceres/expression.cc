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

static Expression& MakeExpression(ExpressionType type) {
  auto pool = GetCurrentExpressionGraph();
  CHECK(pool)
      << "The ExpressionGraph has to be created before using Expressions. This "
         "is achieved by calling ceres::StartRecordingExpressions.";
  return pool->CreateExpression(type);
}

ExpressionId Expression::CreateCompileTimeConstant(double v) {
  auto& expr = MakeExpression(ExpressionType::COMPILE_TIME_CONSTANT);
  expr.value_ = v;
  return expr.id_;
}

ExpressionId Expression::CreateRuntimeConstant(const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::RUNTIME_CONSTANT);
  expr.name_ = name;
  return expr.id_;
}

ExpressionId Expression::CreateParameter(const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::PARAMETER);
  expr.name_ = name;
  return expr.id_;
}

ExpressionId Expression::CreateAssignment(ExpressionId v) {
  auto& expr = MakeExpression(ExpressionType::ASSIGNMENT);
  expr.arguments_.push_back(v);
  return expr.id_;
}

ExpressionId Expression::CreateUnaryArithmetic(ExpressionType type,
                                               ExpressionId v) {
  auto& expr = MakeExpression(type);
  expr.arguments_.push_back(v);
  return expr.id_;
}

ExpressionId Expression::CreateOutputAssignment(ExpressionId v,
                                                const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::OUTPUT_ASSIGNMENT);
  expr.arguments_.push_back(v);
  expr.name_ = name;
  return expr.id_;
}

ExpressionId Expression::CreateFunctionCall(
    const std::string& name, const std::vector<ExpressionId>& params) {
  auto& expr = MakeExpression(ExpressionType::FUNCTION_CALL);
  expr.arguments_ = params;
  expr.name_ = name;
  return expr.id_;
}

ExpressionId Expression::CreateTernary(ExpressionId condition,
                                       ExpressionId if_true,
                                       ExpressionId if_false) {
  auto& expr = MakeExpression(ExpressionType::TERNARY);
  expr.arguments_.push_back(condition);
  expr.arguments_.push_back(if_true);
  expr.arguments_.push_back(if_false);
  return expr.id_;
}

ExpressionId Expression::CreateBinaryCompare(const std::string& name,
                                             ExpressionId l,
                                             ExpressionId r) {
  auto& expr = MakeExpression(ExpressionType::BINARY_COMPARISON);
  expr.arguments_.push_back(l);
  expr.arguments_.push_back(r);
  expr.name_ = name;
  return expr.id_;
}

ExpressionId Expression::CreateLogicalNegation(ExpressionId v) {
  auto& expr = MakeExpression(ExpressionType::LOGICAL_NEGATION);
  expr.arguments_.push_back(v);
  return expr.id_;
}

ExpressionId Expression::CreateBinaryArithmetic(ExpressionType type,
                                                ExpressionId l,
                                                ExpressionId r) {
  auto& expr = MakeExpression(type);
  expr.arguments_.push_back(l);
  expr.arguments_.push_back(r);
  return expr.id_;
}
Expression::Expression(ExpressionType type, ExpressionId id)
    : type_(type), id_(id) {}

bool Expression::IsArithmetic() const {
  switch (type_) {
    case ExpressionType::PLUS:
    case ExpressionType::MULTIPLICATION:
    case ExpressionType::DIVISION:
    case ExpressionType::MINUS:
    case ExpressionType::UNARY_MINUS:
    case ExpressionType::UNARY_PLUS:
      return true;
    default:
      return false;
  }
}

bool Expression::IsReplaceableBy(const Expression& other) const {
  // Check everything except the id.
  return (type_ == other.type_ && name_ == other.name_ &&
          value_ == other.value_ && arguments_ == other.arguments_);
}

void Expression::Replace(const Expression& other) {
  if (other.id_ == id_) {
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

}  // namespace internal
}  // namespace ceres
