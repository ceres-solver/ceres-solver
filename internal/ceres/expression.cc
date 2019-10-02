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
#include "assert.h"

#include <sstream>
namespace ceres {
namespace internal {

// During execution, the expressions add themself into this vector. This
// allows us to see what code was executed and optimize it later.
static ExpressionTree* expression_pool = nullptr;

static Expression& MakeExpression(ExpressionType type) {
  assert(expression_pool);
  return expression_pool->MakeExpression(type);
}

Expression& ExpressionTree::MakeExpression(ExpressionType type) {
  auto id = data_.size();
  ExpressionRef e;
  e.id = id;
  Expression expr(type, e);
  data_.push_back(expr);
  return data_.back();
}

bool ExpressionTree::DependsOn(ExpressionRef A, ExpressionRef B) const {
  // Depth first search on the expression tree
  // Equivalent Recursive Implementation:
  //  if (A.DirectlyDependsOn(B)) return true;
  //  for (auto p : A.params_) {
  //    if (pool[p.id].DependsOn(B, pool)) return true;
  //  }
  std::vector<ExpressionRef> stack = get(A).params_;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top.id == B.id) return true;
    auto& expr = get(top);
    stack.insert(stack.end(), expr.params_.begin(), expr.params_.end());
  }
  return false;
}

void StartRecordingExpressions() {
  assert(expression_pool == nullptr);
  expression_pool = new ExpressionTree;
}

ExpressionTree StopRecordingExpressions() {
  assert(expression_pool);
  ExpressionTree result = std::move(*expression_pool);
  delete expression_pool;
  expression_pool = nullptr;
  return result;
}

std::string ExpressionRef::ToString() { return std::to_string(id); }

bool Expression::IsSimpleArithmetic() const {
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
  if (type_ == ExpressionType::NOP) {
    return false;
  }

  // Check everything except the id and the params.
  if (!(type_ == other.type_ && name_ == other.name_ &&
        value_ == other.value_ && params_.size() == other.params_.size())) {
    return false;
  }

  // Check if the argument ids are equal.
  for (int i = 0; i < params_.size(); ++i) {
    if (params_[i].id != other.params_[i].id) {
      return false;
    }
  }
  return true;
}

void Expression::Replace(const Expression& other) {
  if (other.id_.id == id_.id) {
    return;
  }
  auto currentId = id_;
  (*this) = other;
  id_ = currentId;
}

bool Expression::DirectlyDependsOn(ExpressionRef other) const {
  for (auto p : params_) {
    if (p.id == other.id) {
      return true;
    }
  }
  return false;
}

std::string Expression::ResultTypeAsString() const {
  if (type_ == ExpressionType::BINARY_COMPARE) {
    return "const bool";
  }
  return "const double";
}

bool Expression::IsConstant(double constant) const {
  return type_ == ExpressionType::COMPILE_TIME_CONSTANT && value_ == constant;
}

void Expression::TurnIntoNop() {
  type_ = ExpressionType::NOP;
  params_.clear();
}

std::string Expression::LhsName() const {
  return "v_" + std::to_string(id_.id);
}

ExpressionRef Expression::MakeConstant(double v) {
  auto& expr =
      expression_pool->MakeExpression(ExpressionType::COMPILE_TIME_CONSTANT);
  expr.value_ = v;
  return expr.id_;
}

ExpressionRef Expression::MakeRuntimeConstant(const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::RUNTIME_CONSTANT);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeParameter(const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::PARAMETER);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeAssignment(ExpressionRef v) {
  auto& expr = MakeExpression(ExpressionType::ASSIGNMENT);
  expr.params_.push_back(v);
  return expr.id_;
}

ExpressionRef Expression::MakeUnaryArithmetic(ExpressionRef v,
                                              ExpressionType type_) {
  auto& expr = MakeExpression(type_);
  expr.params_.push_back(v);
  return expr.id_;
}

ExpressionRef Expression::MakeOutputAssignment(ExpressionRef v,
                                               const std::string& name) {
  auto& expr = MakeExpression(ExpressionType::OUTPUT_ASSIGNMENT);
  expr.params_.push_back(v);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeFunctionCall(
    const std::string& name, const std::vector<ExpressionRef>& params) {
  auto& expr = MakeExpression(ExpressionType::FUNCTION_CALL);
  expr.params_ = params;
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeTernary(ComparisonExpressionRef c,
                                      ExpressionRef a,
                                      ExpressionRef b) {
  auto& expr = MakeExpression(ExpressionType::TERNARY);
  expr.params_.push_back(c.id);
  expr.params_.push_back(a);
  expr.params_.push_back(b);
  return expr.id_;
}

ExpressionRef Expression::MakeBinaryCompare(const std::string& name,
                                            ExpressionRef l,
                                            ExpressionRef r) {
  auto& expr = MakeExpression(ExpressionType::BINARY_COMPARE);
  expr.params_.push_back(l);
  expr.params_.push_back(r);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeBinaryArithmetic(ExpressionType type,
                                               ExpressionRef l,
                                               ExpressionRef r) {
  auto& expr = MakeExpression(type);
  expr.params_.push_back(l);
  expr.params_.push_back(r);
  return expr.id_;
}
Expression::Expression(ExpressionType type, ExpressionRef id)
    : type_(type), id_(id) {}

}  // namespace internal
}  // namespace ceres
