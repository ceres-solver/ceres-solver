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
#include <sstream>
#include "assert.h"
#include "ceres/internal/expression_tree.h"
namespace ceres {
namespace internal {

static Expression& MakeExpression(ExpressionType type) {
  auto pool = GetCurrentExpressionTree();
  assert(pool);
  return pool->MakeExpression(type);
}

ExpressionRef Expression::MakeConstant(double v) {
  auto& expr = MakeExpression(ExpressionType::COMPILE_TIME_CONSTANT);
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
  auto current_id = id_;
  (*this) = other;
  id_ = current_id;
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

}  // namespace internal
}  // namespace ceres
