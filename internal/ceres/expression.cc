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
namespace ceres {
namespace internal {

std::vector<Expression> Expression::expression_data;

bool Expression::IsSimpleArithmetic() {
  switch (type_) {
    case ExpressionType::PLUS:
    case ExpressionType::MULTIPLICATION:
    case ExpressionType::DIVISION:
    case ExpressionType::MINUS:
    case ExpressionType::UNARY_MINUS:
      return true;
    default:
      return false;
  }
}

bool Expression::IsReplaceableBy(const Expression& other) {
  if (type_ == ExpressionType::NOP) {
    return false;
  }

  // Check everything except the id and the params.
  if (!(std::make_tuple(type_, name_, constant_value_, params_.size()) ==
        std::make_tuple(other.type_,
                        other.name_,
                        other.constant_value_,
                        other.params_.size()))) {
    return false;
  }

  // Check if the params are equal.
  for (int i = 0; i < params_.size(); ++i) {
    if (params_[i].id != other.params_[i].id) {
      return false;
    }
  }
  return true;
}

bool Expression::DependsOn(ExpressionRef other) {
  for (auto p : params_) {
    if (p.id == other.id) {
      return true;
    }
  }
  return false;
}

void Expression::Replace(const Expression& other) {
  if (other.id_.id == id_.id) {
    return;
  }
  auto currentId = id_;
  (*this) = other;
  id_ = currentId;
}

std::string Expression::ResultTypeString() const {
  if (type_ == ExpressionType::BINARY_COMPARE) {
    return "const bool";
  }
  return "const double";
}

bool Expression::IsConstantZero() {
  return type_ == ExpressionType::COMPILE_TIME_CONSTANT && constant_value_ == 0;
}
bool Expression::IsConstantOne() {
  return type_ == ExpressionType::COMPILE_TIME_CONSTANT && constant_value_ == 1;
}

void Expression::MakeNop() {
  type_ = ExpressionType::NOP;
  params_.clear();
}

std::string Expression::LhsName() const {
  return "v_" + std::to_string(id_.id);
}

ExpressionRef Expression::MakeConstant(double v) {
  auto& expr = MakeExpr(ExpressionType::COMPILE_TIME_CONSTANT);
  expr.constant_value_ = v;
  return expr.id_;
}

ExpressionRef Expression::MakeExternalConstant(const std::string& name) {
  auto& expr = MakeExpr(ExpressionType::EXTERNAL_CONSTANT);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeParameter(const std::string& name) {
  auto& expr = MakeExpr(ExpressionType::PARAMETER);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeAssign(ExpressionRef v) {
  auto& expr = MakeExpr(ExpressionType::ASSIGN);
  expr.params_.push_back(v);
  return expr.id_;
}

ExpressionRef Expression::MakeUnaryMinus(ExpressionRef v) {
  auto& expr = MakeExpr(ExpressionType::UNARY_MINUS);
  expr.params_.push_back(v);
  return expr.id_;
}

ExpressionRef Expression::MakeOutput(ExpressionRef v, const std::string& name) {
  auto& expr = MakeExpr(ExpressionType::OUTPUT_ASSIGN);
  expr.params_.push_back(v);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeFunction(const std::string& name,
                                       std::vector<ExpressionRef> params) {
  auto& expr = MakeExpr(ExpressionType::FUNCTION_CALL);
  expr.params_ = params;
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeTernary(ComparisonExpressionRef c,
                                      ExpressionRef a,
                                      ExpressionRef b) {
  auto& expr = MakeExpr(ExpressionType::TERNARY);
  expr.params_.push_back(c.id);
  expr.params_.push_back(a);
  expr.params_.push_back(b);
  return expr.id_;
}

ExpressionRef Expression::MakeBinaryCompare(const std::string& name,
                                            ExpressionRef l,
                                            ExpressionRef r) {
  auto& expr = MakeExpr(ExpressionType::BINARY_COMPARE);
  expr.params_.push_back(l);
  expr.params_.push_back(r);
  expr.name_ = name;
  return expr.id_;
}

ExpressionRef Expression::MakeBinaryArithmetic(ExpressionType type,
                                               ExpressionRef l,
                                               ExpressionRef r) {
  auto& expr = MakeExpr(type);
  expr.params_.push_back(l);
  expr.params_.push_back(r);
  return expr.id_;
}
Expression::Expression(ExpressionType type, ExpressionRef id)
    : type_(type), id_(id) {}

Expression& Expression::MakeExpr(ExpressionType type) {
  auto id = expression_data.size();
  ExpressionRef e;
  e.id = id;
  Expression expr(type, e);
  expression_data.push_back(expr);
  return expression_data.back();
}

}  // namespace internal
}  // namespace ceres
