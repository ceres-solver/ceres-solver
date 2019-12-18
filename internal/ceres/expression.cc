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

#include "ceres/codegen/internal/expression.h"
#include <algorithm>
#include "glog/logging.h"

namespace ceres {
namespace internal {

std::string ExpressionReturnTypeToString(ExpressionReturnType type) {
  switch (type) {
    case ExpressionReturnType::SCALAR:
      return "double";
    case ExpressionReturnType::BOOLEAN:
      return "bool";
    case ExpressionReturnType::VOID:
      return "void";
    default:
      CHECK(false) << "Unknown ExpressionReturnType.";
      return "";
  }
}

Expression::Expression(ExpressionType type,
                       ExpressionReturnType return_type,
                       ExpressionId lhs_id,
                       const std::vector<ExpressionId>& arguments,
                       const std::string& name,
                       double value)
    : type_(type),
      return_type_(return_type),
      lhs_id_(lhs_id),
      arguments_(arguments),
      name_(name),
      value_(value) {}

Expression Expression::CreateCompileTimeConstant(double v) {
  return Expression(ExpressionType::COMPILE_TIME_CONSTANT,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    {},
                    "",
                    v);
}

Expression Expression::CreateInputAssignment(const std::string& name) {
  return Expression(ExpressionType::INPUT_ASSIGNMENT,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    {},
                    name);
}

Expression Expression::CreateOutputAssignment(ExpressionId v,
                                              const std::string& name) {
  return Expression(ExpressionType::OUTPUT_ASSIGNMENT,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    {v},
                    name);
}

Expression Expression::CreateAssignment(ExpressionId dst, ExpressionId src) {
  return Expression(
      ExpressionType::ASSIGNMENT, ExpressionReturnType::SCALAR, dst, {src});
}

Expression Expression::CreateBinaryArithmetic(const std::string& op,
                                              ExpressionId l,
                                              ExpressionId r) {
  return Expression(ExpressionType::BINARY_ARITHMETIC,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    {l, r},
                    op);
}

Expression Expression::CreateUnaryArithmetic(const std::string& op,
                                             ExpressionId v) {
  return Expression(ExpressionType::UNARY_ARITHMETIC,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    {v},
                    op);
}

Expression Expression::CreateBinaryCompare(const std::string& name,
                                           ExpressionId l,
                                           ExpressionId r) {
  return Expression(ExpressionType::BINARY_COMPARISON,
                    ExpressionReturnType::BOOLEAN,
                    kInvalidExpressionId,
                    {l, r},
                    name);
}

Expression Expression::CreateLogicalNegation(ExpressionId v) {
  return Expression(ExpressionType::LOGICAL_NEGATION,
                    ExpressionReturnType::BOOLEAN,
                    kInvalidExpressionId,
                    {v});
}

Expression Expression::CreateScalarFunctionCall(
    const std::string& name, const std::vector<ExpressionId>& params) {
  return Expression(ExpressionType::FUNCTION_CALL,
                    ExpressionReturnType::SCALAR,
                    kInvalidExpressionId,
                    params,
                    name);
}

Expression Expression::CreateLogicalFunctionCall(
    const std::string& name, const std::vector<ExpressionId>& params) {
  return Expression(ExpressionType::FUNCTION_CALL,
                    ExpressionReturnType::BOOLEAN,
                    kInvalidExpressionId,
                    params,
                    name);
}

Expression Expression::CreateIf(ExpressionId condition) {
  return Expression(ExpressionType::IF,
                    ExpressionReturnType::VOID,
                    kInvalidExpressionId,
                    {condition});
}

Expression Expression::CreateElse() { return Expression(ExpressionType::ELSE); }

Expression Expression::CreateEndIf() {
  return Expression(ExpressionType::ENDIF);
}

Expression Expression::CreateComment(const std::string& comment) {
  return Expression(ExpressionType::COMMENT,
                    ExpressionReturnType::VOID,
                    kInvalidExpressionId,
                    {},
                    comment);
}

bool Expression::IsArithmeticExpression() const {
  return !IsControlExpression();
}

bool Expression::IsControlExpression() const {
  return type_ == ExpressionType::IF || type_ == ExpressionType::ELSE ||
         type_ == ExpressionType::ENDIF || type_ == ExpressionType::NOP ||
         type_ == ExpressionType::COMMENT;
}

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
  // The default constructor creates a NOP expression!
  *this = Expression();
}

bool Expression::operator==(const Expression& other) const {
  return type() == other.type() && return_type() == other.return_type() &&
         name() == other.name() && value() == other.value() &&
         lhs_id() == other.lhs_id() && arguments() == other.arguments();
}

bool Expression::IsSemanticallyEquivalentTo(const Expression& other) const {
  return type() == other.type() && name() == other.name() &&
         value() == other.value() &&
         arguments().size() == other.arguments().size();
}

}  // namespace internal
}  // namespace ceres
