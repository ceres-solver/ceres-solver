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

#include "ceres/internal/code_generator.h"
#include "assert.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

CodeGenerator::CodeGenerator(const ExpressionGraph& graph) : graph_(graph) {}

void CodeGenerator::Print(std::ostream& strm) {
  for (int i = 0; i < graph_.Size(); ++i) {
    strm << ToString(i) << std::endl;
  }
}

std::string CodeGenerator::ToString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);
  return LhsTypeString(expr) + LhsNameString(expr) + OperatorString(expr) +
         RhsString(expr);
}

std::string CodeGenerator::LhsTypeString(const Expression& expr) {
  switch (expr.Type()) {
    case ExpressionType::BINARY_COMPARISON:
    case ExpressionType::LOGICAL_NEGATION:
      return "bool";
    case ExpressionType::NOP:
      return "";
    default:
      return "double ";
  }
}

std::string CodeGenerator::LhsNameString(const Expression& expr) {
  switch (expr.Type()) {
    case ExpressionType::NOP:
      return "";
    default: {
      auto id = expr.Id();
      return "v_" + std::to_string(id) + " ";
    }
  }
}

std::string CodeGenerator::OperatorString(const Expression& expr) {
  switch (expr.Type()) {
    case ExpressionType::NOP:
      return "";
    default:
      return "= ";
  }
}

std::string CodeGenerator::RhsString(const Expression& expr) {
  auto args = expr.Arguments();
  switch (expr.Type()) {
    case ExpressionType::NOP:
      return "";
    case ExpressionType::COMPILE_TIME_CONSTANT:
      return ValueToString(expr.Value()) + ";";
    case ExpressionType::RUNTIME_CONSTANT:
    case ExpressionType::PARAMETER:
      return expr.Name() + ";";
    case ExpressionType::ASSIGNMENT:
      return LhsNameStringCheck(args[0]) + ";";
    case ExpressionType::PLUS:
      return LhsNameStringCheck(args[0]) + "+ " + LhsNameStringCheck(args[1]) +
             ";";
    case ExpressionType::MINUS:
      return LhsNameStringCheck(args[0]) + "- " + LhsNameStringCheck(args[1]) +
             ";";
    case ExpressionType::MULTIPLICATION:
      return LhsNameStringCheck(args[0]) + "* " + LhsNameStringCheck(args[1]) +
             ";";
    case ExpressionType::DIVISION:
      return LhsNameStringCheck(args[0]) + "/ " + LhsNameStringCheck(args[1]) +
             ";";
      // TODO: remaining types...
    default:
      return "ERROR: TODO";
  }
}

std::string CodeGenerator::LhsNameStringCheck(ExpressionId id) {
  auto str = LhsNameString(graph_.ExpressionForId(id));
  assert(!str.empty());
  return str;
}

std::string CodeGenerator::ValueToString(double value) {
  // TODO: higher precision.
  return std::to_string(value);
}

}  // namespace internal
}  // namespace ceres
