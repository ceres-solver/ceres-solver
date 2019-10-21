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
#include <sstream>
#include "assert.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

CodeGenerator::CodeGenerator(const ExpressionGraph& graph,
                             const Options& options)
    : graph_(graph), options_(options) {}

std::vector<std::string> CodeGenerator::Generate() {
  std::vector<std::string> code;

  // 1. Print the header
  if (!options_.function_name.empty()) {
    code.emplace_back(options_.function_name);
  }

  code.emplace_back("{");
  PushIndentation();

  // 2. Print declarations
  for (ExpressionId id : graph_) {
    // By definition of the lhs_id, an expression defines a new variable only if
    // the current id idendical to the lhs_id.
    auto& expr = graph_.ExpressionForId(id);
    if (id != expr.lhs_id()) {
      continue;
    }

    std::string type_string;
    switch (expr.type()) {
      case ExpressionType::BINARY_COMPARISON:
      case ExpressionType::LOGICAL_NEGATION:
        type_string = "bool";
        break;
      default:
        type_string = "double";
    }
    std::string declaration_string = current_indentation_ + type_string + " " +
                                     VariableForExpressionId(id) + ";";
    code.emplace_back(declaration_string);
  }

  // 3. Print code
  for (ExpressionId id : graph_) {
    code.emplace_back(ExpressionToString(id));
  }

  PopIndentation();
  CHECK(current_indentation_.empty()) << "IF - ENDIF missmatch detected.";
  code.emplace_back("}");

  return code;
}

std::string CodeGenerator::ExpressionToString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);
  auto args = expr.arguments();

  switch (expr.type()) {
    case ExpressionType::NOP:
      return "// <NOP>";
    case ExpressionType::COMPILE_TIME_CONSTANT: {
      std::stringstream sstream;
      sstream.precision(kFloatingPointPrecision);
      sstream << expr.value();
      return current_indentation_ + VariableForExpressionId(expr.lhs_id()) +
             " = " + sstream.str() + ";";
    }
    case ExpressionType::RUNTIME_CONSTANT:
    case ExpressionType::PARAMETER:
      return current_indentation_ + VariableForExpressionId(expr.lhs_id()) +
             " = " + expr.name() + ";";
    case ExpressionType::ASSIGNMENT:
      return current_indentation_ + VariableForExpressionId(expr.lhs_id()) +
             " = " + VariableForExpressionId(args[0]) + ";";
    case ExpressionType::BINARY_ARITHMETIC:
      return current_indentation_ + VariableForExpressionId(expr.lhs_id()) +
             " = " + VariableForExpressionId(args[0]) + " " + expr.name() +
             " " + VariableForExpressionId(args[1]) + ";";
    case ExpressionType::OUTPUT_ASSIGNMENT:
      return current_indentation_ + expr.name() + " = " +
             VariableForExpressionId(args[0]) + ";";
    case ExpressionType::FUNCTION_CALL: {
      std::string argument_list;
      for (auto a : args) {
        argument_list += VariableForExpressionId(a) + ", ";
      }
      // Remove last ", "
      if (!args.empty()) {
        argument_list.pop_back();
        argument_list.pop_back();
      }
      return current_indentation_ + VariableForExpressionId(expr.lhs_id()) +
             " = " + expr.name() + "(" + argument_list + ");";
    }
    case ExpressionType::IF: {
      std::string result = current_indentation_ + "if(" +
                           VariableForExpressionId(args[0]) + "){";
      PushIndentation();
      return result;
    }
    case ExpressionType::ELSE: {
      PopIndentation();
      std::string result = current_indentation_ + "} else {";
      PushIndentation();
      return result;
    }
    case ExpressionType::ENDIF: {
      PopIndentation();
      return current_indentation_ + "}";
    }
      // TODO: remaining types...
    default:
      CHECK(false) << "CodeGenerator::ToString for ExpressionType "
                   << (int)expr.type() << " not implemented!";
  }
}

std::string CodeGenerator::VariableForExpressionId(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);
  CHECK(expr.lhs_id() == id)
      << "ExpressionId " << id
      << " does not have a name (it has not been declared).";
  return "v_" + std::to_string(expr.lhs_id());
}

void CodeGenerator::PushIndentation() {
  for (int i = 0; i < options_.indentation_spaces_per_level; ++i) {
    current_indentation_.push_back(' ');
  }
}

void CodeGenerator::PopIndentation() {
  for (int i = 0; i < options_.indentation_spaces_per_level; ++i) {
    CHECK(!current_indentation_.empty()) << "IF - ENDIF missmatch detected.";
    current_indentation_.pop_back();
  }
}

}  // namespace internal
}  // namespace ceres
