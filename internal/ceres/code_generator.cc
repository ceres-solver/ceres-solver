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
  for (ExpressionId id = 0; id < graph_.Size(); ++id) {
    // By definition of the lhs_id, an expression defines a new variable only if
    // the current_id is identical to the lhs_id.
    const auto& expr = graph_.ExpressionForId(id);
    if (id != expr.lhs_id()) {
      continue;
    }
    //
    // Format:     <type> <id>;
    // Example:    double v_0;
    //
    const std::string declaration_string =
        indentation_ + DataTypeForExpression(expr.type()) + " " +
        VariableForExpressionId(id) + ";";
    code.emplace_back(declaration_string);
  }

  // 3. Print code
  for (ExpressionId id = 0; id < graph_.Size(); ++id) {
    code.emplace_back(ExpressionToString(id));
  }

  PopIndentation();
  CHECK(indentation_.empty()) << "IF - ENDIF missmatch detected.";
  code.emplace_back("}");

  return code;
}

std::string CodeGenerator::ExpressionToString(ExpressionId id) {
  // An expression is converted into a string, by first adding the required
  // indentation spaces and then adding a ExpressionType-specific string. The
  // following list shows the exact output format for each ExpressionType. The
  // placeholders <value>, <name>,... stand for the respective members value_,
  // name_, ... of the current expression. ExpressionIds such as lhs_id and
  // arguments are converted to the corresponding variable name (7 -> "v_7").

  auto& expr = graph_.ExpressionForId(id);

  std::stringstream result;
  result.precision(kFloatingPointPrecision);

  // Convert the variable names of lhs and arguments to string. This makes the
  // big switch/case below more readable.
  std::string lhs;
  if (expr.HasValidLhs()) {
    lhs = VariableForExpressionId(expr.lhs_id());
  }
  std::vector<std::string> args;
  for (ExpressionId id : expr.arguments()) {
    args.push_back(VariableForExpressionId(id));
  }
  auto value = expr.value();
  const auto& name = expr.name();

  switch (expr.type()) {
    case ExpressionType::COMPILE_TIME_CONSTANT: {
      //
      // Format:     <lhs_id> = <value>;
      // Example:    v_0      = 3.1415;
      //
      result << indentation_ << lhs << " = " << value << ";";
      break;
    }
    case ExpressionType::INPUT_ASSIGNMENT: {
      //
      // Format:     <lhs_id> = <name>;
      // Example:    v_0      = _observed_point_x;
      //
      result << indentation_ << lhs << " = " << name << ";";
      break;
    }
    case ExpressionType::OUTPUT_ASSIGNMENT: {
      //
      // Format:     <name>      = <arguments[0]>;
      // Example:    residual[0] = v_51;
      //
      result << indentation_ << name << " = " << args[0] << ";";
      break;
    }
    case ExpressionType::ASSIGNMENT: {
      //
      // Format:     <lhs_id> = <arguments[0]>;
      // Example:    v_1      = v_0;
      //
      result << indentation_ << lhs << " = " << args[0] << ";";
      break;
    }
    case ExpressionType::BINARY_ARITHMETIC: {
      //
      // Format:     <lhs_id> = <arguments[0]> <name> <arguments[1]>;
      // Example:    v_2      = v_0 + v_1;
      //
      result << indentation_ << lhs << " = " << args[0] << " " << name << " "
             << args[1] << ";";
      break;
    }
    case ExpressionType::UNARY_ARITHMETIC: {
      //
      // Format:     <lhs_id> = <name><arguments[0]>;
      // Example:    v_1      = -v_0;
      //
      result << indentation_ << lhs << " = " << name << args[0] << ";";
      break;
    }
    case ExpressionType::BINARY_COMPARISON: {
      //
      // Format:     <lhs_id> =  <arguments[0]> <name> <arguments[1]>;
      // Example:    v_2   = v_0 < v_1;
      //
      result << indentation_ << lhs << " = " << args[0] << " " << name << " "
             << args[1] << ";";
      break;
    }
    case ExpressionType::LOGICAL_NEGATION: {
      //
      // Format:     <lhs_id> = !<arguments[0]>;
      // Example:    v_1   = !v_0;
      //
      result << indentation_ << lhs << " = !" << args[0] << ";";
      break;
    }
    case ExpressionType::FUNCTION_CALL: {
      //
      // Format:     <lhs_id> = <name>(<arguments[0]>, <arguments[1]>, ...);
      // Example:    v_1   = sin(v_0);
      //
      result << indentation_ << lhs << " = " << name << "(";
      result << (args.size() ? args[0] : "");
      for (int i = 1; i < args.size(); ++i) {
        result << ", " << args[i];
      }
      result << ");";
      break;
    }
    case ExpressionType::IF: {
      //
      // Format:     if (<arguments[0]>) {
      // Example:    if (v_0) {
      // Special:    Adds 1 level of indentation for all following
      //             expressions.
      //
      result << indentation_ << "if (" << args[0] << ") {";
      PushIndentation();
      break;
    }
    case ExpressionType::ELSE: {
      //
      // Format:     } else {
      // Example:    } else {
      // Special:    This expression is printed with one less level of
      //             indentation.
      //
      PopIndentation();
      result << indentation_ << "} else {";
      PushIndentation();
      break;
    }
    case ExpressionType::ENDIF: {
      //
      // Format:     }
      // Example:    }
      // Special:    Removes 1 level of indentation for this and all
      //             following expressions.
      //
      PopIndentation();
      result << indentation_ << "}";
      break;
    }
    case ExpressionType::NOP: {
      //
      // Format:     // <NOP>
      // Example:    // <NOP>
      //
      result << indentation_ << "// <NOP>";
      break;
    }
    default:
      CHECK(false) << "CodeGenerator::ToString for ExpressionType "
                   << static_cast<int>(expr.type()) << " not implemented!";
  }
  return result.str();
}

std::string CodeGenerator::VariableForExpressionId(ExpressionId id) {
  //
  // Format:     <variable_prefix><id>
  // Example:    v_42
  //
  auto& expr = graph_.ExpressionForId(id);
  CHECK(expr.lhs_id() == id)
      << "ExpressionId " << id
      << " does not have a name (it has not been declared).";
  return options_.variable_prefix + std::to_string(expr.lhs_id());
}

std::string CodeGenerator::DataTypeForExpression(ExpressionType type) {
  std::string type_string;
  switch (type) {
    case ExpressionType::BINARY_COMPARISON:
    case ExpressionType::LOGICAL_NEGATION:
      type_string = "bool";
      break;
    case ExpressionType::IF:
    case ExpressionType::ELSE:
    case ExpressionType::ENDIF:
    case ExpressionType::NOP:
      type_string = "void";
      break;
    default:
      type_string = "double";
  }
  return type_string;
}

void CodeGenerator::PushIndentation() {
  for (int i = 0; i < options_.indentation_spaces_per_level; ++i) {
    indentation_.push_back(' ');
  }
}

void CodeGenerator::PopIndentation() {
  for (int i = 0; i < options_.indentation_spaces_per_level; ++i) {
    CHECK(!indentation_.empty()) << "IF - ENDIF missmatch detected.";
    indentation_.pop_back();
  }
}

}  // namespace internal
}  // namespace ceres
