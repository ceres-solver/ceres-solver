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

void CodeGenerator::Print(std::ostream& strm) {
  int indentation_level_start = 1;
  int indentation_level = indentation_level_start;

  if (!options_.function_name.empty()) {
    strm << options_.function_name << std::endl;
  }

  strm << "{" << std::endl;

  PrintDeclarations(strm, indentation_level);
  for (int i = 0; i < graph_.Size(); ++i) {
    strm << ToString(i, indentation_level) << std::endl;
  }
  assert(indentation_level == indentation_level_start);

  strm << "}" << std::endl;
}

void CodeGenerator::PrintDeclarations(std::ostream& strm,
                                      int indentation_level) {
  for (int i = 0; i < graph_.Size(); ++i) {
    strm << IndentationString(indentation_level) << LhsTypeString(i) << " "
         << LhsNameString(i) << std::endl;
  }
}

std::string CodeGenerator::ToString(ExpressionId id) {
  assert(id != kInvalidExpressionId);
  return LhsNameString(id) + OperatorString(id) + RhsString(id);
}

std::string CodeGenerator::ToString(ExpressionId id, int& indentation_level) {
  return IndentationString(indentation_level) + ToString(id);
}

std::string CodeGenerator::LhsTypeString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);

  switch (expr.type()) {
    case ExpressionType::BINARY_COMPARISON:
    case ExpressionType::LOGICAL_NEGATION:
      return "bool ";
    case ExpressionType::NOP:
    case ExpressionType::OUTPUT_ASSIGNMENT:
      return "";
    default:
      return options_.floating_point_type_string + " ";
  }
}

std::string CodeGenerator::LhsNameString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);

  switch (expr.type()) {
    case ExpressionType::NOP:
      return "";
    case ExpressionType::OUTPUT_ASSIGNMENT:
      return expr.name();
    default:
      return "v_" + std::to_string(id);
  }
}

std::string CodeGenerator::OperatorString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);
  switch (expr.type()) {
    case ExpressionType::NOP:
      return "// <NOP>";
    default:
      return " = ";
  }
}

std::string CodeGenerator::RhsString(ExpressionId id) {
  auto& expr = graph_.ExpressionForId(id);
  auto args = expr.arguments();
  switch (expr.type()) {
    case ExpressionType::NOP:
      return "";
    case ExpressionType::COMPILE_TIME_CONSTANT:
      return ValueToString(expr.value()) + ";";
    case ExpressionType::RUNTIME_CONSTANT:
    case ExpressionType::PARAMETER:
      return expr.name() + ";";
    case ExpressionType::ASSIGNMENT:
      return LhsNameStringCheck(args[0]) + ";";
    case ExpressionType::BINARY_ARITHMETIC:
      return LhsNameStringCheck(args[0]) + +" " + expr.name() + " " +
             LhsNameStringCheck(args[1]) + ";";
    case ExpressionType::OUTPUT_ASSIGNMENT:
      return LhsNameStringCheck(args[0]) + ";";
      // TODO: remaining types...
    default:
      return "ERROR: TODO";
  }
}

std::string CodeGenerator::LhsNameStringCheck(ExpressionId id) {
  auto str = LhsNameString(id);
  assert(!str.empty());
  return str;
}

std::string CodeGenerator::ValueToString(double value) {
  std::stringstream sstream;
  // Todo: how much do we actually need for doubles?
  sstream.precision(25);
  sstream << value;
  return sstream.str();
}

std::string CodeGenerator::IndentationString(int indentation_level) {
  std::string indentation;
  for (int i = 0; i < indentation_level * options_.indentation_spaces_per_level;
       ++i) {
    indentation += " ";
  }
  return indentation;
}

}  // namespace internal
}  // namespace ceres
