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
//
#ifndef CERES_PUBLIC_CODE_GENERATOR_H_
#define CERES_PUBLIC_CODE_GENERATOR_H_

#include "expression_graph.h"

namespace ceres {
namespace internal {

// This class is used to convert an expression graph into a string. The typical
// pipeline is:
//
// 1. Record ExpressionGraph
// 2. Optimize ExpressionGraph
// 3. Generate code (this class here)
class CodeGenerator {
 public:
  struct Options {
    // Additional string printed before the code.
    // The typical use case is to add the function declaration here.
    // Example:
    // header = "bool Evaluate(const double* x, double* res)"
    //
    // Note: The curly braces can be added by setting add_root_block = true
    std::string header = "";

    // Add an additional block around everything.
    // - Add { } at the beginning and end.
    // - Add 1 level of indentation.
    bool add_root_block = false;

    // Precision settings for converting compile time constants to strings.
    int compile_time_constant_precision = 25;

    // String added to the left of an expression for each level of indentation.
    std::string indentation_string = "  ";

    // The prefix added to each variable name.
    std::string variable_prefix = "v_";

    // The string used for the floating point and boolean type.
    std::string floating_point_type_string = "double";
    std::string boolean_type_string = "bool";

    // Print NOPs as "// <NOP> <id>"
    bool show_nops = false;
  };

  CodeGenerator(const ExpressionGraph& graph, const Options& options);
  void Print(std::ostream& strm);

 private:
  // The following format is used for converting expressions to strings.
  // <LhsType> <LhsName> <Operator> <Rhs>
  //
  // double    v_1       =          sin(v_0)
  // bool      v_2       =          v_1 < v_0
  //           v_2       =          v_0
  //                     if         (v_7)
  //
  std::string ToString(ExpressionId id);

  // Returns "IndentationLevel*Indentation + ToString(id)"
  // The identation level sets the number of identation strings that are added
  // to the left. Expressions that open or close a new block modify the
  // identation level.
  std::string ToString(ExpressionId id, int& indentation_level);

  // Data type as string of the left hand side. 'bool' for comparisons, 'double'
  // for arithmetic/function expressions, and empty for special types such as
  // NOP/IF/MULTI_ASSIGNMENT/...
  std::string LhsTypeString(ExpressionId id);

  // Variable name of the left hand side. The name consists of the
  // variable_prefix + the target id. Can be empty for special expression types.
  // Example: v_72
  std::string LhsNameString(ExpressionId id);

  // The operator between the left and right hand side. This returns '=' in
  // almost all cases. Exceptions are NOP and IF/ELSE/... expressions.
  std::string OperatorString(ExpressionId id);

  // The right hand side converted to a string. Parameters are converted using
  // the function LhsNameStringCheck.
  std::string RhsString(ExpressionId id);

  // Helper function to get the name of an expression.
  // If the expression does not have a valid name an error is generated.
  std::string LhsNameStringCheck(ExpressionId id);

  // Convert a compile time constant value to a string. The precision is set by
  // a code generator parameter.
  std::string ValueToString(double value);

  const ExpressionGraph& graph_;
  const Options options_;
};

}  // namespace internal
}  // namespace ceres
#endif
