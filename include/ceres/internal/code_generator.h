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

#include "expression.h"
#include "expression_graph.h"

#include <string>
#include <vector>

namespace ceres {
namespace internal {

// This class is used to convert an expression graph into a string. The typical
// pipeline is:
//
// 1. Record ExpressionGraph
// 2. Optimize ExpressionGraph
// 3. Generate code (this class here)
//
// The CodeGenerator operates in the following way:
//
// 1. Print Header
//    - The header string is defined in the options.
//    - This is usually the function name including the parameter list.
//
// 2. Print Declarations
//    - Declare all used variables
//    - Example:
//      double v_0;
//      double v_1;
//      bool v_3;
//      ...
//
// 3. Print Code
//    - Convert each expression line by line to a string
//    - Example:
//      v_2 = v_0 + v_1
//      if(v_5) {
//      ....
//
// An expression is converted into a string, by first adding the required
// indentation spaces and then adding a ExpressionType-specific string. The
// following list shows the exact output format for each ExpressionType. The
// placeholders <value>, <name>,... stand for the respective members value_,
// name_, ... of the current expression. ExpressionIds such as lhs_id and
// arguments are converted to the corresponding variable name (7 -> "v_7").
//
// COMPILE_TIME_CONSTANT
//    Format:     <lhs_id> = <value>;
//    Example:    v_0      = 3.1415;
//
// RUNTIME_CONSTANT
//    Format:     <lhs_id> = <name>;
//    Example:    v_0      = _observed_point_x;
//
// PARAMETER
//    Format:     <lhs_id> = <name>;
//    Example:    v_0      = parameters[1][5];
//
// OUTPUT_ASSIGNMENT
//    Format:     <name>      = <arguments[0]>;
//    Example:    residual[0] = v_51;
//
// ASSIGNMENT
//    Format:     <lhs_id> = <arguments[0]>;
//    Example:    v_1      = v_0;
//
// BINARY_ARITHMETIC
//    Format:     <lhs_id> = <arguments[0]> <name> <arguments[1]>;
//    Example:    v_2      = v_0 + v_1;
//
// UNARY_ARITHMETIC
//    Format:     <lhs_id> = <name><arguments[0]>;
//    Example:    v_1      = -v_0;
//
// BINARY_COMPARISON
//    Format:     <lhs_id> =  <arguments[0]> <name> <arguments[1]>;
//    Example:    v_2   = v_0 < v_1;
//
// LOGICAL_NEGATION
//    Format:     <lhs_id> = !<arguments[0]>;
//    Example:    v_1   = !v_0;
//
// FUNCTION_CALL
//    Format:     <lhs_id> = <name>(<arguments[0]>, <arguments[1]>, ...);
//    Example:    v_1   = sin(v_0);
//
// IF
//    Format:     if (<arguments[0]>) {
//    Example:    if (v_0) {
//    Special:    Adds 1 level of indentation for all following expressions.
//
// ELSE
//    Format:     } else {
//    Example:    } else {
//    Special:    This expression is printed with one less level of indentation.
//
// ENDIF
//    Format:     }
//    Example:    }
//    Special:    Removes 1 level of indentation for this and all following
//                expressions.
//
// NOP
//    Format:     // <NOP>
//    Example:    // <NOP>

class CodeGenerator {
 public:
  struct Options {
    // Name of the function.
    // Example:
    // "bool Evaluate(const double* x, double* res)"
    std::string function_name = "";

    // Number of spaces added for each level of indentation.
    int indentation_spaces_per_level = 2;

    // The prefix added to each variable name.
    std::string variable_prefix = "v_";
  };

  CodeGenerator(const ExpressionGraph& graph, const Options& options);
  std::vector<std::string> Generate();

 private:
  // Converts a single expression given by id to a string.
  // The format depends on the ExpressionType.
  // See ExpressionType in expression.h for more detailed how the different
  // lines will look like.
  std::string ExpressionToString(ExpressionId id);

  // Helper function to get the name of an expression.
  // If the expression does not have a valid name an error is generated.
  std::string VariableForExpressionId(ExpressionId id);

  // Adds one level of indentation. Currently only called when parsing IF
  // expressions.
  void PushIndentation();

  // Removes one level of indentation. Currently only used by ENDIF.
  void PopIndentation();

  const ExpressionGraph& graph_;
  const Options options_;
  std::string current_indentation_ = "";
  static constexpr int kFloatingPointPrecision = 25;
};

}  // namespace internal
}  // namespace ceres
#endif
