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

    // The string used for the floating point
    std::string floating_point_type_string = "double";
  };

  CodeGenerator(const ExpressionGraph& graph, const Options& options);
  std::vector<std::string> Generate();

 private:
  std::string ExpressionToString(ExpressionId id);

  // Helper function to get the name of an expression.
  // If the expression does not have a valid name an error is generated.
  std::string VariableForExpressionId(ExpressionId id);

  void PushIndentation();
  void PopIndentation();

  const ExpressionGraph& graph_;
  const Options options_;
  std::string current_indentation_ = "";
  static constexpr int kFloatingPointPrecision = 25;
};

}  // namespace internal
}  // namespace ceres
#endif
