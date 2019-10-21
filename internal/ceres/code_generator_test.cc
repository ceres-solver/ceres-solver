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
#define CERES_CODEGEN

#include "ceres/internal/code_generator.h"
#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

static void GenerateAndCheck(const ExpressionGraph& graph,
                             const std::vector<std::string>& reference) {
  CodeGenerator::Options generator_options;
  CodeGenerator gen(graph, generator_options);
  auto code = gen.Generate();
  EXPECT_EQ(code.size(), reference.size());

  for (int i = 0; i < code.size(); ++i) {
    EXPECT_EQ(code[i], reference[i]) << "Invalid Line: " << (i + 1);
  }
}

using T = ExpressionRef;

TEST(CodeGenerator, Empty) {
  StartRecordingExpressions();
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{", "}"};
  GenerateAndCheck(graph, expected_code);
}

// Now we add one TEST for each ExpressionType.
TEST(CodeGenerator, COMPILE_TIME_CONSTANT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(123.5);
  T c = T(1 + 1);
  T d;  // Uninitialized variables should not generate code!
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  v_0 = 0;",
                                            "  v_1 = 123.5;",
                                            "  v_2 = 2;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, INPUT_ASSIGNMENT) {
  double local_variable = 5.0;
  StartRecordingExpressions();
  T a = CERES_LOCAL_VARIABLE(local_variable);
  T b = MakeParameter("parameters[0][0]");
  T c = a + b;
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  v_0 = local_variable;",
                                            "  v_1 = parameters[0][0];",
                                            "  v_2 = v_0 + v_1;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, OUTPUT_ASSIGNMENT) {
  double local_variable = 5.0;
  StartRecordingExpressions();
  T a = 1;
  T b = 0;
  MakeOutput(a, "residual[0]");
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  v_0 = 1;",
                                            "  v_1 = 0;",
                                            "  residual[0] = v_0;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, ASSIGNMENT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  T c = a;  // < This should not generate a line!
  a = b;
  a = a + b;  // < Create temporary + assignment
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_3;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_0 = v_1;",
                                            "  v_3 = v_0 + v_1;",
                                            "  v_0 = v_3;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, BINARY_ARITHMETIC_SIMPLE) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  T r1 = a + b;
  T r2 = a - b;
  T r3 = a * b;
  T r4 = a / b;
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  double v_3;",
                                            "  double v_4;",
                                            "  double v_5;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_0 + v_1;",
                                            "  v_3 = v_0 - v_1;",
                                            "  v_4 = v_0 * v_1;",
                                            "  v_5 = v_0 / v_1;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, BINARY_ARITHMETIC_COMPOUND) {
  // For each binary compound arithmetic operation, two lines are generated:
  //    - The actual operation assigning to a new temporary variable
  //    - An assignment from the temporary to the lhs
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  b += a;
  b -= a;
  b *= a;
  b /= a;
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  double v_4;",
                                            "  double v_6;",
                                            "  double v_8;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_1 + v_0;",
                                            "  v_1 = v_2;",
                                            "  v_4 = v_1 - v_0;",
                                            "  v_1 = v_4;",
                                            "  v_6 = v_1 * v_0;",
                                            "  v_1 = v_6;",
                                            "  v_8 = v_1 / v_0;",
                                            "  v_1 = v_8;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, UNARY_ARITHMETIC) {
  StartRecordingExpressions();
  T a = T(0);
  T r1 = -a;
  T r2 = +a;
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  v_0 = 0;",
                                            "  v_1 = -v_0;",
                                            "  v_2 = +v_0;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, BINARY_COMPARISON) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  auto r1 = a < b;
  auto r2 = a <= b;
  auto r3 = a > b;
  auto r4 = a >= b;
  auto r5 = a == b;
  auto r6 = a != b;
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  bool v_2;",
                                            "  bool v_3;",
                                            "  bool v_4;",
                                            "  bool v_5;",
                                            "  bool v_6;",
                                            "  bool v_7;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_0 < v_1;",
                                            "  v_3 = v_0 <= v_1;",
                                            "  v_4 = v_0 > v_1;",
                                            "  v_5 = v_0 >= v_1;",
                                            "  v_6 = v_0 == v_1;",
                                            "  v_7 = v_0 != v_1;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, LOGICAL_OPERATORS) {
  // Tests binary logical operators &&, || and the unary logical operator !
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  auto r1 = a < b;
  auto r2 = a <= b;

  auto r3 = r1 && r2;
  auto r4 = r1 || r2;
  auto r5 = !r1;

  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  bool v_2;",
                                            "  bool v_3;",
                                            "  bool v_4;",
                                            "  bool v_5;",
                                            "  bool v_6;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_0 < v_1;",
                                            "  v_3 = v_0 <= v_1;",
                                            "  v_4 = v_2 && v_3;",
                                            "  v_5 = v_2 || v_3;",
                                            "  v_6 = !v_2;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, FUNCTION_CALL) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);

  abs(a);
  acos(a);
  asin(a);
  atan(a);
  cbrt(a);
  ceil(a);
  cos(a);
  cosh(a);
  exp(a);
  exp2(a);
  floor(a);
  log(a);
  log2(a);
  sin(a);
  sinh(a);
  sqrt(a);
  tan(a);
  tanh(a);
  atan2(a, b);
  pow(a, b);

  auto graph = StopRecordingExpressions();

  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  double v_3;",
                                            "  double v_4;",
                                            "  double v_5;",
                                            "  double v_6;",
                                            "  double v_7;",
                                            "  double v_8;",
                                            "  double v_9;",
                                            "  double v_10;",
                                            "  double v_11;",
                                            "  double v_12;",
                                            "  double v_13;",
                                            "  double v_14;",
                                            "  double v_15;",
                                            "  double v_16;",
                                            "  double v_17;",
                                            "  double v_18;",
                                            "  double v_19;",
                                            "  double v_20;",
                                            "  double v_21;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = abs(v_0);",
                                            "  v_3 = acos(v_0);",
                                            "  v_4 = asin(v_0);",
                                            "  v_5 = atan(v_0);",
                                            "  v_6 = cbrt(v_0);",
                                            "  v_7 = ceil(v_0);",
                                            "  v_8 = cos(v_0);",
                                            "  v_9 = cosh(v_0);",
                                            "  v_10 = exp(v_0);",
                                            "  v_11 = exp2(v_0);",
                                            "  v_12 = floor(v_0);",
                                            "  v_13 = log(v_0);",
                                            "  v_14 = log2(v_0);",
                                            "  v_15 = sin(v_0);",
                                            "  v_16 = sinh(v_0);",
                                            "  v_17 = sqrt(v_0);",
                                            "  v_18 = tan(v_0);",
                                            "  v_19 = tanh(v_0);",
                                            "  v_20 = atan2(v_0, v_1);",
                                            "  v_21 = pow(v_0, v_1);",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, IF_SIMPLE) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  auto r1 = a < b;
  CERES_IF(r1) {}
  CERES_ELSE {}
  CERES_ENDIF;

  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  bool v_2;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_0 < v_1;",
                                            "  if (v_2) {",
                                            "  } else {",
                                            "  }",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, IF_ASSIGNMENT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);
  auto r1 = a < b;

  T result = 0;
  CERES_IF(r1) { result = 5.0; }
  CERES_ELSE { result = 6.0; }
  CERES_ENDIF;
  MakeOutput(result, "result");

  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  bool v_2;",
                                            "  double v_3;",
                                            "  double v_5;",
                                            "  double v_8;",
                                            "  double v_11;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = v_0 < v_1;",
                                            "  v_3 = 0;",
                                            "  if (v_2) {",
                                            "    v_5 = 5;",
                                            "    v_3 = v_5;",
                                            "  } else {",
                                            "    v_8 = 6;",
                                            "    v_3 = v_8;",
                                            "  }",
                                            "  result = v_3;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

TEST(CodeGenerator, IF_NESTED_ASSIGNMENT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(1);

  T result = 0;
  CERES_IF(a <= b) {
    result = 5.0;
    CERES_IF(a == b) { result = 7.0; }
    CERES_ENDIF;
  }
  CERES_ELSE { result = 6.0; }
  CERES_ENDIF;
  MakeOutput(result, "result");

  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  bool v_3;",
                                            "  double v_5;",
                                            "  bool v_7;",
                                            "  double v_9;",
                                            "  double v_13;",
                                            "  double v_16;",
                                            "  v_0 = 0;",
                                            "  v_1 = 1;",
                                            "  v_2 = 0;",
                                            "  v_3 = v_0 <= v_1;",
                                            "  if (v_3) {",
                                            "    v_5 = 5;",
                                            "    v_2 = v_5;",
                                            "    v_7 = v_0 == v_1;",
                                            "    if (v_7) {",
                                            "      v_9 = 7;",
                                            "      v_2 = v_9;",
                                            "    }",
                                            "  } else {",
                                            "    v_13 = 6;",
                                            "    v_2 = v_13;",
                                            "  }",
                                            "  result = v_2;",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

}  // namespace internal
}  // namespace ceres
