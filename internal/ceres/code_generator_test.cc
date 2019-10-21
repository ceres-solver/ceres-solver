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

TEST(CodeGenerator, FUNCTION_CALL) {
  StartRecordingExpressions();
  T a = T(0);
  sin(a);
  // todo: remaining functions are defined in the expressionRef patch.
  sin(a);
  sin(a);
  auto graph = StopRecordingExpressions();
  std::vector<std::string> expected_code = {"{",
                                            "  double v_0;",
                                            "  double v_1;",
                                            "  double v_2;",
                                            "  double v_3;",
                                            "  v_0 = 0;",
                                            "  v_1 = sin(v_0);",
                                            "  v_2 = sin(v_0);",
                                            "  v_3 = sin(v_0);",
                                            "}"};
  GenerateAndCheck(graph, expected_code);
}

// TODO: Requires Patch ExpressionRef fix
//
// TEST(CodeGenerator, RUNTIME_CONSTANT) {
//  double local_variable = 5.0;
//  StartRecordingExpressions();
//  T a = CERES_EXPRESSION_RUNTIME_CONSTANT(local_variable);
//  auto graph = StopRecordingExpressions();
//  std::vector<std::string> expected_code = {
//      "{", "  double v_0;", "  v_0 = local_variable;", "}"};
//  GenerateAndCheck(graph, expected_code);
//}

// TODO: Tests for remaining ExpressionTypes

}  // namespace internal
}  // namespace ceres
