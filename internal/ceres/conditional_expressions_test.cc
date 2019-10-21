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

#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

void TestSingleExpressionGraph(const ExpressionGraph& graph,
                               ExpressionId id,
                               ExpressionId lhs,
                               ExpressionType type,
                               std::vector<ExpressionId> args) {
  EXPECT_LT(id, graph.Size());
  auto& expr = graph.ExpressionForId(id);
  EXPECT_EQ(expr.Id(), lhs);
  EXPECT_EQ(expr.Type(), type);
  EXPECT_EQ(expr.Arguments(), args);
}

#define TEST_EXPR(_id, _type, ...) \
  TestSingleExpressionGraph(       \
      graph, id++, _id, ExpressionType::_type, {__VA_ARGS__})

TEST(Expression, Conditionals) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T result;
  T a(2);
  T b(3);
  auto c = a < b;
  CERES_IF(c) { result = a + b; }
  CERES_ELSE { result = a - b; }
  CERES_ENDIF
  result += a;
  auto graph = StopRecordingExpressions();

  // Expected code
  //   v_0 = 2;
  //   v_1 = 3;
  //   v_2 = v_0 < v_1;
  //   if(v_2);
  //     v_4 = v_0 + v_1;
  //   else
  //     v_6 = v_0 - v_1;
  //     v_4 = v_6
  //   endif
  //   v_9 = v_4 + v_0;
  //   v_4 = v_9;

  ExpressionId id = 0;
  TEST_EXPR(0, COMPILE_TIME_CONSTANT);
  TEST_EXPR(1, COMPILE_TIME_CONSTANT);
  TEST_EXPR(2, BINARY_COMPARISON, 0, 1);
  TEST_EXPR(-1, IF, 2);
  TEST_EXPR(4, PLUS, 0, 1);
  TEST_EXPR(-1, ELSE);
  TEST_EXPR(6, MINUS, 0, 1);
  TEST_EXPR(4, ASSIGNMENT, 6);
  TEST_EXPR(-1, ENDIF);
  TEST_EXPR(9, PLUS, 4, 0);
  TEST_EXPR(4, ASSIGNMENT, 9);

  // Variables after execution:
  //
  // a      <=> v_0
  // b      <=> v_1
  // result <=> v_4
  EXPECT_EQ(a.id, 0);
  EXPECT_EQ(b.id, 1);
  EXPECT_EQ(result.id, 4);

  // a and b are in ssa-form.
  // result is not ssa because it is assigned multiple times.
  ASSERT_TRUE(graph.ExpressionForId(a.id).Ssa());
  ASSERT_TRUE(graph.ExpressionForId(b.id).Ssa());
  ASSERT_FALSE(graph.ExpressionForId(result.id).Ssa());
}
// Todo: remaining functions of Expression

}  // namespace internal
}  // namespace ceres
