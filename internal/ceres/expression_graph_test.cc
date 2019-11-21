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
// Test expression creation and logic.

#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(ExpressionGraph, Creation) {
  using T = ExpressionRef;
  ExpressionGraph graph;

  StartRecordingExpressions();
  graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 0);

  StartRecordingExpressions();
  T a(1);
  T b(2);
  T c = a + b;
  graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 3);
}

TEST(ExpressionGraph, Dependencies) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T unused(6);
  T a(2), b(3);
  T c = a + b;
  T d = c + a;

  auto tree = StopRecordingExpressions();

  // Recursive dependency check
  ASSERT_TRUE(tree.DependsOn(d.id, c.id));
  ASSERT_TRUE(tree.DependsOn(d.id, a.id));
  ASSERT_TRUE(tree.DependsOn(d.id, b.id));
  ASSERT_FALSE(tree.DependsOn(d.id, unused.id));
}

TEST(ExpressionGraph, InsertExpression_UpdateReferences) {
  // This test checks if references to shifted expressions are updated
  // accordingly.
  using T = ExpressionRef;
  StartRecordingExpressions();
  T a(2);       // 0
  T b(3);       // 1
  T c = a + b;  // 2
  auto graph = StopRecordingExpressions();

  // Test if 'a' and 'c' are actually at location 0 and 2
  auto& a_expr = graph.ExpressionForId(0);
  EXPECT_EQ(a_expr.type(), ExpressionType::COMPILE_TIME_CONSTANT);
  EXPECT_EQ(a_expr.value(), 2);

  // At this point 'c' should have 0 and 1 as arguments.
  auto& c_expr = graph.ExpressionForId(2);
  EXPECT_EQ(c_expr.type(), ExpressionType::BINARY_ARITHMETIC);
  EXPECT_EQ(c_expr.arguments()[0], 0);
  EXPECT_EQ(c_expr.arguments()[1], 1);

  // We insert at the beginning, which shifts everything by one spot.
  graph.InsertExpression(
      0, ExpressionType::COMPILE_TIME_CONSTANT, 0, {}, "", 10.2);

  // Test if 'a' and 'c' are actually at location 1 and 3
  auto& a_expr2 = graph.ExpressionForId(1);
  EXPECT_EQ(a_expr2.type(), ExpressionType::COMPILE_TIME_CONSTANT);
  EXPECT_EQ(a_expr2.value(), 2);

  // At this point 'c' should have 1 and 2 as arguments.
  auto& c_expr2 = graph.ExpressionForId(3);
  EXPECT_EQ(c_expr2.type(), ExpressionType::BINARY_ARITHMETIC);
  EXPECT_EQ(c_expr2.arguments()[0], 1);
  EXPECT_EQ(c_expr2.arguments()[1], 2);
}

TEST(ExpressionGraph, InsertExpression) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  {
    T a(2);                   // 0
    T b(3);                   // 1
    T five = 5;               // 2
    T tmp = a + five;         // 3
    a = tmp;                  // 4
    T c = a + b;              // 5
    T d = a * b;              // 6
    T e = c + d;              // 7
    MakeOutput(e, "result");  // 8
  }
  auto reference = StopRecordingExpressions();
  EXPECT_EQ(reference.Size(), 9);

  StartRecordingExpressions();

  {
    // The expressions 2,3,4 from above are missing.
    T a(2);                   // 0
    T b(3);                   // 1
    T c = a + b;              // 2
    T d = a * b;              // 3
    T e = c + d;              // 4
    MakeOutput(e, "result");  // 5
  }

  auto graph1 = StopRecordingExpressions();
  EXPECT_EQ(graph1.Size(), 6);
  ASSERT_FALSE(reference == graph1);

  // We manually insert the 3 missing expressions
  // clang-format off
  graph1.InsertExpression(2, ExpressionType::COMPILE_TIME_CONSTANT, 2,     {},   "",  5);
  graph1.InsertExpression(3,     ExpressionType::BINARY_ARITHMETIC, 3, {0, 2},  "+",  0);
  graph1.InsertExpression(4,            ExpressionType::ASSIGNMENT, 0,    {3},   "",  0);
  // clang-format on

  // Now the graphs are identical!
  EXPECT_EQ(graph1.Size(), 9);
  ASSERT_TRUE(reference == graph1);
}

}  // namespace internal
}  // namespace ceres
