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
// This file tests the ExpressionGraph class. This test depends on the
// correctness of Expression.
//
#include "ceres/codegen/internal/expression_graph.h"

#include "ceres/codegen/internal/expression.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(ExpressionGraph, Size) {
  ExpressionGraph graph;
  EXPECT_EQ(graph.Size(), 0);
  // Insert 3 NOPs
  graph.InsertBack(Expression());
  graph.InsertBack(Expression());
  graph.InsertBack(Expression());
  EXPECT_EQ(graph.Size(), 3);
}

TEST(ExpressionGraph, Recording) {
  EXPECT_EQ(GetCurrentExpressionGraph(), nullptr);
  StartRecordingExpressions();
  EXPECT_NE(GetCurrentExpressionGraph(), nullptr);
  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph, ExpressionGraph());
  EXPECT_EQ(GetCurrentExpressionGraph(), nullptr);
}

TEST(ExpressionGraph, InsertBackControl) {
  // Control expression are inserted to the back without any modifications.
  auto expr1 = Expression::CreateIf(ExpressionId(0));
  auto expr2 = Expression::CreateElse();
  auto expr3 = Expression::CreateEndIf();

  ExpressionGraph graph;
  graph.InsertBack(expr1);
  graph.InsertBack(expr2);
  graph.InsertBack(expr3);

  EXPECT_EQ(graph.Size(), 3);
  EXPECT_EQ(graph.ExpressionForId(0), expr1);
  EXPECT_EQ(graph.ExpressionForId(1), expr2);
  EXPECT_EQ(graph.ExpressionForId(2), expr3);
}

TEST(ExpressionGraph, InsertBackNewVariable) {
  // If an arithmetic expression with lhs=kinvalidValue is inserted in the back,
  // then a new variable name is created and set to the lhs_id.
  auto expr1 = Expression::CreateCompileTimeConstant(42);
  auto expr2 = Expression::CreateCompileTimeConstant(10);
  auto expr3 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(0), ExpressionId(1));

  ExpressionGraph graph;
  graph.InsertBack(expr1);
  graph.InsertBack(expr2);
  graph.InsertBack(expr3);
  EXPECT_EQ(graph.Size(), 3);

  // The ExpressionGraph has a copy of the inserted expression with the correct
  // lhs_ids. We set them here manually for comparision.
  expr1.set_lhs_id(0);
  expr2.set_lhs_id(1);
  expr3.set_lhs_id(2);
  EXPECT_EQ(graph.ExpressionForId(0), expr1);
  EXPECT_EQ(graph.ExpressionForId(1), expr2);
  EXPECT_EQ(graph.ExpressionForId(2), expr3);
}

TEST(ExpressionGraph, InsertBackExistingVariable) {
  auto expr1 = Expression::CreateCompileTimeConstant(42);
  auto expr2 = Expression::CreateCompileTimeConstant(10);
  auto expr3 = Expression::CreateAssignment(1, 0);

  ExpressionGraph graph;
  graph.InsertBack(expr1);
  graph.InsertBack(expr2);
  graph.InsertBack(expr3);
  EXPECT_EQ(graph.Size(), 3);

  expr1.set_lhs_id(0);
  expr2.set_lhs_id(1);
  expr3.set_lhs_id(1);
  EXPECT_EQ(graph.ExpressionForId(0), expr1);
  EXPECT_EQ(graph.ExpressionForId(1), expr2);
  EXPECT_EQ(graph.ExpressionForId(2), expr3);
}

TEST(ExpressionGraph, DependsOn) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(10));
  graph.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(0), ExpressionId(1)));
  graph.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(2), ExpressionId(0)));

  // Code:
  // v_0 = 42
  // v_1 = 10
  // v_2 = v_0 + v_1
  // v_3 = v_2 + v_0

  // Direct dependencies dependency check
  ASSERT_TRUE(graph.DependsOn(2, 0));
  ASSERT_TRUE(graph.DependsOn(2, 1));
  ASSERT_TRUE(graph.DependsOn(3, 2));
  ASSERT_TRUE(graph.DependsOn(3, 0));
  ASSERT_FALSE(graph.DependsOn(1, 0));
  ASSERT_FALSE(graph.DependsOn(1, 1));
  ASSERT_FALSE(graph.DependsOn(2, 3));

  // Recursive test
  ASSERT_TRUE(graph.DependsOn(3, 1));
}

TEST(ExpressionGraph, FindMatchingEndif) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(1));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateElse());
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateElse());
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateIf(2));  // < if without matching endif
  EXPECT_EQ(graph.Size(), 12);

  // Code              <id>
  // v_0 = 1            0
  // v_1 = 2            1
  // v_2 = v_0 < v_1    2
  // IF (v_2)           3
  //   IF (v_2)         4
  //   ELSE             5
  //   ENDIF            6
  // ELSE               7
  //   IF (v_2)         8
  //   ENDIF            9
  // ENDIF              10
  // IF(v_2)            11

  EXPECT_EQ(graph.FindMatchingEndif(3), 10);
  EXPECT_EQ(graph.FindMatchingEndif(4), 6);
  EXPECT_EQ(graph.FindMatchingEndif(8), 9);
  EXPECT_EQ(graph.FindMatchingEndif(11), kInvalidExpressionId);
}

TEST(ExpressionGraph, FindMatchingElse) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(1));
  graph.InsertBack(Expression::CreateCompileTimeConstant(2));
  graph.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateElse());
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateElse());
  graph.InsertBack(Expression::CreateIf(2));
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateEndIf());
  graph.InsertBack(Expression::CreateIf(2));  // < if without matching endif
  EXPECT_EQ(graph.Size(), 12);

  // Code              <id>
  // v_0 = 1            0
  // v_1 = 2            1
  // v_2 = v_0 < v_1    2
  // IF (v_2)           3
  //   IF (v_2)         4
  //   ELSE             5
  //   ENDIF            6
  // ELSE               7
  //   IF (v_2)         8
  //   ENDIF            9
  // ENDIF              10
  // IF(v_2)            11

  EXPECT_EQ(graph.FindMatchingElse(3), 7);
  EXPECT_EQ(graph.FindMatchingElse(4), 5);
  EXPECT_EQ(graph.FindMatchingElse(8), kInvalidExpressionId);
  EXPECT_EQ(graph.FindMatchingEndif(11), kInvalidExpressionId);
}

TEST(ExpressionGraph, InsertExpression_UpdateReferences) {
  // This test checks if references to shifted expressions are updated
  // accordingly.
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(10));
  graph.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(0), ExpressionId(1)));
  // Code:
  // v_0 = 42
  // v_1 = 10
  // v_2 = v_0 + v_1

  // Insert another compile time constant at the beginning
  graph.Insert(0, Expression::CreateCompileTimeConstant(5));
  // This should shift all indices like this:
  // v_0 = 5
  // v_1 = 42
  // v_2 = 10
  // v_3 = v_1 + v_2

  // Test by inserting it in the correct order
  ExpressionGraph ref;
  ref.InsertBack(Expression::CreateCompileTimeConstant(5));
  ref.InsertBack(Expression::CreateCompileTimeConstant(42));
  ref.InsertBack(Expression::CreateCompileTimeConstant(10));
  ref.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(1), ExpressionId(2)));
  EXPECT_EQ(graph.Size(), ref.Size());
  EXPECT_EQ(graph, ref);
}

TEST(ExpressionGraph, Erase) {
  // This test checks if references to shifted expressions are updated
  // accordingly.
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(10));
  graph.InsertBack(Expression::CreateCompileTimeConstant(3));
  graph.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(0), ExpressionId(2)));
  // Code:
  // v_0 = 42
  // v_1 = 10
  // v_2 = 3
  // v_3 = v_0 + v_2

  // Erase the unused expression v_1 = 10
  graph.Erase(1);
  // This should shift all indices like this:
  // v_0 = 42
  // v_1 = 3
  // v_2 = v_0 + v_1

  // Test by inserting it in the correct order
  ExpressionGraph ref;
  ref.InsertBack(Expression::CreateCompileTimeConstant(42));
  ref.InsertBack(Expression::CreateCompileTimeConstant(3));
  ref.InsertBack(Expression::CreateBinaryArithmetic(
      "+", ExpressionId(0), ExpressionId(1)));
  EXPECT_EQ(graph.Size(), ref.Size());
  EXPECT_EQ(graph, ref);
}

}  // namespace internal
}  // namespace ceres
