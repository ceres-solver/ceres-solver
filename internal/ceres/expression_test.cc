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

#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "ceres/jet.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(Expression, ConstructorAndAccessors) {
  Expression expr(ExpressionType::LOGICAL_NEGATION,
                  12345,
                  {1, 5, 8, 10},
                  "TestConstructor",
                  57.25);
  EXPECT_EQ(expr.type(), ExpressionType::LOGICAL_NEGATION);
  EXPECT_EQ(expr.lhs_id(), 12345);
  EXPECT_EQ(expr.arguments(), std::vector<ExpressionId>({1, 5, 8, 10}));
  EXPECT_EQ(expr.name(), "TestConstructor");
  EXPECT_EQ(expr.value(), 57.25);
}

TEST(Expression, CreateFunctions) {
  // clang-format off
  // The default constructor creates a NOP!
  EXPECT_EQ(Expression(), Expression(
            ExpressionType::NOP, kInvalidExpressionId, {}, "", 0));

  EXPECT_EQ(Expression::CreateCompileTimeConstant(72), Expression(
            ExpressionType::COMPILE_TIME_CONSTANT, kInvalidExpressionId, {}, "", 72));

  EXPECT_EQ(Expression::CreateInputAssignment("arguments[0][0]"), Expression(
            ExpressionType::INPUT_ASSIGNMENT, kInvalidExpressionId, {}, "arguments[0][0]", 0));

  EXPECT_EQ(Expression::CreateOutputAssignment(ExpressionId(5), "residuals[3]"), Expression(
            ExpressionType::OUTPUT_ASSIGNMENT, kInvalidExpressionId, {5}, "residuals[3]", 0));

  EXPECT_EQ(Expression::CreateAssignment(ExpressionId(3), ExpressionId(5)), Expression(
            ExpressionType::ASSIGNMENT, 3, {5}, "", 0));

  EXPECT_EQ(Expression::CreateBinaryArithmetic("+", ExpressionId(3),ExpressionId(5)), Expression(
            ExpressionType::BINARY_ARITHMETIC, kInvalidExpressionId, {3,5}, "+", 0));

  EXPECT_EQ(Expression::CreateUnaryArithmetic("-", ExpressionId(5)), Expression(
            ExpressionType::UNARY_ARITHMETIC, kInvalidExpressionId, {5}, "-", 0));

  EXPECT_EQ(Expression::CreateBinaryCompare("<",ExpressionId(3),ExpressionId(5)), Expression(
            ExpressionType::BINARY_COMPARISON, kInvalidExpressionId, {3,5}, "<", 0));

  EXPECT_EQ(Expression::CreateLogicalNegation(ExpressionId(5)), Expression(
            ExpressionType::LOGICAL_NEGATION, kInvalidExpressionId, {5}, "", 0));

  EXPECT_EQ(Expression::CreateFunctionCall("pow",{ExpressionId(3),ExpressionId(5)}), Expression(
            ExpressionType::FUNCTION_CALL, kInvalidExpressionId, {3,5}, "pow", 0));

  EXPECT_EQ(Expression::CreateIf(ExpressionId(5)), Expression(
            ExpressionType::IF, kInvalidExpressionId, {5}, "", 0));

  EXPECT_EQ(Expression::CreateElse(), Expression(
            ExpressionType::ELSE, kInvalidExpressionId, {}, "", 0));

  EXPECT_EQ(Expression::CreateEndIf(), Expression(
            ExpressionType::ENDIF, kInvalidExpressionId, {}, "", 0));
  // clang-format on
}

TEST(Expression, IsArithmeticExpression) {
  ASSERT_TRUE(
      Expression::CreateCompileTimeConstant(5).IsArithmeticExpression());
  ASSERT_TRUE(
      Expression::CreateFunctionCall("pow", {3, 5}).IsArithmeticExpression());
  // Logical expression are also arithmetic!
  ASSERT_TRUE(
      Expression::CreateBinaryCompare("<", 3, 5).IsArithmeticExpression());
  ASSERT_FALSE(Expression::CreateIf(5).IsArithmeticExpression());
  ASSERT_FALSE(Expression::CreateEndIf().IsArithmeticExpression());
  ASSERT_FALSE(Expression().IsArithmeticExpression());
}

TEST(Expression, IsControlExpression) {
  // In the current implementation this is the exact opposite of
  // IsArithmeticExpression.
  ASSERT_FALSE(Expression::CreateCompileTimeConstant(5).IsControlExpression());
  ASSERT_FALSE(
      Expression::CreateFunctionCall("pow", {3, 5}).IsControlExpression());
  ASSERT_FALSE(
      Expression::CreateBinaryCompare("<", 3, 5).IsControlExpression());
  ASSERT_TRUE(Expression::CreateIf(5).IsControlExpression());
  ASSERT_TRUE(Expression::CreateEndIf().IsControlExpression());
  ASSERT_TRUE(Expression().IsControlExpression());
}

TEST(Expression, IsCompileTimeConstantAndEqualTo) {
  ASSERT_TRUE(
      Expression::CreateCompileTimeConstant(5).IsCompileTimeConstantAndEqualTo(
          5));
  ASSERT_FALSE(
      Expression::CreateCompileTimeConstant(3).IsCompileTimeConstantAndEqualTo(
          5));
  ASSERT_FALSE(Expression::CreateBinaryCompare("<", 3, 5)
                   .IsCompileTimeConstantAndEqualTo(5));
}

TEST(Expression, IsReplaceableBy) {
  // Create 2 identical expression
  auto expr1 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  auto expr2 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  // They are idendical and of course replaceable
  ASSERT_EQ(expr1, expr2);
  ASSERT_EQ(expr2, expr1);
  ASSERT_TRUE(expr1.IsReplaceableBy(expr2));
  ASSERT_TRUE(expr2.IsReplaceableBy(expr1));

  // Give them different left hand sides
  expr1.set_lhs_id(72);
  expr2.set_lhs_id(42);

  // v_72 = v_3 + v_5
  // v_42 = v_3 + v_5
  // -> They should be replaceable by each other

  ASSERT_NE(expr1, expr2);
  ASSERT_NE(expr2, expr1);

  ASSERT_TRUE(expr1.IsReplaceableBy(expr2));
  ASSERT_TRUE(expr2.IsReplaceableBy(expr1));

  // A slightly differnt expression with the argument flipped
  auto expr3 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(5), ExpressionId(3));

  ASSERT_NE(expr1, expr3);
  ASSERT_FALSE(expr1.IsReplaceableBy(expr3));
}

TEST(Expression, DirectlyDependsOn) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T unused(6);
  T a(2), b(3);
  T c = a + b;
  T d = c + a;

  auto graph = StopRecordingExpressions();

  ASSERT_FALSE(graph.ExpressionForId(a.id).DirectlyDependsOn(unused.id));
  ASSERT_TRUE(graph.ExpressionForId(c.id).DirectlyDependsOn(a.id));
  ASSERT_TRUE(graph.ExpressionForId(c.id).DirectlyDependsOn(b.id));
  ASSERT_TRUE(graph.ExpressionForId(d.id).DirectlyDependsOn(a.id));
  ASSERT_FALSE(graph.ExpressionForId(d.id).DirectlyDependsOn(b.id));
  ASSERT_TRUE(graph.ExpressionForId(d.id).DirectlyDependsOn(c.id));
}

TEST(Expression, Ternary) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a(2);                   // 0
  T b(3);                   // 1
  auto c = a < b;           // 2
  T d = Ternary(c, a, b);   // 3
  MakeOutput(d, "result");  // 4
  auto graph = StopRecordingExpressions();

  EXPECT_EQ(graph.Size(), 5);

  // Expected code
  //   v_0 = 2;
  //   v_1 = 3;
  //   v_2 = v_0 < v_1;
  //   v_3 = Ternary(v_2, v_0, v_1);
  //   result = v_3;

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   0,      {},        "",  2});
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   1,      {},        "",  3});
  reference.InsertBack({     ExpressionType::BINARY_COMPARISON,   2,   {0,1},       "<",  0});
  reference.InsertBack({         ExpressionType::FUNCTION_CALL,   3, {2,0,1}, "Ternary",  0});
  reference.InsertBack({     ExpressionType::OUTPUT_ASSIGNMENT,   4,     {3},  "result",  0});
  // clang-format on
  EXPECT_EQ(reference, graph);
}

TEST(Expression, Assignment) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a = 1;
  T b = 2;
  b = a;
  auto graph = StopRecordingExpressions();

  EXPECT_EQ(graph.Size(), 3);

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   0,       {},        "",  1});
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   1,       {},        "",  2});
  reference.InsertBack({            ExpressionType::ASSIGNMENT,   1,      {0},        "",  0});
  // clang-format on
  EXPECT_EQ(reference, graph);
}

TEST(Expression, AssignmentCreate) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a = 2;
  T b = a;
  auto graph = StopRecordingExpressions();

  EXPECT_EQ(graph.Size(), 2);

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   0,       {},        "",  2});
  reference.InsertBack({            ExpressionType::ASSIGNMENT,   1,      {0},        "",  0});
  // clang-format on
  EXPECT_EQ(reference, graph);
}

TEST(Expression, MoveAssignmentCreate) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a = 1;
  T b = std::move(a);
  auto graph = StopRecordingExpressions();

  EXPECT_EQ(graph.Size(), 1);

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   0,      {},        "",  1});
  // clang-format on
  EXPECT_EQ(reference, graph);
}

TEST(Expression, MoveAssignment) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a = 1;
  T b = 2;
  b = std::move(a);
  auto graph = StopRecordingExpressions();

  EXPECT_EQ(graph.Size(), 3);

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   0,       {},        "",  1});
  reference.InsertBack({ ExpressionType::COMPILE_TIME_CONSTANT,   1,       {},        "",  2});
  reference.InsertBack({            ExpressionType::ASSIGNMENT,   1,      {0},        "",  0});
  // clang-format on
  EXPECT_EQ(reference, graph);
}

}  // namespace internal
}  // namespace ceres
