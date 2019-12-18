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
// This file tests the Expression class. For each member function one test is
// included here.
//
#include "ceres/codegen/internal/expression.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(Expression, ConstructorAndAccessors) {
  Expression expr(ExpressionType::LOGICAL_NEGATION,
                  ExpressionReturnType::BOOLEAN,
                  12345,
                  {1, 5, 8, 10},
                  "TestConstructor",
                  57.25);
  EXPECT_EQ(expr.type(), ExpressionType::LOGICAL_NEGATION);
  EXPECT_EQ(expr.return_type(), ExpressionReturnType::BOOLEAN);
  EXPECT_EQ(expr.lhs_id(), 12345);
  EXPECT_EQ(expr.arguments(), std::vector<ExpressionId>({1, 5, 8, 10}));
  EXPECT_EQ(expr.name(), "TestConstructor");
  EXPECT_EQ(expr.value(), 57.25);
}

TEST(Expression, CreateFunctions) {
  // The default constructor creates a NOP!
  EXPECT_EQ(Expression(),
            Expression(ExpressionType::NOP,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {},
                       "",
                       0));

  EXPECT_EQ(Expression::CreateCompileTimeConstant(72),
            Expression(ExpressionType::COMPILE_TIME_CONSTANT,
                       ExpressionReturnType::SCALAR,
                       kInvalidExpressionId,
                       {},
                       "",
                       72));

  EXPECT_EQ(Expression::CreateInputAssignment("arguments[0][0]"),
            Expression(ExpressionType::INPUT_ASSIGNMENT,
                       ExpressionReturnType::SCALAR,
                       kInvalidExpressionId,
                       {},
                       "arguments[0][0]",
                       0));

  EXPECT_EQ(Expression::CreateOutputAssignment(ExpressionId(5), "residuals[3]"),
            Expression(ExpressionType::OUTPUT_ASSIGNMENT,
                       ExpressionReturnType::SCALAR,
                       kInvalidExpressionId,
                       {5},
                       "residuals[3]",
                       0));

  EXPECT_EQ(Expression::CreateAssignment(ExpressionId(3), ExpressionId(5)),
            Expression(ExpressionType::ASSIGNMENT,
                       ExpressionReturnType::SCALAR,
                       3,
                       {5},
                       "",
                       0));

  EXPECT_EQ(
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5)),
      Expression(ExpressionType::BINARY_ARITHMETIC,
                 ExpressionReturnType::SCALAR,
                 kInvalidExpressionId,
                 {3, 5},
                 "+",
                 0));

  EXPECT_EQ(Expression::CreateUnaryArithmetic("-", ExpressionId(5)),
            Expression(ExpressionType::UNARY_ARITHMETIC,
                       ExpressionReturnType::SCALAR,
                       kInvalidExpressionId,
                       {5},
                       "-",
                       0));

  EXPECT_EQ(
      Expression::CreateBinaryCompare("<", ExpressionId(3), ExpressionId(5)),
      Expression(ExpressionType::BINARY_COMPARISON,
                 ExpressionReturnType::BOOLEAN,
                 kInvalidExpressionId,
                 {3, 5},
                 "<",
                 0));

  EXPECT_EQ(Expression::CreateLogicalNegation(ExpressionId(5)),
            Expression(ExpressionType::LOGICAL_NEGATION,
                       ExpressionReturnType::BOOLEAN,
                       kInvalidExpressionId,
                       {5},
                       "",
                       0));

  EXPECT_EQ(Expression::CreateScalarFunctionCall(
                "pow", {ExpressionId(3), ExpressionId(5)}),
            Expression(ExpressionType::FUNCTION_CALL,
                       ExpressionReturnType::SCALAR,
                       kInvalidExpressionId,
                       {3, 5},
                       "pow",
                       0));

  EXPECT_EQ(
      Expression::CreateLogicalFunctionCall("isfinite", {ExpressionId(3)}),
      Expression(ExpressionType::FUNCTION_CALL,
                 ExpressionReturnType::BOOLEAN,
                 kInvalidExpressionId,
                 {3},
                 "isfinite",
                 0));

  EXPECT_EQ(Expression::CreateIf(ExpressionId(5)),
            Expression(ExpressionType::IF,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {5},
                       "",
                       0));

  EXPECT_EQ(Expression::CreateElse(),
            Expression(ExpressionType::ELSE,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {},
                       "",
                       0));

  EXPECT_EQ(Expression::CreateEndIf(),
            Expression(ExpressionType::ENDIF,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {},
                       "",
                       0));

  EXPECT_EQ(Expression::CreateComment("Test"),
            Expression(ExpressionType::COMMENT,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {},
                       "Test",
                       0));
}

TEST(Expression, IsArithmeticExpression) {
  ASSERT_TRUE(
      Expression::CreateCompileTimeConstant(5).IsArithmeticExpression());
  ASSERT_TRUE(Expression::CreateScalarFunctionCall("pow", {3, 5})
                  .IsArithmeticExpression());
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
  ASSERT_FALSE(Expression::CreateScalarFunctionCall("pow", {3, 5})
                   .IsControlExpression());
  ASSERT_FALSE(
      Expression::CreateBinaryCompare("<", 3, 5).IsControlExpression());
  ASSERT_TRUE(Expression::CreateIf(5).IsControlExpression());
  ASSERT_TRUE(Expression::CreateEndIf().IsControlExpression());
  ASSERT_TRUE(Expression::CreateComment("Test").IsControlExpression());
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

TEST(Expression, Replace) {
  auto expr1 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));
  expr1.set_lhs_id(13);

  auto expr2 =
      Expression::CreateAssignment(kInvalidExpressionId, ExpressionId(7));

  // We replace the arithmetic expr1 by an assignment from the variable 7. This
  // is the typical usecase in subexpression elimination.
  expr1.Replace(expr2);

  // expr1 should now be an assignment from 7 to 13
  EXPECT_EQ(expr1,
            Expression(ExpressionType::ASSIGNMENT,
                       ExpressionReturnType::SCALAR,
                       13,
                       {7},
                       "",
                       0));
}

TEST(Expression, DirectlyDependsOn) {
  auto expr1 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  ASSERT_TRUE(expr1.DirectlyDependsOn(ExpressionId(3)));
  ASSERT_TRUE(expr1.DirectlyDependsOn(ExpressionId(5)));
  ASSERT_FALSE(expr1.DirectlyDependsOn(ExpressionId(kInvalidExpressionId)));
  ASSERT_FALSE(expr1.DirectlyDependsOn(ExpressionId(42)));
}
TEST(Expression, MakeNop) {
  auto expr1 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  expr1.MakeNop();

  EXPECT_EQ(expr1,
            Expression(ExpressionType::NOP,
                       ExpressionReturnType::VOID,
                       kInvalidExpressionId,
                       {},
                       "",
                       0));
}

TEST(Expression, IsSemanticallyEquivalentTo) {
  // Create 2 identical expression
  auto expr1 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  auto expr2 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(5));

  ASSERT_TRUE(expr1.IsSemanticallyEquivalentTo(expr1));
  ASSERT_TRUE(expr1.IsSemanticallyEquivalentTo(expr2));

  auto expr3 =
      Expression::CreateBinaryArithmetic("+", ExpressionId(3), ExpressionId(8));

  ASSERT_TRUE(expr1.IsSemanticallyEquivalentTo(expr3));

  auto expr4 =
      Expression::CreateBinaryArithmetic("-", ExpressionId(3), ExpressionId(5));

  ASSERT_FALSE(expr1.IsSemanticallyEquivalentTo(expr4));
}

}  // namespace internal
}  // namespace ceres
