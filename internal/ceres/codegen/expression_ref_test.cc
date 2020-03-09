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
// This file tests the ExpressionRef class. This test depends on the
// correctness of Expression and ExpressionGraph.
//
#define CERES_CODEGEN

#include "ceres/codegen/internal/expression_ref.h"

#include "ceres/codegen/internal/expression_graph.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using T = ExpressionRef;

TEST(ExpressionRef, COMPILE_TIME_CONSTANT) {
  StartRecordingExpressions();
  T a = T(0);
  T b = T(123.5);
  T c = T(1 + 1);
  T d;  // Uninitialized variables are also compile time constants
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(0));
  reference.InsertBack(Expression::CreateCompileTimeConstant(123.5));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateCompileTimeConstant(0));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, INPUT_ASSIGNMENT) {
  double local_variable = 5.0;
  StartRecordingExpressions();
  T a = CERES_LOCAL_VARIABLE(T, local_variable);
  T b = MakeParameter("parameters[0][0]");
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateInputAssignment("local_variable"));
  reference.InsertBack(Expression::CreateInputAssignment("parameters[0][0]"));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, OUTPUT_ASSIGNMENT) {
  StartRecordingExpressions();
  T a = 1;
  T b = 0;
  MakeOutput(a, "residual[0]");
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(0));
  reference.InsertBack(Expression::CreateOutputAssignment(0, "residual[0]"));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, Assignment) {
  StartRecordingExpressions();
  T a = 1;
  T b = 2;
  b = a;
  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 3);

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateAssignment(1, 0));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, AssignmentCreate) {
  StartRecordingExpressions();
  T a = 2;
  T b = a;
  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 2);

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateAssignment(kInvalidExpressionId, 0));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, MoveAssignment) {
  StartRecordingExpressions();
  T a = 1;
  T b = 2;
  b = std::move(a);
  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 3);

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateAssignment(1, 0));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, BINARY_ARITHMETIC_SIMPLE) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  T r1 = a + b;
  T r2 = a - b;
  T r3 = a * b;
  T r4 = a / b;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryArithmetic("+", 0, 1));
  reference.InsertBack(Expression::CreateBinaryArithmetic("-", 0, 1));
  reference.InsertBack(Expression::CreateBinaryArithmetic("*", 0, 1));
  reference.InsertBack(Expression::CreateBinaryArithmetic("/", 0, 1));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, BINARY_ARITHMETIC_NESTED) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  T r1 = b - a * (a + b) / a;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryArithmetic("+", 0, 1));
  reference.InsertBack(Expression::CreateBinaryArithmetic("*", 0, 2));
  reference.InsertBack(Expression::CreateBinaryArithmetic("/", 3, 0));
  reference.InsertBack(Expression::CreateBinaryArithmetic("-", 1, 4));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, BINARY_ARITHMETIC_COMPOUND) {
  // For each binary compound arithmetic operation, two lines are generated:
  //    - The actual operation assigning to a new temporary variable
  //    - An assignment from the temporary to the lhs
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  a += b;
  a -= b;
  a *= b;
  a /= b;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryArithmetic("+", 0, 1));
  reference.InsertBack(Expression::CreateAssignment(0, 2));
  reference.InsertBack(Expression::CreateBinaryArithmetic("-", 0, 1));
  reference.InsertBack(Expression::CreateAssignment(0, 4));
  reference.InsertBack(Expression::CreateBinaryArithmetic("*", 0, 1));
  reference.InsertBack(Expression::CreateAssignment(0, 6));
  reference.InsertBack(Expression::CreateBinaryArithmetic("/", 0, 1));
  reference.InsertBack(Expression::CreateAssignment(0, 8));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, UNARY_ARITHMETIC) {
  StartRecordingExpressions();
  T a = T(1);
  T r1 = -a;
  T r2 = +a;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateUnaryArithmetic("-", 0));
  reference.InsertBack(Expression::CreateUnaryArithmetic("+", 0));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, BINARY_COMPARISON) {
  using BOOL = ComparisonExpressionRef;
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  BOOL r1 = a < b;
  BOOL r2 = a <= b;
  BOOL r3 = a > b;
  BOOL r4 = a >= b;
  BOOL r5 = a == b;
  BOOL r6 = a != b;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("<=", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare(">", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare(">=", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("==", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("!=", 0, 1));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, LOGICAL_OPERATORS) {
  using BOOL = ComparisonExpressionRef;
  // Tests binary logical operators &&, || and the unary logical operator !
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  BOOL r1 = a < b;
  BOOL r2 = a <= b;
  BOOL r3 = r1 && r2;
  BOOL r4 = r1 || r2;
  BOOL r5 = !r1;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("<=", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("&&", 2, 3));
  reference.InsertBack(Expression::CreateBinaryCompare("||", 2, 3));
  reference.InsertBack(Expression::CreateLogicalNegation(2));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, SCALAR_FUNCTION_CALL) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
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

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::abs", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::acos", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::asin", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::atan", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::cbrt", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::ceil", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::cos", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::cosh", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::exp", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::exp2", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::floor", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::log", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::log2", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::sin", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::sinh", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::sqrt", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::tan", {0}));
  reference.InsertBack(Expression::CreateScalarFunctionCall("std::tanh", {0}));
  reference.InsertBack(
      Expression::CreateScalarFunctionCall("std::atan2", {0, 1}));
  reference.InsertBack(
      Expression::CreateScalarFunctionCall("std::pow", {0, 1}));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, LOGICAL_FUNCTION_CALL) {
  StartRecordingExpressions();
  T a = T(1);
  isfinite(a);
  isinf(a);
  isnan(a);
  isnormal(a);
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(
      Expression::CreateLogicalFunctionCall("std::isfinite", {0}));
  reference.InsertBack(
      Expression::CreateLogicalFunctionCall("std::isinf", {0}));
  reference.InsertBack(
      Expression::CreateLogicalFunctionCall("std::isnan", {0}));
  reference.InsertBack(
      Expression::CreateLogicalFunctionCall("std::isnormal", {0}));
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, IF) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  auto r1 = a < b;
  CERES_IF(r1) {}
  CERES_ENDIF;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  reference.InsertBack(Expression::CreateIf(2));
  reference.InsertBack(Expression::CreateEndIf());
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, IF_ELSE) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  auto r1 = a < b;
  CERES_IF(r1) {}
  CERES_ELSE {}
  CERES_ENDIF;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  reference.InsertBack(Expression::CreateIf(2));
  reference.InsertBack(Expression::CreateElse());
  reference.InsertBack(Expression::CreateEndIf());
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, IF_NESTED) {
  StartRecordingExpressions();
  T a = T(1);
  T b = T(2);
  auto r1 = a < b;
  auto r2 = a == b;
  CERES_IF(r1) {
    CERES_IF(r2) {}
    CERES_ENDIF;
  }
  CERES_ELSE {}
  CERES_ENDIF;
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateCompileTimeConstant(1));
  reference.InsertBack(Expression::CreateCompileTimeConstant(2));
  reference.InsertBack(Expression::CreateBinaryCompare("<", 0, 1));
  reference.InsertBack(Expression::CreateBinaryCompare("==", 0, 1));
  reference.InsertBack(Expression::CreateIf(2));
  reference.InsertBack(Expression::CreateIf(3));
  reference.InsertBack(Expression::CreateEndIf());
  reference.InsertBack(Expression::CreateElse());
  reference.InsertBack(Expression::CreateEndIf());
  EXPECT_EQ(reference, graph);
}

TEST(ExpressionRef, COMMENT) {
  StartRecordingExpressions();
  CERES_COMMENT("This is a comment");
  auto graph = StopRecordingExpressions();

  ExpressionGraph reference;
  reference.InsertBack(Expression::CreateComment("This is a comment"));
  EXPECT_EQ(reference, graph);
}

}  // namespace internal
}  // namespace ceres
