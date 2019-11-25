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

TEST(Expression, AssignmentElimination) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a(2);
  T b;
  b = a;
  auto graph = StopRecordingExpressions();

  // b is invalid during the assignment so we expect no expression to be
  // generated. The only expression in the graph should be the constant
  // assignment to a.
  EXPECT_EQ(graph.Size(), 1);

  // Expected code
  //   v_0 = 2;

  ExpressionGraph reference;
  // clang-format off
  // Id  Type                   Lhs  Value Name  Arguments
  reference.InsertExpression(0,  ExpressionType::COMPILE_TIME_CONSTANT, 0,   {}, "", 2);
  // clang-format on
  EXPECT_EQ(reference, graph);

  // Variables after execution:
  //
  // a      <=> v_0
  // b      <=> v_0
  EXPECT_EQ(a.id, 0);
  EXPECT_EQ(b.id, 0);
}

TEST(Expression, Assignment) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a(2);
  T b(4);
  b = a;
  auto graph = StopRecordingExpressions();

  // b is valid during the assignment so we expect an
  // additional assignment expression.
  EXPECT_EQ(graph.Size(), 3);

  // Expected code
  //   v_0 = 2;
  //   v_1 = 4;
  //   v_1 = v_0;

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  reference.InsertExpression(  0,  ExpressionType::COMPILE_TIME_CONSTANT,   0,    {}  , "",  2);
  reference.InsertExpression(  1,  ExpressionType::COMPILE_TIME_CONSTANT,   1,     {} , "",  4);
  reference.InsertExpression(  2,             ExpressionType::ASSIGNMENT,   1,     {0}, "",  0);
  // clang-format on
  EXPECT_EQ(reference, graph);

  // Variables after execution:
  //
  // a      <=> v_0
  // b      <=> v_1
  EXPECT_EQ(a.id, 0);
  EXPECT_EQ(b.id, 1);
}

TEST(Expression, ConditionalMinimal) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T a(2);
  T b(3);
  auto c = a < b;
  CERES_IF(c) {}
  CERES_ELSE {}
  CERES_ENDIF
  auto graph = StopRecordingExpressions();

  // Expected code
  //   v_0 = 2;
  //   v_1 = 3;
  //   v_2 = v_0 < v_1;
  //   if(v_2);
  //   else
  //   endif

  EXPECT_EQ(graph.Size(), 6);

  ExpressionGraph reference;
  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments...
  reference.InsertExpression(  0, ExpressionType::COMPILE_TIME_CONSTANT,   0,  {}     ,    "", 2);
  reference.InsertExpression(  1, ExpressionType::COMPILE_TIME_CONSTANT,   1,  {}     ,    "", 3);
  reference.InsertExpression(  2,     ExpressionType::BINARY_COMPARISON,   2,  {0, 1} ,   "<", 0);
  reference.InsertExpression(  3,                    ExpressionType::IF,  -1,  {2}    ,    "", 0);
  reference.InsertExpression(  4,                  ExpressionType::ELSE,  -1,  {}     ,    "", 0);
  reference.InsertExpression(  5,                 ExpressionType::ENDIF,  -1,  {}     ,    "", 0);
  // clang-format on
  EXPECT_EQ(reference, graph);
}

TEST(Expression, ConditionalAssignment) {
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

  ExpressionGraph reference;
  // clang-format off
  // Id,   Type,                  Lhs, Value, Name, Arguments...
  reference.InsertExpression(  0,  ExpressionType::COMPILE_TIME_CONSTANT,    0,   {}    ,   "",   2);
  reference.InsertExpression(  1,  ExpressionType::COMPILE_TIME_CONSTANT,    1,   {}    ,   "",   3);
  reference.InsertExpression(  2,      ExpressionType::BINARY_COMPARISON,    2,   {0, 1},  "<",   0);
  reference.InsertExpression(  3,                     ExpressionType::IF,   -1,   {2}   ,   "",   0);
  reference.InsertExpression(  4,      ExpressionType::BINARY_ARITHMETIC,    4,   {0, 1},  "+",   0);
  reference.InsertExpression(  5,                   ExpressionType::ELSE,   -1,   {}    ,   "",   0);
  reference.InsertExpression(  6,      ExpressionType::BINARY_ARITHMETIC,    6,   {0, 1},  "-",   0);
  reference.InsertExpression(  7,             ExpressionType::ASSIGNMENT,    4,   {6}   ,   "",   0);
  reference.InsertExpression(  8,                  ExpressionType::ENDIF,   -1,   {}    ,   "",   0);
  reference.InsertExpression(  9,      ExpressionType::BINARY_ARITHMETIC,    9,   {4, 0},  "+",   0);
  reference.InsertExpression( 10,             ExpressionType::ASSIGNMENT,    4,   {9}   ,   "",   0);
  // clang-format on
  EXPECT_EQ(reference, graph);

  // Variables after execution:
  //
  // a      <=> v_0
  // b      <=> v_1
  // result <=> v_4
  EXPECT_EQ(a.id, 0);
  EXPECT_EQ(b.id, 1);
  EXPECT_EQ(result.id, 4);
}

}  // namespace internal
}  // namespace ceres
