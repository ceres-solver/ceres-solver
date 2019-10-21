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

#include "expression_test.h"

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

  // clang-format off
  // Id  Type                   Lhs  Value Name  Arguments
  TE(0,  COMPILE_TIME_CONSTANT, 0,   2,     "");
  // clang-format on

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

  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments
  TE(  0,  COMPILE_TIME_CONSTANT,   0,   2,   "",   );
  TE(  1,  COMPILE_TIME_CONSTANT,   1,   4,   "",   );
  TE(  2,             ASSIGNMENT,   1,   0,   "",  0);
  // clang-format on

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

  // clang-format off
  // Id, Type, Lhs, Value, Name, Arguments...
  TE(  0, COMPILE_TIME_CONSTANT,   0,   2,   "",      );
  TE(  1, COMPILE_TIME_CONSTANT,   1,   3,   "",      );
  TE(  2,     BINARY_COMPARISON,   2,   0,  "<",  0, 1);
  TE(  3,                    IF,  -1,   0,   "",     2);
  TE(  4,                  ELSE,  -1,   0,   "",      );
  TE(  5,                 ENDIF,  -1,   0,   "",      );
  // clang-format on
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

  // clang-format off
  // Id,   Type,                  Lhs, Value, Name, Arguments...
  TE(  0,  COMPILE_TIME_CONSTANT,    0,    2,   "",      );
  TE(  1,  COMPILE_TIME_CONSTANT,    1,    3,   "",      );
  TE(  2,      BINARY_COMPARISON,    2,    0,  "<",  0, 1);
  TE(  3,                     IF,   -1,    0,   "",     2);
  TE(  4,      BINARY_ARITHMETIC,    4,    0,  "+",  0, 1);
  TE(  5,                   ELSE,   -1,    0,   "",      );
  TE(  6,      BINARY_ARITHMETIC,    6,    0,  "-",  0, 1);
  TE(  7,             ASSIGNMENT,    4,    0,   "",  6   );
  TE(  8,                  ENDIF,   -1,    0,   "",      );
  TE(  9,      BINARY_ARITHMETIC,    9,    0,  "+",  4, 0);
  TE( 10,             ASSIGNMENT,    4,    0,   "",  9   );
  // clang-format on

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
