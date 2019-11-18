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

#define CERES_CODEGEN

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

TEST(ExpressionGraph, IsValid) {
  using T = ExpressionRef;

  {
    // Empty graphs are always valid
    StartRecordingExpressions();
    auto graph = StopRecordingExpressions();
    ASSERT_TRUE(graph.IsValid());
  }

  {
    // A valid graph with only a few  simple expressions
    StartRecordingExpressions();
    T a(2), b(3);
    T c = a + b;
    T d = c + a;
    auto graph = StopRecordingExpressions();
    ASSERT_TRUE(graph.IsValid());
  }

  {
    // A valid graph with a single if and a few assignments
    StartRecordingExpressions();
    T v_0(2);
    T v_1(3);
    auto c = v_0 < v_1;
    CERES_IF(c) { v_0 += 2.0; }
    CERES_ELSE { v_0 += 3.0; }
    CERES_ENDIF;
    v_0 += 1.0;
    auto graph = StopRecordingExpressions();
    ASSERT_TRUE(graph.IsValid());
  }

  {
    // A valid graph with a nested if and a few assignments
    StartRecordingExpressions();
    T v_0(2);
    T v_1(3);
    auto c = v_0 < v_1;
    CERES_IF(c) {
      CERES_IF(c) { v_0 += 2.0; }
      CERES_ELSE { v_0 += 2.0; }
      CERES_ENDIF;
      v_0 += 2.0;
    }
    CERES_ELSE {
      CERES_IF(c) {
        v_0 += 2.0;
        CERES_IF(c) { v_0 += 2.0; }
        CERES_ENDIF;
      }
      CERES_ELSE {}
      CERES_ENDIF;
      v_0 += 2.0;
    }
    CERES_ENDIF;
    v_0 += 1.0;
    auto graph = StopRecordingExpressions();
    ASSERT_TRUE(graph.IsValid());
  }

  {
    // Invalid Graph: missing ENDIF
    StartRecordingExpressions();
    T v_0(2);
    T v_1(3);
    auto c = v_0 < v_1;
    CERES_IF(c) { v_0 += 2.0; }
    CERES_ELSE { v_0 += 3.0; }
    v_0 += 1.0;
    auto graph = StopRecordingExpressions();
    ASSERT_FALSE(graph.IsValid());
  }
  {
    // Invalid Graph: missing IF
    StartRecordingExpressions();
    T v_0(2);
    T v_1(3);
    auto c = v_0 < v_1;
    CERES_IF(c) { v_0 += 2.0; }
    CERES_ELSE {
      CERES_ELSE { v_0 += 3.0; }
      CERES_ENDIF;
    }
    CERES_ENDIF;
    v_0 += 1.0;
    auto graph = StopRecordingExpressions();
    ASSERT_FALSE(graph.IsValid());
  }

  {
    // Invalid Graph: manually setting expression id
    StartRecordingExpressions();
    T a(2), b(3);
    T c = a + b;
    c.id = 10;
    T d = c + a;
    auto graph = StopRecordingExpressions();
    ASSERT_FALSE(graph.IsValid());
  }
}

TEST(ExpressionGraph, FindMatchingIf) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T v_0(2);
  T v_1(3);
  auto c = v_0 < v_1;
  CERES_IF(c) {       // 3
    CERES_IF(c) {}    // 4
    CERES_ELSE {}     // 5
    CERES_ENDIF;      // 6
  }                   //
  CERES_ELSE {        // 7
    CERES_IF(c) {     // 8
      CERES_IF(c) {}  // 9
      CERES_ENDIF;    // 10
    }                 //
    CERES_ELSE {}     // 11
    CERES_ENDIF;      // 12
  }                   //
  CERES_ENDIF;        // 13
  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 14);

  EXPECT_EQ(graph.FindMatchingIf(5), 4);
  EXPECT_EQ(graph.FindMatchingIf(6), 4);
  EXPECT_EQ(graph.FindMatchingIf(7), 3);
  EXPECT_EQ(graph.FindMatchingIf(10), 9);
  EXPECT_EQ(graph.FindMatchingIf(11), 8);
  EXPECT_EQ(graph.FindMatchingIf(12), 8);
  EXPECT_EQ(graph.FindMatchingIf(13), 3);
}

TEST(ExpressionGraph, GetParentControlExpression) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  T v_0(2);               // 0  Parent = root
  T v_1(3);               // 1  Parent = root
  auto c = v_0 < v_1;     // 2  Parent = root
  CERES_IF(c) {           // 3  Parent = root
    CERES_IF(c) {         // 4  Parent = 3
      T a = v_0 + v_1;    // 5  Parent = 4
    }                     //
    CERES_ELSE {          // 6  Parent = 4
      T a = v_0 + v_1;    // 7  Parent = 6
    }                     //
    CERES_ENDIF;          // 8  Parent = 6
    T a = v_0 + v_1;      // 9  Parent = 3
  }                       //
  CERES_ELSE {            // 10  Parent = 3
    T a = v_0 + v_1;      // 11  Parent = 10
    CERES_IF(c) {         // 12  Parent = 10
      T a = v_0 + v_1;    // 13  Parent = 12
      CERES_IF(c) {       // 14  Parent = 12
        T a = v_0 + v_1;  // 15  Parent = 14
      }                   //
      CERES_ENDIF;        // 16  Parent = 14
      T b = v_0 + v_1;    // 17  Parent = 12
    }                     //
    CERES_ELSE {          // 18  Parent = 12
      T a = v_0 + v_1;    // 19  Parent = 18
    }                     //
    CERES_ENDIF;          // 20  Parent = 12
    T b = v_0 + v_1;      // 21  Parent = 10
  }                       //
  CERES_ENDIF;            // 22  Parent = 10
  T a = v_0 + v_1;        // 23  Parent = root

  auto graph = StopRecordingExpressions();
  EXPECT_EQ(graph.Size(), 24);

  auto root = kInvalidExpressionId;
  EXPECT_EQ(graph.GetParentControlExpression(0), root);
  EXPECT_EQ(graph.GetParentControlExpression(1), root);
  EXPECT_EQ(graph.GetParentControlExpression(2), root);
  EXPECT_EQ(graph.GetParentControlExpression(3), root);
  EXPECT_EQ(graph.GetParentControlExpression(4), 3);
  EXPECT_EQ(graph.GetParentControlExpression(5), 4);
  EXPECT_EQ(graph.GetParentControlExpression(6), 4);
  EXPECT_EQ(graph.GetParentControlExpression(7), 6);
  EXPECT_EQ(graph.GetParentControlExpression(8), 6);
  EXPECT_EQ(graph.GetParentControlExpression(9), 3);
  EXPECT_EQ(graph.GetParentControlExpression(10), 3);
  EXPECT_EQ(graph.GetParentControlExpression(11), 10);
  EXPECT_EQ(graph.GetParentControlExpression(12), 10);
  EXPECT_EQ(graph.GetParentControlExpression(13), 12);
  EXPECT_EQ(graph.GetParentControlExpression(14), 12);
  EXPECT_EQ(graph.GetParentControlExpression(15), 14);
  EXPECT_EQ(graph.GetParentControlExpression(16), 14);
  EXPECT_EQ(graph.GetParentControlExpression(17), 12);
  EXPECT_EQ(graph.GetParentControlExpression(18), 12);
  EXPECT_EQ(graph.GetParentControlExpression(19), 18);
  EXPECT_EQ(graph.GetParentControlExpression(20), 18);
  EXPECT_EQ(graph.GetParentControlExpression(21), 10);
  EXPECT_EQ(graph.GetParentControlExpression(22), 10);
  EXPECT_EQ(graph.GetParentControlExpression(23), root);
}

TEST(ExpressionGraph, IsEquivalentTo) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(e, "result");
  }
  auto reference = StopRecordingExpressions();

  // Every graph must be equivalent to itself
  ASSERT_TRUE(reference.IsEquivalentTo(reference));

  // Exactly the same graph
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(e, "result");
  }
  auto graph1 = StopRecordingExpressions();
  ASSERT_TRUE(reference.IsEquivalentTo(graph1));
  ASSERT_TRUE(graph1.IsEquivalentTo(graph1));

  // Change the definition order of a and b
  // -> Not equivalent anymore
  StartRecordingExpressions();
  {
    T b(3);
    T a(2);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(e, "result");
  }
  auto graph2 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph2));
  ASSERT_TRUE(graph2.IsEquivalentTo(graph2));
}

TEST(ExpressionGraph, IsSemanticallyEquivalentTo) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(d, "result_d");
    MakeOutput(e, "result_e");
  }
  auto reference = StopRecordingExpressions();

  // Change definition order of a and b
  // -> not equivalent but still semantically equivalent
  StartRecordingExpressions();
  {
    T b(3);
    T a(2);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(d, "result_d");
    MakeOutput(e, "result_e");
  }
  auto graph1 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph1));
  ASSERT_TRUE(reference.IsSemanticallyEquivalentTo(graph1));

  // Renamed an output
  // -> not semantically equivalent
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(d, "result_d_renamed");
    MakeOutput(e, "result_e");
  }
  auto graph2 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph2));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph2));

  // Change output order
  // -> not semantically equivalent
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    MakeOutput(e, "result_e");
    MakeOutput(d, "result_d");
  }
  auto graph3 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph3));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph3));

  // Reverse arguments when computing e
  // -> not semantically equivalent (floating point order matters!)
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T c = a + b;
    T d = a * b;
    T e = d + c;
    MakeOutput(d, "result_d");
    MakeOutput(e, "result_e");
  }
  auto graph4 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph4));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph4));

  // Added some unused variables
  // -> semantically equivalent
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    T unused(5);
    T c = a + b;
    T d = a * b;
    T e = c + d;
    unused += e;
    MakeOutput(d, "result_d");
    MakeOutput(e, "result_e");
  }
  auto graph5 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph5));
  ASSERT_TRUE(reference.IsSemanticallyEquivalentTo(graph5));
}

TEST(ExpressionGraph, IsSemanticallyEquivalentTo_MultipleAssignments) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  {
    T a(2);                     // 0
    T b(3);                     // 1
    a = 1;                      // 3
    a = 2;                      // 5
    b = 1;                      // 7
    T c = a + b;                // 8
    MakeOutput(c, "result_c");  // 9
  }
  auto reference = StopRecordingExpressions();
  EXPECT_EQ(reference.Size(), 10);

  // The first assignemnt to a is removed.
  // -> not equivalent
  // Note: after live time analysis and unused code elemination they would be
  // equivalent, because a = 1 in the reference graph would be eliminated.
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    a = 2;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph1 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph1));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph1));

  // Change order of assignments to a
  // -> invalid
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    a = 2;
    a = 1;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph2 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph2));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph2));

  // Change order of assignments to a and b
  // -> still valid
  StartRecordingExpressions();
  {
    T a(2);                     // 0
    T b(3);                     // 1
    a = 1;                      // 3
    b = 1;                      // 5
    a = 2;                      // 7
    T c = a + b;                // 8
    MakeOutput(c, "result_c");  // 9
  }
  auto graph3 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph3));
  ASSERT_TRUE(reference.IsSemanticallyEquivalentTo(graph3));

  // Change order of assignments to a and b and declaration of a and b
  // -> still valid
  StartRecordingExpressions();
  {
    T b(3);                     // 0
    T a(2);                     // 1
    a = 1;                      // 3
    b = 1;                      // 5
    a = 2;                      // 7
    T c = a + b;                // 8
    MakeOutput(c, "result_c");  // 9
  }
  auto graph4 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph4));
  ASSERT_TRUE(reference.IsSemanticallyEquivalentTo(graph4));
}

TEST(ExpressionGraph, IsSemanticallyEquivalentTo_Branches) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  {
    T a(2);                     // 0
    T b(3);                     // 1
    auto con = a < b;           // 2
    CERES_IF(con) {             // 3
      a = 1;                    // 5
      a = 2;                    // 7
    }                           //
    CERES_ELSE {                // 8
      b = 3;                    // 10
    }                           //
    CERES_ENDIF;                // 11
    b = 1;                      // 13
    T c = a + b;                // 14
    MakeOutput(c, "result_c");  // 15
  }
  auto reference = StopRecordingExpressions();
  EXPECT_EQ(reference.Size(), 16);

  // Change the condition
  // -> invalid
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    auto con = b < a;
    CERES_IF(con) {
      a = 1;
      a = 2;
    }
    CERES_ELSE { b = 3; }
    CERES_ENDIF;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph1 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph1));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph1));

  // Change the branch content
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    auto con = b < a;
    CERES_IF(con) { b = 3; }
    CERES_ELSE {
      a = 1;
      a = 2;
    }
    CERES_ENDIF;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph2 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph2));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph2));

  // Move last assignemnt to b outside else-branch
  StartRecordingExpressions();
  {
    T a(2);
    T b(3);
    auto con = b < a;
    CERES_IF(con) {
      a = 1;
      a = 2;
    }
    CERES_ELSE {}
    CERES_ENDIF;
    b = 3;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph3 = StopRecordingExpressions();
  ASSERT_FALSE(reference.IsEquivalentTo(graph3));
  ASSERT_FALSE(reference.IsSemanticallyEquivalentTo(graph3));

  // Add a nested if which doesn't contibute to c
  // -> valid
  StartRecordingExpressions();
  {
    T unused(5);
    T a(2);
    T b(3);
    auto con = a < b;
    CERES_IF(con) {
      a = 1;
      a = 2;
      auto con2 = a < b;
      CERES_IF(con2) { unused = 5; }
      CERES_ENDIF;
    }
    CERES_ELSE { b = 3; }
    CERES_ENDIF;
    b = 1;
    T c = a + b;
    MakeOutput(c, "result_c");
  }
  auto graph4 = StopRecordingExpressions();
  EXPECT_EQ(graph4.Size(), 22);
  ASSERT_FALSE(reference.IsEquivalentTo(graph4));
  ASSERT_TRUE(reference.IsSemanticallyEquivalentTo(graph4));
}

TEST(ExpressionGraph, InsertExpression) {
  using T = ExpressionRef;

  StartRecordingExpressions();
  {
    T a(2);                   // 0
    T b(3);                   // 1
    T five = 5;               // 2
    a += five;                // 3 + 4
    T c = a + b;              // 5
    T d = a * b;              // 6
    T e = c + d;              // 7
    MakeOutput(e, "result");  // 8
  }
  auto reference = StopRecordingExpressions();
  EXPECT_EQ(reference.Size(), 9);

  // The expression  a += 5; is missing
  StartRecordingExpressions();
  {
    T a(2);                   // 0
    T b(3);                   // 1
    T c = a + b;              // 2
    T d = a * b;              // 3
    T e = c + d;              // 4
    MakeOutput(e, "result");  // 5
  }
  auto graph1 = StopRecordingExpressions();
  EXPECT_EQ(graph1.Size(), 6);
  ASSERT_FALSE(reference.IsEquivalentTo(graph1));

  // We manually insert the 3 missing expressions
  graph1.InsertExpression(
      2, ExpressionType::COMPILE_TIME_CONSTANT, 2, {}, "", 5);
  graph1.InsertExpression(
      3, ExpressionType::BINARY_ARITHMETIC, 3, {0, 2}, "+", 0);
  graph1.InsertExpression(4, ExpressionType::ASSIGNMENT, 0, {3}, "", 0);

  // Now the graphs should be idendical
  EXPECT_EQ(graph1.Size(), 9);
  ASSERT_TRUE(reference.IsEquivalentTo(graph1));
}
}  // namespace internal
}  // namespace ceres
