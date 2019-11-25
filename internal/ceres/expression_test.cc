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
#include "ceres/jet.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(Expression, IsArithmetic) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T a(2), b(3);
  T c = a + b;
  T d = c + a;

  auto graph = StopRecordingExpressions();

  ASSERT_TRUE(graph.ExpressionForId(a.id).IsArithmeticExpression());
  ASSERT_TRUE(graph.ExpressionForId(b.id).IsArithmeticExpression());
  ASSERT_TRUE(graph.ExpressionForId(c.id).IsArithmeticExpression());
  ASSERT_TRUE(graph.ExpressionForId(d.id).IsArithmeticExpression());
}

TEST(Expression, IsCompileTimeConstantAndEqualTo) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T a(2), b(3);
  T c = a + b;

  auto graph = StopRecordingExpressions();

  ASSERT_TRUE(graph.ExpressionForId(a.id).IsCompileTimeConstantAndEqualTo(2));
  ASSERT_FALSE(graph.ExpressionForId(a.id).IsCompileTimeConstantAndEqualTo(0));
  ASSERT_TRUE(graph.ExpressionForId(b.id).IsCompileTimeConstantAndEqualTo(3));
  ASSERT_FALSE(graph.ExpressionForId(c.id).IsCompileTimeConstantAndEqualTo(0));
}

TEST(Expression, IsReplaceableBy) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  // a2 should be replaceable by a
  T a(2), b(3), a2(2);

  // two redundant expressions
  // -> d should be replaceable by c
  T c = a + b;
  T d = a + b;

  auto graph = StopRecordingExpressions();

  ASSERT_TRUE(graph.ExpressionForId(a2.id).IsReplaceableBy(
      graph.ExpressionForId(a.id)));
  ASSERT_TRUE(
      graph.ExpressionForId(d.id).IsReplaceableBy(graph.ExpressionForId(c.id)));
  ASSERT_FALSE(graph.ExpressionForId(d.id).IsReplaceableBy(
      graph.ExpressionForId(a2.id)));
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

}  // namespace internal
}  // namespace ceres
