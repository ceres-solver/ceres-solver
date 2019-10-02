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

}  // namespace internal
}  // namespace ceres
