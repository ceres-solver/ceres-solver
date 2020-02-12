// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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

#include "ceres/codegen/internal/eliminate_nops.h"

#include "ceres/codegen/internal/code_generator.h"
#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/expression_ref.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using T = ExpressionRef;

TEST(EliminateNops, SimpleLinear) {
  StartRecordingExpressions();
  {
    T a = T(0);
    // The Expression default constructor creates a NOP.
    AddExpressionToGraph(Expression());
    AddExpressionToGraph(Expression());
    T b = T(2);
    AddExpressionToGraph(Expression());
    MakeOutput(b, "residual[0]");
    AddExpressionToGraph(Expression());
  }
  auto graph = StopRecordingExpressions();

  StartRecordingExpressions();
  {
    T a = T(0);
    T b = T(2);
    MakeOutput(b, "residual[0]");
  }
  auto reference = StopRecordingExpressions();

  auto summary = EliminateNops(&graph);
  EXPECT_TRUE(summary.expression_graph_changed);
  EXPECT_EQ(graph, reference);
}

TEST(EliminateNops, Branches) {
  StartRecordingExpressions();
  {
    T a = T(0);
    // The Expression default constructor creates a NOP.
    AddExpressionToGraph(Expression());
    AddExpressionToGraph(Expression());
    T b = T(2);
    CERES_IF(a < b) {
      AddExpressionToGraph(Expression());
      T c = T(3);
    }
    CERES_ELSE {
      AddExpressionToGraph(Expression());
      MakeOutput(b, "residual[0]");
      AddExpressionToGraph(Expression());
    }
    CERES_ENDIF
    AddExpressionToGraph(Expression());
  }
  auto graph = StopRecordingExpressions();

  StartRecordingExpressions();
  {
    T a = T(0);
    T b = T(2);
    CERES_IF(a < b) { T c = T(3); }
    CERES_ELSE { MakeOutput(b, "residual[0]"); }
    CERES_ENDIF
  }
  auto reference = StopRecordingExpressions();

  auto summary = EliminateNops(&graph);
  EXPECT_TRUE(summary.expression_graph_changed);
  EXPECT_EQ(graph, reference);
}

}  // namespace internal
}  // namespace ceres
