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

#include "ceres/codegen/internal/expression_dependencies.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(ExpressoinDependencies, LinearSSA) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateBinaryArithmetic("+", 0, 1));
  graph.InsertBack(Expression::CreateScalarFunctionCall("sin", {2}));

  std::vector<std::vector<ExpressionId>> written_to_reference = {
      {0}, {1}, {2}, {3}};

  std::vector<std::vector<ExpressionId>> used_by_reference = {
      {2}, {2}, {3}, {}};

  ExpressionDependencies ep(graph);
  for (int i = 0; i < graph.Size(); ++i) {
    EXPECT_EQ(ep.DataForExpressionId(i).written_to, written_to_reference[i]);
    EXPECT_EQ(ep.DataForExpressionId(i).used_by, used_by_reference[i]);
  }
}

TEST(ExpressoinDependencies, Linear) {
  ExpressionGraph graph;
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateCompileTimeConstant(42));
  graph.InsertBack(Expression::CreateBinaryArithmetic("+", 0, 1));
  graph.InsertBack(Expression::CreateScalarFunctionCall("sin", {2}));
  // Add some assignments to previous variables
  graph.InsertBack(Expression::CreateAssignment(0, 1));
  graph.InsertBack(Expression::CreateAssignment(3, 2));
  graph.InsertBack(Expression::CreateAssignment(3, 3));

  std::vector<std::vector<ExpressionId>> written_to_reference = {
      {0, 4}, {1}, {2}, {3, 5, 6}, {0, 4}, {3, 5, 6}, {3, 5, 6}};

  std::vector<std::vector<ExpressionId>> used_by_reference = {
      {2}, {2, 4}, {3, 5}, {6}, {2}, {6}, {6}};

  ExpressionDependencies ep(graph);
  for (int i = 0; i < graph.Size(); ++i) {
    auto lhs_id = graph.ExpressionForId(i).lhs_id();
    EXPECT_EQ(ep.DataForExpressionId(lhs_id).written_to,
              written_to_reference[i]);
    EXPECT_EQ(ep.DataForExpressionId(lhs_id).used_by, used_by_reference[i]);
  }
}

}  // namespace internal
}  // namespace ceres
