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

#include "ceres/internal/expression_tree.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(Expression, Dependencies) {
  using T = ExpressionRef;

  StartRecordingExpressions();

  T unused(6);
  T a(2), b(3);
  T c = a + b;
  T d = c + a;

  auto tree = StopRecordingExpressions();

  // Direct dependency check
  ASSERT_FALSE(tree.get(a).DirectlyDependsOn(b));
  ASSERT_FALSE(tree.get(a).DirectlyDependsOn(unused));
  ASSERT_TRUE(tree.get(c).DirectlyDependsOn(a));
  ASSERT_TRUE(tree.get(c).DirectlyDependsOn(b));
  ASSERT_TRUE(tree.get(d).DirectlyDependsOn(a));
  ASSERT_FALSE(tree.get(d).DirectlyDependsOn(b));
  ASSERT_TRUE(tree.get(d).DirectlyDependsOn(c));

  // Recursive dependency check
  ASSERT_TRUE(tree.DependsOn(d, c));
  ASSERT_TRUE(tree.DependsOn(d, a));
  ASSERT_TRUE(tree.DependsOn(d, b));
  ASSERT_FALSE(tree.DependsOn(d, unused));
}

}  // namespace internal
}  // namespace ceres
