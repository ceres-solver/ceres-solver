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

#include "ceres/internal/expression_tree.h"
#include "assert.h"

namespace ceres {
namespace internal {

// During execution, the expressions add themself into this vector. This
// allows us to see what code was executed and optimize it later.
static ExpressionTree* expression_pool = nullptr;

void StartRecordingExpressions() {
  assert(expression_pool == nullptr);
  expression_pool = new ExpressionTree;
}

ExpressionTree StopRecordingExpressions() {
  assert(expression_pool);
  ExpressionTree result = std::move(*expression_pool);
  delete expression_pool;
  expression_pool = nullptr;
  return result;
}

ExpressionTree* GetCurrentExpressionTree() { return expression_pool; }

Expression& ExpressionTree::MakeExpression(ExpressionType type) {
  auto id = data_.size();
  ExpressionRef e;
  e.id = id;
  Expression expr(type, e);
  data_.push_back(expr);
  return data_.back();
}

bool ExpressionTree::DependsOn(ExpressionRef A, ExpressionRef B) const {
  // Depth first search on the expression tree
  // Equivalent Recursive Implementation:
  //  if (A.DirectlyDependsOn(B)) return true;
  //  for (auto p : A.params_) {
  //    if (pool[p.id].DependsOn(B, pool)) return true;
  //  }
  std::vector<ExpressionRef> stack = get(A).params_;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top.id == B.id) return true;
    auto& expr = get(top);
    stack.insert(stack.end(), expr.params_.begin(), expr.params_.end());
  }
  return false;
}
}  // namespace internal
}  // namespace ceres
