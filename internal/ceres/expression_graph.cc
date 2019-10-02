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

#include "ceres/internal/expression_graph.h"

#include "glog/logging.h"
namespace ceres {
namespace internal {

static ExpressionGraph* expression_pool = nullptr;

void StartRecordingExpressions() {
  CHECK(expression_pool == nullptr)
      << "Expression recording must be stopped before calling "
         "StartRecordingExpressions again.";
  expression_pool = new ExpressionGraph;
}

ExpressionGraph StopRecordingExpressions() {
  CHECK(expression_pool)
      << "Expression recording hasn't started yet or you tried "
         "to stop it twice.";
  ExpressionGraph result = std::move(*expression_pool);
  delete expression_pool;
  expression_pool = nullptr;
  return result;
}

ExpressionGraph* GetCurrentExpressionGraph() { return expression_pool; }

Expression& ExpressionGraph::CreateExpression(ExpressionType type) {
  auto id = expressions_.size();
  Expression expr(type, id);
  expressions_.push_back(expr);
  return expressions_.back();
}

bool ExpressionGraph::DependsOn(ExpressionId A, ExpressionId B) const {
  // Depth first search on the expression graph
  // Equivalent Recursive Implementation:
  //   if (A.DirectlyDependsOn(B)) return true;
  //   for (auto p : A.params_) {
  //     if (pool[p.id].DependsOn(B, pool)) return true;
  //   }
  std::vector<ExpressionId> stack = ExpressionForId(A).arguments_;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top == B) {
      return true;
    }
    auto& expr = ExpressionForId(top);
    stack.insert(stack.end(), expr.arguments_.begin(), expr.arguments_.end());
  }
  return false;
}
}  // namespace internal
}  // namespace ceres
