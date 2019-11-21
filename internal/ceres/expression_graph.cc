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

Expression& ExpressionGraph::CreateArithmeticExpression(ExpressionType type,
                                                        ExpressionId lhs_id) {
  if (lhs_id == kInvalidExpressionId) {
    // We are creating a new temporary variable.
    // -> The new lhs_id is the index into the graph
    lhs_id = static_cast<ExpressionId>(expressions_.size());
  } else {
    // The left hand side already exists.
  }

  Expression expr(type, lhs_id);
  expressions_.push_back(expr);
  return expressions_.back();
}

Expression& ExpressionGraph::CreateControlExpression(ExpressionType type) {
  Expression expr(type, kInvalidExpressionId);
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

bool ExpressionGraph::operator==(const ExpressionGraph& other) const {
  if (Size() != other.Size()) {
    return false;
  }
  for (ExpressionId id = 0; id < Size(); ++id) {
    if (!(ExpressionForId(id) == other.ExpressionForId(id))) {
      return false;
    }
  }
  return true;
}

void ExpressionGraph::InsertExpression(
    ExpressionId location,
    ExpressionType type,
    ExpressionId lhs_id,
    const std::vector<ExpressionId>& arguments,
    const std::string& name,
    double value) {
  ExpressionId last_expression_id = Size() - 1;
  // Increase size by adding a dummy expression.
  expressions_.push_back(Expression(ExpressionType::NOP, kInvalidExpressionId));

  // Move everything after id back and update references
  for (ExpressionId id = last_expression_id; id >= location; --id) {
    auto& expression = expressions_[id];
    // Increment reference if it points to a shifted variable.
    if (expression.lhs_id_ >= location) {
      expression.lhs_id_++;
    }
    for (auto& arg : expression.arguments_) {
      if (arg >= location) {
        arg++;
      }
    }
    expressions_[id + 1] = expression;
  }

  // Insert new expression at the correct place
  Expression expr(type, lhs_id);
  expr.arguments_ = arguments;
  expr.name_ = name;
  expr.value_ = value;
  expressions_[location] = expr;
}

}  // namespace internal
}  // namespace ceres
