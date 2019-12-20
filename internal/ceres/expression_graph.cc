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

#include "ceres/codegen/internal/expression_graph.h"

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

bool ExpressionGraph::DependsOn(ExpressionId A, ExpressionId B) const {
  // Depth first search on the expression graph
  // Equivalent Recursive Implementation:
  //   if (A.DirectlyDependsOn(B)) return true;
  //   for (auto p : A.params_) {
  //     if (pool[p.id].DependsOn(B, pool)) return true;
  //   }
  std::vector<ExpressionId> stack = ExpressionForId(A).arguments();
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top == B) {
      return true;
    }
    auto& expr = ExpressionForId(top);
    stack.insert(stack.end(), expr.arguments().begin(), expr.arguments().end());
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

void ExpressionGraph::Insert(ExpressionId location,
                             const Expression& expression) {
  ExpressionId last_expression_id = Size() - 1;
  // Increase size by adding a dummy expression.
  expressions_.push_back(Expression());

  // Move everything after id back and update references
  for (ExpressionId id = last_expression_id; id >= location; --id) {
    auto& expression = expressions_[id];
    // Increment reference if it points to a shifted variable.
    if (expression.lhs_id() >= location) {
      expression.set_lhs_id(expression.lhs_id() + 1);
    }
    for (auto& arg : *expression.mutable_arguments()) {
      if (arg >= location) {
        arg++;
      }
    }
    expressions_[id + 1] = expression;
  }

  if (expression.IsControlExpression() ||
      expression.lhs_id() != kInvalidExpressionId) {
    // Insert new expression at the correct place
    expressions_[location] = expression;
  } else {
    // Arithmetic expression with invalid lhs
    // -> Set lhs to location
    Expression copy = expression;
    copy.set_lhs_id(location);
    expressions_[location] = copy;
  }
}

ExpressionId ExpressionGraph::InsertBack(const Expression& expression) {
  if (expression.IsControlExpression()) {
    // Control expression are just added to the list. We do not return a
    // reference to them.
    CHECK(expression.lhs_id() == kInvalidExpressionId)
        << "Control expressions must have an invalid lhs.";
    expressions_.push_back(expression);
    return kInvalidExpressionId;
  }

  if (expression.lhs_id() == kInvalidExpressionId) {
    // Create a new variable name for this expression and set it as the lhs
    Expression copy = expression;
    copy.set_lhs_id(static_cast<ExpressionId>(expressions_.size()));
    expressions_.push_back(copy);
  } else {
    // The expressions writes to a variable declared in the past
    // -> Just add it to the list
    CHECK_LE(expression.lhs_id(), expressions_.size())
        << "The left hand side must reference a variable in the past.";
    expressions_.push_back(expression);
  }

  return Size() - 1;
}

ExpressionId ExpressionGraph::FindMatchingEndif(ExpressionId id) const {
  CHECK(ExpressionForId(id).type() == ExpressionType::IF)
      << "FindClosingControlExpression is only valid on IF "
         "expressions.";

  // Traverse downwards
  for (ExpressionId i = id + 1; i < Size(); ++i) {
    const auto& expr = ExpressionForId(i);
    if (expr.type() == ExpressionType::ENDIF) {
      return i;

    } else if (expr.type() == ExpressionType::IF) {
      // Found a nested IF.
      // -> Jump over the block and continue behind it.
      auto matching_endif = FindMatchingEndif(i);
      if (matching_endif == kInvalidExpressionId) {
        return kInvalidExpressionId;
      }
      i = matching_endif;
      continue;
    }
  }
  return kInvalidExpressionId;
}

ExpressionId ExpressionGraph::FindMatchingElse(ExpressionId id) const {
  CHECK(ExpressionForId(id).type() == ExpressionType::IF)
      << "FindClosingControlExpression is only valid on IF "
         "expressions.";

  // Traverse downwards
  for (ExpressionId i = id + 1; i < Size(); ++i) {
    const auto& expr = ExpressionForId(i);
    if (expr.type() == ExpressionType::ELSE) {
      // Found it!
      return i;
    } else if (expr.type() == ExpressionType::ENDIF) {
      // Found an endif even though we were looking for an ELSE.
      // -> Return invalidId
      return kInvalidExpressionId;
    } else if (expr.type() == ExpressionType::IF) {
      // Found a nested IF.
      // -> Jump over the block and continue behind it.
      auto matching_endif = FindMatchingEndif(i);
      if (matching_endif == kInvalidExpressionId) {
        return kInvalidExpressionId;
      }
      i = matching_endif;
      continue;
    }
  }
  return kInvalidExpressionId;
}

}  // namespace internal
}  // namespace ceres
