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

#ifndef CERES_PUBLIC_EXPRESSION_TREE_H_
#define CERES_PUBLIC_EXPRESSION_TREE_H_

#include <vector>

#include "expression.h"
#include "expression_ref.h"

namespace ceres {
namespace internal {

// The expression tree is stored linear in the data_ array. The order is
// identical to the execution order. Each expression can have multiple children
// and multiple parents.
// A is child of B     <=>  B has A as a parameter    <=> B.DirectlyDependsOn(A)
// A is parent of B    <=>  A has B as a parameter    <=> A.DirectlyDependsOn(B)
//
// Note:
// This is not a tree.
// It's an undirected, non-cyclic, unconnected graph.
class ExpressionTree {
 public:
  // Creates an expression and adds it to data_.
  // The returned reference will be invalid after this function is called again.
  Expression& MakeExpression(ExpressionType type);

  // Checks if A depends on B.
  // -> B is a descendant of A
  bool DependsOn(ExpressionRef A, ExpressionRef B) const;

  Expression& get(ExpressionRef id) { return data_[id.id]; }
  const Expression& get(ExpressionRef id) const { return data_[id.id]; }

 private:
  std::vector<Expression> data_;
};

// After calling this method, all operations on 'ExpressionRef' objects will be
// recorded into an internal array. You can obtain this array by calling
// StopRecordingExpressions.
//
// Performing expression operations before calling StartRecordingExpressions is
// an error.
void StartRecordingExpressions();

// Stops recording and returns all expressions that have been executed since the
// call to StartRecordingExpressions.
ExpressionTree StopRecordingExpressions();

// Returns a pointer to the active expression tree.
// Normal users should not use this functions.
ExpressionTree* GetCurrentExpressionTree();

}  // namespace internal
}  // namespace ceres
#endif
