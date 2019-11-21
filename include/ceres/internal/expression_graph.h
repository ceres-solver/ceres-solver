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

namespace ceres {
namespace internal {

// A directed, acyclic, unconnected graph containing all expressions of a
// program.
//
// The expression graph is stored linear in the expressions_ array. The order is
// identical to the execution order. Each expression can have multiple children
// and multiple parents.
// A is child of B     <=>  B has A as a parameter    <=> B.DirectlyDependsOn(A)
// A is parent of B    <=>  A has B as a parameter    <=> A.DirectlyDependsOn(B)
class ExpressionGraph {
 public:
  // Creates an arithmetic expression of the following form:
  // <lhs> = <rhs>;
  //
  // For example:
  //   CreateArithmeticExpression(PLUS, 5)
  // will generate:
  //   v_5 = __ + __;
  // The place holders are then set by the CreateXX functions of Expression.
  //
  // If lhs_id == kInvalidExpressionId, then a new lhs_id will be generated and
  // assigned to the created expression.
  // Calling this function with a lhs_id that doesn't exist results in an
  // error.
  Expression& CreateArithmeticExpression(ExpressionType type,
                                         ExpressionId lhs_id);

  // Control expression don't have a left hand side.
  // Supported types: IF/ELSE/ENDIF/NOP
  Expression& CreateControlExpression(ExpressionType type);

  // Checks if A depends on B.
  // -> B is a descendant of A
  bool DependsOn(ExpressionId A, ExpressionId B) const;

  bool operator==(const ExpressionGraph& other) const;

  Expression& ExpressionForId(ExpressionId id) { return expressions_[id]; }
  const Expression& ExpressionForId(ExpressionId id) const {
    return expressions_[id];
  }

  int Size() const { return expressions_.size(); }

  // Insert a new expression at "location" into the graph. All expression
  // after "location" are moved by one element to the back. References to moved
  // expression are updated.
  void InsertExpression(ExpressionId location,
                        ExpressionType type,
                        ExpressionId lhs_id,
                        const std::vector<ExpressionId>& arguments,
                        const std::string& name,
                        double value);

 private:
  // All Expressions are referenced by an ExpressionId. The ExpressionId is the
  // index into this array. Each expression has a list of ExpressionId as
  // arguments. These references form the graph.
  std::vector<Expression> expressions_;
};

// After calling this method, all operations on 'ExpressionRef' objects will be
// recorded into an ExpressionGraph. You can obtain this graph by calling
// StopRecordingExpressions.
//
// Performing expression operations before calling StartRecordingExpressions or
// calling StartRecodring. twice is an error.
void StartRecordingExpressions();

// Stops recording and returns all expressions that have been executed since the
// call to StartRecordingExpressions. The internal ExpressionGraph will be
// invalidated and a second consecutive call to this method results in an error.
ExpressionGraph StopRecordingExpressions();

// Returns a pointer to the active expression tree.
// Normal users should not use this functions.
ExpressionGraph* GetCurrentExpressionGraph();

}  // namespace internal
}  // namespace ceres
#endif
