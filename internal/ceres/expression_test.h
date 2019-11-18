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

#ifndef CERES_PUBLIC_EXPRESSION_TEST_H_
#define CERES_PUBLIC_EXPRESSION_TEST_H_

#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"

#include "gtest/gtest.h"

// This file adds a few helper functions to test Expressions and
// ExpressionGraphs for correctness.
namespace ceres {
namespace internal {

inline void InsertExpression(ExpressionGraph& graph,
                             ExpressionId id,
                             ExpressionType type,
                             ExpressionId lhs_id,
                             double value,
                             const std::string& name,
                             const std::vector<ExpressionId>& arguments) {
  //  EXPECT_EQ(static_cast<int>(expr.type()), static_cast<int>(type));
  //  EXPECT_EQ(expr.lhs_id(), lhs_id);
  //  EXPECT_EQ(expr.value(), value);
  //  EXPECT_EQ(expr.name(), name);
  //  EXPECT_EQ(expr.arguments(), arguments);
  graph.InsertExpression(id, type, lhs_id, arguments, name, value);
}

#define INSERT_EXPRESSION(_graph, _id, _type, _lhs_id, _value, _name, ...) \
  InsertExpression(_graph,                                                 \
                   _id,                                                    \
                   ExpressionType::_type,                                  \
                   _lhs_id,                                                \
                   _value,                                                 \
                   _name,                                                  \
                   {__VA_ARGS__})

}  // namespace internal
}  // namespace ceres

#endif
