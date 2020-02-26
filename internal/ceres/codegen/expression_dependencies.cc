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

#include <iostream>

#include "glog/logging.h"
namespace ceres {
namespace internal {

ExpressionDependencies::ExpressionDependencies(const ExpressionGraph& graph)
    : data_(graph.Size()), graph_(graph) {
  Rebuild();
}

void ExpressionDependencies::Rebuild() {
  for (auto& d : data_) {
    d.used_by.clear();
    d.written_to.clear();
  }

  for (ExpressionId id = 0; id < graph_.Size(); ++id) {
    auto& expr = graph_.ExpressionForId(id);
    auto& data = data_[id];

    if (expr.HasValidLhs()) {
      data_[expr.lhs_id()].written_to.push_back(id);
    }

    for (auto arg : expr.arguments()) {
      data_[arg].used_by.push_back(id);
    }
  }
}

const ExpressionDependencies::Data& ExpressionDependencies::DataForExpressionId(
    ExpressionId id) const {
  CHECK_NE(id, kInvalidExpressionId);
  CHECK_NE(graph_.ExpressionForId(id).lhs_id(), kInvalidExpressionId);
  CHECK_EQ(graph_.ExpressionForId(id).lhs_id(), id);
  return data_[id];
}

}  // namespace internal
}  // namespace ceres
