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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_OPTIMIZE_EXPRESSION_GRAPH_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_OPTIMIZE_EXPRESSION_GRAPH_H_

#include <memory>
#include <vector>

#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass_summary.h"

namespace ceres {
namespace internal {

struct OptimizeExpressionGraphOptions {
  int max_num_iterations = 100;
  bool eliminate_nops = true;
};

struct OptimizeExpressionGraphSummary {
  int num_iterations;
  std::vector<OptimizationPassSummary> summaries;
};

// Optimize the given ExpressionGraph in-place according to the defined
// OptimizeExpressionGraphOptions. This will change the ExpressionGraph, but the
// generated code will output the same numerical values.
//
// The Optimization iteratively applies all OptimizationPasses until the graph
// does not change anymore or max_num_iterations is reached. Pseudo Code:
//
//   for(int it = 0; it < max_num_iterations; ++it) {
//      graph.hasChanged = false;
//      for each pass in OptimizationPasses
//         pass.applyTo(graph)
//      if(!graph.hasChanged)
//         break;
//   }
//
OptimizeExpressionGraphSummary OptimizeExpressionGraph(
    const OptimizeExpressionGraphOptions& options, ExpressionGraph* graph);

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_OPTIMIZE_EXPRESSION_GRAPH_H_
