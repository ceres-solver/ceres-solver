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
#ifndef CERES_PUBLIC_CODEGEN_INTERNAL_OPTIMIZER_H_
#define CERES_PUBLIC_CODEGEN_INTERNAL_OPTIMIZER_H_

#include <memory>
#include <vector>

#include "ceres/codegen/internal/expression_graph.h"
#include "ceres/codegen/internal/optimization_pass.h"

namespace ceres {
namespace internal {

// The Optimizer manages and applies OptimizationPasses to an ExpressionGraph.
// This will change the ExpressionGraph, but the generated code will output the
// same numerical values.
//
// The Optimizer operates in the following way (pseudo code):
//
//   while (true) {
//      graph.hasChanged = false;
//      for each pass in OptimizationPasses
//         pass.applyTo(graph)
//      if(!graph.hasChanged)
//         break;
//   }
//
class Optimizer {
 public:
  struct Options {
    int max_iterations = 100;

    bool pass_nop_cleanup = true;
  };

  Optimizer(const Options& options);

  // Run the optimizer on the given graph.
  // Return: The number of required iterations.
  int run(ExpressionGraph& graph) const;

 private:
  const Options options_;
  std::vector<std::unique_ptr<OptimizationPass>> optimizaton_passes_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_CODEGEN_INTERNAL_CODE_GENERATOR_H_
