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
#ifndef CERES_PUBLIC_AUTODIFF_CODE_GEN_H_
#define CERES_PUBLIC_AUTODIFF_CODE_GEN_H_

#include "ceres/autodiff_codegen_cost_function.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/code_generator.h"
#include "ceres/internal/expression_graph.h"
#include "ceres/internal/expression_ref.h"
#include "ceres/jet.h"

#include <fstream>

namespace ceres {

template <typename CostFunctor>
class AutoDiffCodeGenerator {
 public:
  AutoDiffCodeGenerator() {
    using ParameterDims = typename CostFunctor::ParameterDims;
    using T = internal::ExpressionRef;
    using JetT = Jet<T, ParameterDims::kNumParameters>;
    using Parameters = typename ParameterDims::Parameters;
    constexpr int kNumParameters = ParameterDims::kNumParameters;
    constexpr int kNumParameterBlocks = ParameterDims::kNumParameterBlocks;
    constexpr int kNumResiduals = CostFunctor::kNumResiduals;

    // Creating the arrays below will zero-initialize the jets. We need
    // expression recording for that.
    internal::StartRecordingExpressions();

    std::array<JetT, kNumParameters> all_parameters;
    std::array<JetT, kNumResiduals> residuals;
    std::array<JetT*, kNumParameterBlocks> unpacked_parameters =
        ParameterDims::GetUnpackedParameters(all_parameters.data());

    // Create input expressions
    // v_0 = parameters[0][0]
    // v_1 = parameters[0][1]
    // ...
    for (int i = 0; i < kNumParameterBlocks; ++i) {
      for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
        JetT& J = unpacked_parameters[i][j];
        J.a = internal::MakeInputAssignment<T>(
            0.0,
            ("parameters[" + std::to_string(i) + "][" + std::to_string(j) + "]")
                .c_str());
      }
    }

    // Initialize Jet partial derivatives
    for (int i = 0; i < kNumParameters; ++i) {
      all_parameters[i].v(i) = T(1);
    }

    // Generate graph from the cost functor
    internal::VariadicEvaluate<ParameterDims>(
        functor_, unpacked_parameters.data(), residuals.data());

    // Create output expressions for the residuals:
    // residuals[0] = v_200;
    // residuals[1] = v_201;
    // ...
    for (int i = 0; i < kNumResiduals; ++i) {
      auto& J = residuals[i];
      internal::MakeOutput(J.a, "residuals[" + std::to_string(i) + "]");
    }

    // Make a copy of the current graph
    graph_residual = *internal::GetCurrentExpressionGraph();

    // Create output expressions for the jacobians:
    // jacobians[0][0] = v_351;
    // jacobians[0][1] = v_352;
    // ...
    for (int i = 0, total_param_id = 0; i < kNumParameterBlocks;
         ++i, total_param_id += ParameterDims::GetDim(i)) {
      for (int r = 0; r < kNumResiduals; ++r) {
        for (int j = 0; j < ParameterDims::GetDim(i); ++j) {
          auto& J = residuals[r];
          // all partial derivatives
          internal::MakeOutput(
              (J.v[total_param_id + j]),
              "jacobians[" + std::to_string(i) + "][" +
                  std::to_string(r * ParameterDims::GetDim(i) + j) + "]");
        }
      }
    }
    graph_residual_and_jacobian = internal::StopRecordingExpressions();
  }

  void ToFile(const std::string& name) {
    // Write to file
    std::ofstream file(name);

    file << "// This file is generated with ceres::AutoDiffCodeGen."
         << std::endl;
    file << "// http://code.google.com/p/ceres-solver/" << std::endl;
    file << std::endl;

    GenerateAndAppendFunction(
        file,
        graph_residual,
        "void EvaluateResidual(double const* const* parameters, double* "
        "residuals)");

    file << std::endl;

    GenerateAndAppendFunction(
        file,
        graph_residual_and_jacobian,
        "void EvaluateResidualAndJacobian(double const* const* parameters, "
        "double* "
        "residuals, double** jacobians)");

    // let's also create a combined function
    // clang-format off
    std::string combined =
        "bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) \n"
        "{\n"
        "   if (residuals && jacobians) {\n"
        "     EvaluateResidualAndJacobian(parameters,residuals,jacobians);\n"
        "   }\n"
        "   else if (residuals) {\n"
        "     EvaluateResidual(parameters,residuals);\n"
        "   }\n"
        "   // Returning from a cost functor is not supported yet.\n"
        "   return true;\n"
        "}\n";
    // clang-format on

    file << std::endl << combined << std::endl;
  }

 private:
  void GenerateAndAppendFunction(std::ostream& file,
                                 internal::ExpressionGraph& graph,
                                 const std::string& name) {
    // TODO:
    // Run optimizer on 'graph'

    internal::CodeGenerator::Options generator_options;
    generator_options.function_name = name;
    internal::CodeGenerator gen(graph, generator_options);
    auto code = gen.Generate();

    for (auto& line : code) {
      file << line << std::endl;
    }
  }

  CostFunctor functor_;
  internal::ExpressionGraph graph_residual, graph_residual_and_jacobian;
};

}  // namespace ceres
#endif  // CERES_PUBLIC_AUTODIFF_CODE_GEN_H_
