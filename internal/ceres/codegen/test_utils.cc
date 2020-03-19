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

#include "ceres/codegen/test_utils.h"
#include "ceres/test_util.h"

namespace ceres {
namespace internal {

std::pair<std::vector<double>, std::vector<double> > EvaluateCostFunction(
    CostFunction* cost_function, double value) {
  auto num_residuals = cost_function->num_residuals();
  auto parameter_block_sizes = cost_function->parameter_block_sizes();
  auto num_parameter_blocks = parameter_block_sizes.size();

  int total_num_parameters = 0;
  for (auto block_size : parameter_block_sizes) {
    total_num_parameters += block_size;
  }

  std::vector<double> params_array(total_num_parameters, value);
  std::vector<double*> params(num_parameter_blocks);
  std::vector<double> residuals(num_residuals, 0);
  std::vector<double> jacobians_array(num_residuals * total_num_parameters, 0);
  std::vector<double*> jacobians(num_parameter_blocks);

  for (int i = 0, k = 0; i < num_parameter_blocks;
       k += parameter_block_sizes[i], ++i) {
    params[i] = &params_array[k];
  }

  for (int i = 0, k = 0; i < num_parameter_blocks;
       k += parameter_block_sizes[i], ++i) {
    jacobians[i] = &jacobians_array[k * num_residuals];
  }

  cost_function->Evaluate(params.data(), residuals.data(), jacobians.data());

  return std::make_pair(residuals, jacobians_array);
}

void CompareCostFunctions(CostFunction* cost_function1,
                          CostFunction* cost_function2,

                          double value,
                          double tol) {
  auto residuals_and_jacobians_1 = EvaluateCostFunction(cost_function1, value);
  auto residuals_and_jacobians_2 = EvaluateCostFunction(cost_function2, value);

  ExpectArraysClose(residuals_and_jacobians_1.first.size(),
                    residuals_and_jacobians_1.first.data(),
                    residuals_and_jacobians_2.first.data(),
                    tol);
  ExpectArraysClose(residuals_and_jacobians_1.second.size(),
                    residuals_and_jacobians_1.second.data(),
                    residuals_and_jacobians_2.second.data(),
                    tol);
}

}  // namespace internal
}  // namespace ceres
