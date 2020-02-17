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
#include "ceres/internal/autodiff.h"
#include "ceres/random.h"
#include "ceres/sized_cost_function.h"
#include "test_util.h"
namespace ceres {
namespace internal {

// This struct is used to convert combined cost functions (evaluate method +
// operator()) to cost functors (only operator()). As a result, putting such a
// type into ceres::Autodiff will use the operator() instead of evaluate().
template <typename CostFunction>
struct CostFunctionToFunctor {
  template <typename... _Args>
  CostFunctionToFunctor(_Args&&... __args)
      : cost_function(std::forward<_Args>(__args)...) {}

  template <typename... _Args>
  bool operator()(_Args&&... __args) const {
    return cost_function(std::forward<_Args>(__args)...);
  }

  CostFunction cost_function;
};


// Evalutes sizes cost functions with either random parameters or all parameters set to a given value. The residuals and jacobians are returned as a pair.
template <int kNumResiduals, int... Ns>
std::pair<std::vector<double>, std::vector<double>> EvaluateSizedCostFunction(
    SizedCostFunction<kNumResiduals,Ns...>* f1, bool random_values = true, double value = 0) {
  using Params = StaticParameterDims<Ns...>;

  std::vector<double> params_array(Params::kNumParameters);
  std::vector<double*> params(Params::kNumParameters);
  std::vector<double> residuals_0(kNumResiduals, 0);
  std::vector<double> jacobians_array_0(kNumResiduals * Params::kNumParameters,
                                        0);

  for (auto& p : params_array) {
    if (random_values) {
      p = ceres::RandDouble() * 2.0 - 1.0;
    } else {
      p = value;
    }
  }
  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    params[i] = &params_array[k];
  }

  std::vector<double*> jacobians_0(Params::kNumParameterBlocks);
  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    jacobians_0[i] = &jacobians_array_0[k * kNumResiduals];
  }

  f1->Evaluate(params.data(), residuals_0.data(), jacobians_0.data());

  return std::make_pair(residuals_0, jacobians_array_0);
}

template <int kNumResiduals, int... Ns>
void CompareSizedCostFunctions(SizedCostFunction<kNumResiduals,Ns...>* f1,
                            SizedCostFunction<kNumResiduals,Ns...>* f2,
                            bool random_values = true,
                            double value = 0) {
  ceres::SetRandomState(956113);
  auto residuals_jacobians_1 =
      EvaluateSizedCostFunction(f1, random_values, value);
  ceres::SetRandomState(956113);
  auto residuals_jacobians_2 =
      EvaluateSizedCostFunction(f2, random_values, value);

  ExpectArraysClose(residuals_jacobians_1.first.size(),
                    residuals_jacobians_1.first.data(),
                    residuals_jacobians_2.first.data(),
                    1e-20);
  ExpectArraysClose(residuals_jacobians_1.second.size(),
                    residuals_jacobians_1.second.data(),
                    residuals_jacobians_2.second.data(),
                    1e-20);
}
}
}  // namespace test
