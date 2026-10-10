// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
//
#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_CONSTANT_COST_FUNCTIONS_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_CONSTANT_COST_FUNCTIONS_H_

#include <array>
#include <memory>
#include <numeric>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/cost_function_benchmark_utils.h"

namespace ceres {

template <int kParameterBlockSize>
struct ConstantCostFunction {
  template <typename T>
  inline bool operator()(const T* const x, T* residuals) const {
    residuals[0] = T(5);
    return true;
  }
};

template <int kParameterBlockSize, DiffType kDiffType>
static void BM_Constant(benchmark::State& state) {
  constexpr int kNumResiduals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, kNumResiduals> residuals{};
  std::array<double, kNumResiduals * kParameterBlockSize> jacobian_values{};
  double* jacobians[] = {jacobian_values.data()};

  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<ConstantCostFunction<kParameterBlockSize>,
                                  1,
                                  kParameterBlockSize>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals.data(), jacobians));
  }
}

#define REGISTER_CONSTANT_BENCHMARKS(N)          \
  BENCHMARK_TEMPLATE(BM_Constant, N, kAutoDiff); \
  BENCHMARK_TEMPLATE(BM_Constant, N, kDynamicAutoDiff)

REGISTER_CONSTANT_BENCHMARKS(1);
REGISTER_CONSTANT_BENCHMARKS(10);
REGISTER_CONSTANT_BENCHMARKS(20);
REGISTER_CONSTANT_BENCHMARKS(30);
REGISTER_CONSTANT_BENCHMARKS(40);
REGISTER_CONSTANT_BENCHMARKS(50);
REGISTER_CONSTANT_BENCHMARKS(60);

#undef REGISTER_CONSTANT_BENCHMARKS

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_CONSTANT_COST_FUNCTIONS_H_
