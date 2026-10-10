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
// Authors: darius.rueckert@fau.de (Darius Rueckert)
//          sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_RAT43_COST_FUNCTION_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_RAT43_COST_FUNCTION_H_

#include <cmath>
#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/cost_function_benchmark_utils.h"

namespace ceres {

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  inline bool operator()(const T* parameters, T* residuals) const {
    const T& b1 = parameters[0];
    const T& b2 = parameters[1];
    const T& b3 = parameters[2];
    const T& b4 = parameters[3];
    residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
    return true;
  }

 private:
  const double x_;
  const double y_;
};

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Rat43(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals = 0.0;
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Rat43CostFunctor, 1, 4>(
          x, y);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, &residuals, jacobians_ptr));
  }
}

#define REGISTER_RAT43_BENCHMARKS(Diff)                        \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsOnly);          \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsAndJacobians)

REGISTER_RAT43_BENCHMARKS(kAutoDiff);
REGISTER_RAT43_BENCHMARKS(kDynamicAutoDiff);
REGISTER_RAT43_BENCHMARKS(kNumericForward);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericForward);
REGISTER_RAT43_BENCHMARKS(kNumericCentral);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericCentral);
REGISTER_RAT43_BENCHMARKS(kNumericRidders);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericRidders);

#undef REGISTER_RAT43_BENCHMARKS

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_RAT43_COST_FUNCTION_H_
