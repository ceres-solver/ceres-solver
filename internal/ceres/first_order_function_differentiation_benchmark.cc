// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include <array>
#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_first_order_function.h"
#include "ceres/first_order_function.h"
#include "ceres/numeric_diff_first_order_function.h"

namespace ceres::internal {
namespace {

enum DiffType {
  kAutoDiff,
  kNumericForward,
  kDynamicNumericForward,
  kNumericCentral,
  kDynamicNumericCentral,
  kNumericRidders,
  kDynamicNumericRidders,
};

template <int kNumParameters>
struct RosenbrockFirstOrderFunctor {
  template <typename T>
  bool operator()(const T* const x, T* cost) const {
    T sum = T(0.0);
    for (int i = 0; i < kNumParameters - 1; ++i) {
      const T t1 = T(1.0) - x[i];
      const T t2 = x[i + 1] - x[i] * x[i];
      sum += t1 * t1 + T(100.0) * t2 * t2;
    }
    *cost = sum;
    return true;
  }
};

template <DiffType kDiffType, typename Functor, int kNumParameters>
std::unique_ptr<FirstOrderFunction> CreateFirstOrderFunction() {
  if constexpr (kDiffType == kAutoDiff) {
    return std::make_unique<AutoDiffFirstOrderFunction<Functor, kNumParameters>>(
        std::make_unique<Functor>());
  } else if constexpr (kDiffType == kNumericForward) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, FORWARD, kNumParameters>>(
        std::make_unique<Functor>());
  } else if constexpr (kDiffType == kDynamicNumericForward) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, FORWARD, DYNAMIC>>(
        std::make_unique<Functor>(), kNumParameters);
  } else if constexpr (kDiffType == kNumericCentral) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, CENTRAL, kNumParameters>>(
        std::make_unique<Functor>());
  } else if constexpr (kDiffType == kDynamicNumericCentral) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, CENTRAL, DYNAMIC>>(
        std::make_unique<Functor>(), kNumParameters);
  } else if constexpr (kDiffType == kNumericRidders) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, RIDDERS, kNumParameters>>(
        std::make_unique<Functor>());
  } else if constexpr (kDiffType == kDynamicNumericRidders) {
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, RIDDERS, DYNAMIC>>(
        std::make_unique<Functor>(), kNumParameters);
  }
}

template <DiffType kDiffType, int kNumParameters>
void BM_Rosenbrock(benchmark::State& state) {
  using Functor = RosenbrockFirstOrderFunctor<kNumParameters>;
  auto function = CreateFirstOrderFunction<kDiffType, Functor, kNumParameters>();

  std::array<double, kNumParameters> x{};
  for (int i = 0; i < kNumParameters; ++i) {
    x[i] = 0.5 + 0.01 * i;
  }
  double cost = 0.0;
  std::array<double, kNumParameters> gradient{};
  double* gradient_ptr = state.range(0) ? gradient.data() : nullptr;

  for (auto _ : state) {
    benchmark::DoNotOptimize(function);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(function->Evaluate(x.data(), &cost, gradient_ptr));
    benchmark::DoNotOptimize(cost);
    benchmark::DoNotOptimize(gradient);
  }
}

#define REGISTER_ROSENBROCK_BENCHMARKS(N)                                     \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kAutoDiff, N)->Arg(0)->Arg(1);            \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kNumericForward, N)->Arg(0)->Arg(1);      \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kDynamicNumericForward, N)                \
      ->Arg(0)                                                                \
      ->Arg(1);                                                               \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kNumericCentral, N)->Arg(0)->Arg(1);      \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kDynamicNumericCentral, N)                \
      ->Arg(0)                                                                \
      ->Arg(1);                                                               \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kNumericRidders, N)->Arg(1);              \
  BENCHMARK_TEMPLATE(BM_Rosenbrock, kDynamicNumericRidders, N)->Arg(1)

REGISTER_ROSENBROCK_BENCHMARKS(4);
REGISTER_ROSENBROCK_BENCHMARKS(10);
REGISTER_ROSENBROCK_BENCHMARKS(16);
REGISTER_ROSENBROCK_BENCHMARKS(24);

#undef REGISTER_ROSENBROCK_BENCHMARKS

}  // namespace
}  // namespace ceres::internal

BENCHMARK_MAIN();
