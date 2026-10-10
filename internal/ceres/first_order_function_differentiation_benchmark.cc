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

#include "absl/log/check.h"
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

// Controls whether FirstOrderFunction::Evaluate is called to compute only the
// objective value (gradient = nullptr) or both the objective value and the
// gradient vector (gradient != nullptr).
enum EvaluationType {
  kCostOnly,
  kCostAndGradient,
};

template <DiffType kDiffType>
constexpr NumericDiffMethodType NumericDiffMethod() {
  if constexpr (kDiffType == kNumericForward ||
                kDiffType == kDynamicNumericForward) {
    return FORWARD;
  } else if constexpr (kDiffType == kNumericCentral ||
                       kDiffType == kDynamicNumericCentral) {
    return CENTRAL;
  } else if constexpr (kDiffType == kNumericRidders ||
                       kDiffType == kDynamicNumericRidders) {
    return RIDDERS;
  }
}

template <DiffType kDiffType>
constexpr bool IsDynamicNumericDiff() {
  return kDiffType == kDynamicNumericForward ||
         kDiffType == kDynamicNumericCentral ||
         kDiffType == kDynamicNumericRidders;
}

template <int kParameters>
struct RosenbrockFirstOrderFunctor {
  static constexpr int kNumParameters = kParameters;

  static std::array<double, kNumParameters> MakeParameters() {
    std::array<double, kNumParameters> x{};
    for (int i = 0; i < kNumParameters; ++i) {
      x[i] = 0.5 + 0.01 * i;
    }
    return x;
  }

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

template <DiffType kDiffType, typename Functor>
std::unique_ptr<FirstOrderFunction> CreateFirstOrderFunction() {
  constexpr int kNumParameters = Functor::kNumParameters;
  if constexpr (kDiffType == kAutoDiff) {
    return std::make_unique<AutoDiffFirstOrderFunction<Functor, kNumParameters>>(
        std::make_unique<Functor>());
  } else if constexpr (IsDynamicNumericDiff<kDiffType>()) {
    constexpr NumericDiffMethodType kMethod = NumericDiffMethod<kDiffType>();
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, kMethod, DYNAMIC>>(
        std::make_unique<Functor>(), kNumParameters);
  } else {
    constexpr NumericDiffMethodType kMethod = NumericDiffMethod<kDiffType>();
    return std::make_unique<
        NumericDiffFirstOrderFunction<Functor, kMethod, kNumParameters>>(
        std::make_unique<Functor>());
  }
}

template <typename Functor, DiffType kDiffType, EvaluationType kEvalType>
void BM_FirstOrderFunction(benchmark::State& state) {
  const std::unique_ptr<FirstOrderFunction> function =
      CreateFirstOrderFunction<kDiffType, Functor>();
  const auto x = Functor::MakeParameters();

  double cost = 0.0;
  std::array<double, Functor::kNumParameters> gradient{};
  double* gradient_ptr =
      (kEvalType == kCostAndGradient) ? gradient.data() : nullptr;

  CHECK(function->Evaluate(x.data(), &cost, gradient_ptr));

  for (auto _ : state) {
    benchmark::DoNotOptimize(function->Evaluate(x.data(), &cost, gradient_ptr));
    benchmark::DoNotOptimize(cost);
    benchmark::DoNotOptimize(gradient);
  }
}

#define REGISTER_ROSENBROCK_FOR_DIFF_TYPE(Diff, N)   \
  BENCHMARK_TEMPLATE(BM_FirstOrderFunction,          \
                     RosenbrockFirstOrderFunctor<N>, \
                     Diff,                           \
                     kCostOnly);                     \
  BENCHMARK_TEMPLATE(BM_FirstOrderFunction,          \
                     RosenbrockFirstOrderFunctor<N>, \
                     Diff,                           \
                     kCostAndGradient)

#define REGISTER_ROSENBROCK_BENCHMARKS(N)                       \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kAutoDiff, N);              \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kNumericForward, N);        \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kDynamicNumericForward, N); \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kNumericCentral, N);        \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kDynamicNumericCentral, N); \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kNumericRidders, N);        \
  REGISTER_ROSENBROCK_FOR_DIFF_TYPE(kDynamicNumericRidders, N)

REGISTER_ROSENBROCK_BENCHMARKS(4);
REGISTER_ROSENBROCK_BENCHMARKS(10);
REGISTER_ROSENBROCK_BENCHMARKS(16);
REGISTER_ROSENBROCK_BENCHMARKS(24);

#undef REGISTER_ROSENBROCK_BENCHMARKS
#undef REGISTER_ROSENBROCK_FOR_DIFF_TYPE

}  // namespace
}  // namespace ceres::internal

BENCHMARK_MAIN();
