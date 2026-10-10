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
//          nikolaus@nikolaus-demmel.de (Nikolaus Demmel)
//          sameeragarwal@google.com (Sameer Agarwal)

#include <array>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"
#include "ceres/benchmark_cost_functions.h"
#include "ceres/ceres.h"

namespace ceres::internal {
namespace {

enum DiffType {
  kAutoDiff,
  kDynamicAutoDiff,
  kAutoDiffDynamicResiduals,
  kNumericForward,
  kDynamicNumericForward,
  kNumericCentral,
  kDynamicNumericCentral,
  kNumericRidders,
  kDynamicNumericRidders,
};

// Controls which outputs CostFunction::Evaluate is called to compute:
//   kResidualsOnly:             evaluate residuals only (jacobians = nullptr)
//   kResidualsAndJacobians:     evaluate residuals and all parameter Jacobians
//   kResidualsAndPointJacobian: evaluate residuals and only the 3D point
//                               Jacobian (at index CostFunctor::kPointBlockIndex)
enum EvaluationType {
  kResidualsOnly,
  kResidualsAndJacobians,
  kResidualsAndPointJacobian,
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

template <typename CostFunctor, DiffType kDiffType>
std::unique_ptr<CostFunction> CreateCostFunction() {
  if constexpr (kDiffType == kAutoDiff) {
    return CostFunctor::template CreateAutoDiff<CostFunctor>();
  } else if constexpr (kDiffType == kAutoDiffDynamicResiduals) {
    return CostFunctor::template CreateAutoDiffDynamicResiduals<CostFunctor>();
  } else if constexpr (kDiffType == kDynamicAutoDiff) {
    return CostFunctor::template CreateDynamicAutoDiff<CostFunctor>();
  } else if constexpr (IsDynamicNumericDiff<kDiffType>()) {
    return CostFunctor::template CreateDynamicNumericDiff<
        CostFunctor,
        NumericDiffMethod<kDiffType>()>();
  } else {
    return CostFunctor::template CreateNumericDiff<
        CostFunctor,
        NumericDiffMethod<kDiffType>()>();
  }
}

// Benchmarks CostFunction::Evaluate for `CostFunctor` wrapped with `kDiffType`
// and evaluated in mode `kEvalType`.
template <typename CostFunctor, DiffType kDiffType, EvaluationType kEvalType>
void BM_CostFunction(benchmark::State& state) {
  const std::unique_ptr<CostFunction> cost_function =
      CreateCostFunction<CostFunctor, kDiffType>();
  const auto parameter_storage = CostFunctor::MakeParameters();
  const auto parameters = parameter_storage.Pointers();

  double residuals[CostFunctor::kNumResiduals] = {};
  double jacobian_storage[CostFunctor::kNumResiduals *
                          CostFunctor::kTotalParameters] = {};
  double* jacobians[CostFunctor::kNumParameterBlocks] = {};

  double* cursor = jacobian_storage;
  for (int i = 0; i < CostFunctor::kNumParameterBlocks; ++i) {
    if constexpr (kEvalType == kResidualsAndJacobians) {
      jacobians[i] = cursor;
    } else if constexpr (kEvalType == kResidualsAndPointJacobian) {
      static_assert(CostFunctor::kPointBlockIndex >= 0 &&
                    CostFunctor::kPointBlockIndex <
                        CostFunctor::kNumParameterBlocks);
      if (i == CostFunctor::kPointBlockIndex) {
        jacobians[i] = cursor;
      }
    }
    cursor += CostFunctor::kNumResiduals * CostFunctor::kBlockSizes[i];
  }
  double** jacobians_ptr = (kEvalType == kResidualsOnly) ? nullptr : jacobians;

  CHECK(cost_function->Evaluate(parameters.data(), residuals, jacobians_ptr));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters.data(), residuals, jacobians_ptr));
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(jacobian_storage);
  }
}

#define REGISTER_AUTODIFF_BENCHMARKS(Functor)                               \
  BENCHMARK_TEMPLATE(BM_CostFunction, Functor, kAutoDiff, kResidualsOnly);  \
  BENCHMARK_TEMPLATE(                                                       \
      BM_CostFunction, Functor, kAutoDiff, kResidualsAndJacobians);         \
  BENCHMARK_TEMPLATE(                                                       \
      BM_CostFunction, Functor, kDynamicAutoDiff, kResidualsOnly);          \
  BENCHMARK_TEMPLATE(                                                       \
      BM_CostFunction, Functor, kDynamicAutoDiff, kResidualsAndJacobians)

#define REGISTER_CONSTANT_BENCHMARKS(N)              \
  BENCHMARK_TEMPLATE(BM_CostFunction,                \
                     ConstantCostFunction<N>,        \
                     kAutoDiff,                      \
                     kResidualsAndJacobians);        \
  BENCHMARK_TEMPLATE(BM_CostFunction,                \
                     ConstantCostFunction<N>,        \
                     kDynamicAutoDiff,               \
                     kResidualsAndJacobians)

REGISTER_CONSTANT_BENCHMARKS(1);
REGISTER_CONSTANT_BENCHMARKS(10);
REGISTER_CONSTANT_BENCHMARKS(20);
REGISTER_CONSTANT_BENCHMARKS(30);
REGISTER_CONSTANT_BENCHMARKS(40);
REGISTER_CONSTANT_BENCHMARKS(50);
REGISTER_CONSTANT_BENCHMARKS(60);

#undef REGISTER_CONSTANT_BENCHMARKS

REGISTER_AUTODIFF_BENCHMARKS(LinearCostFunction<1>);
REGISTER_AUTODIFF_BENCHMARKS(LinearCostFunction<10>);

#define REGISTER_RAT43_BENCHMARKS(Diff)                                \
  BENCHMARK_TEMPLATE(                                                  \
      BM_CostFunction, Rat43CostFunctor, Diff, kResidualsOnly);        \
  BENCHMARK_TEMPLATE(                                                  \
      BM_CostFunction, Rat43CostFunctor, Diff, kResidualsAndJacobians)

REGISTER_RAT43_BENCHMARKS(kAutoDiff);
REGISTER_RAT43_BENCHMARKS(kDynamicAutoDiff);
REGISTER_RAT43_BENCHMARKS(kNumericForward);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericForward);
REGISTER_RAT43_BENCHMARKS(kNumericCentral);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericCentral);
REGISTER_RAT43_BENCHMARKS(kNumericRidders);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericRidders);

#undef REGISTER_RAT43_BENCHMARKS

#define REGISTER_SNAVELY_ALL_MODES(Diff)                                    \
  BENCHMARK_TEMPLATE(                                                       \
      BM_CostFunction, SnavelyReprojectionError, Diff, kResidualsOnly);     \
  BENCHMARK_TEMPLATE(BM_CostFunction,                                       \
                     SnavelyReprojectionError,                              \
                     Diff,                                                  \
                     kResidualsAndJacobians);                               \
  BENCHMARK_TEMPLATE(BM_CostFunction,                                       \
                     SnavelyReprojectionError,                              \
                     Diff,                                                  \
                     kResidualsAndPointJacobian)

REGISTER_SNAVELY_ALL_MODES(kAutoDiff);
REGISTER_SNAVELY_ALL_MODES(kDynamicAutoDiff);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   SnavelyReprojectionError,
                   kAutoDiffDynamicResiduals,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   SnavelyReprojectionError,
                   kAutoDiffDynamicResiduals,
                   kResidualsAndJacobians);
REGISTER_SNAVELY_ALL_MODES(kNumericForward);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericForward);
REGISTER_SNAVELY_ALL_MODES(kNumericCentral);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericCentral);
BENCHMARK_TEMPLATE(
    BM_CostFunction, SnavelyReprojectionError, kNumericRidders, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   SnavelyReprojectionError,
                   kNumericRidders,
                   kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   SnavelyReprojectionError,
                   kDynamicNumericRidders,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   SnavelyReprojectionError,
                   kDynamicNumericRidders,
                   kResidualsAndJacobians);

#undef REGISTER_SNAVELY_ALL_MODES

REGISTER_AUTODIFF_BENCHMARKS(PhotometricError<8>);

REGISTER_AUTODIFF_BENCHMARKS(RelativePoseError);
BENCHMARK_TEMPLATE(
    BM_CostFunction, RelativePoseError, kNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   RelativePoseError,
                   kNumericCentral,
                   kResidualsAndJacobians);
BENCHMARK_TEMPLATE(
    BM_CostFunction, RelativePoseError, kDynamicNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   RelativePoseError,
                   kDynamicNumericCentral,
                   kResidualsAndJacobians);

REGISTER_AUTODIFF_BENCHMARKS(Brdf);

BENCHMARK_TEMPLATE(BM_CostFunction,
                   OuterProjectionFunctor<false>,
                   kAutoDiff,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   OuterProjectionFunctor<false>,
                   kAutoDiff,
                   kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   OuterProjectionFunctor<true>,
                   kAutoDiff,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunction,
                   OuterProjectionFunctor<true>,
                   kAutoDiff,
                   kResidualsAndJacobians);

#define REGISTER_REAL_WORLD_BM(Functor)              \
  REGISTER_AUTODIFF_BENCHMARKS(Functor);             \
  BENCHMARK_TEMPLATE(BM_CostFunction,                \
                     Functor,                        \
                     kNumericCentral,                \
                     kResidualsAndJacobians);        \
  BENCHMARK_TEMPLATE(BM_CostFunction,                \
                     Functor,                        \
                     kDynamicNumericCentral,         \
                     kResidualsAndJacobians)

#define REGISTER_REAL_WORLD_POINT_JAC_BM(Functor)         \
  BENCHMARK_TEMPLATE(BM_CostFunction,                     \
                     Functor,                             \
                     kAutoDiff,                           \
                     kResidualsAndPointJacobian);         \
  BENCHMARK_TEMPLATE(BM_CostFunction,                     \
                     Functor,                             \
                     kDynamicAutoDiff,                    \
                     kResidualsAndPointJacobian);         \
  BENCHMARK_TEMPLATE(BM_CostFunction,                     \
                     Functor,                             \
                     kNumericCentral,                     \
                     kResidualsAndPointJacobian)

REGISTER_REAL_WORLD_BM(ColmapOpenCVReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(ColmapOpenCVReprojectionError);
REGISTER_REAL_WORLD_BM(ColmapFullOpenCVReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(ColmapFullOpenCVReprojectionError);
REGISTER_REAL_WORLD_BM(ColmapRigReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(ColmapRigReprojectionError);
REGISTER_REAL_WORLD_BM(ColmapSampsonError);
REGISTER_REAL_WORLD_BM(ColmapFisheyeReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(ColmapFisheyeReprojectionError);
REGISTER_REAL_WORLD_BM(ColmapSphericalReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(ColmapSphericalReprojectionError);
REGISTER_REAL_WORLD_BM(LibmvBrownReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(LibmvBrownReprojectionError);
REGISTER_REAL_WORLD_BM(LibmvNukeInvertReprojectionError);
REGISTER_REAL_WORLD_POINT_JAC_BM(LibmvNukeInvertReprojectionError);
REGISTER_REAL_WORLD_BM(PoseGraph3dError);

#undef REGISTER_REAL_WORLD_POINT_JAC_BM
#undef REGISTER_REAL_WORLD_BM
#undef REGISTER_AUTODIFF_BENCHMARKS

}  // namespace
}  // namespace ceres::internal

BENCHMARK_MAIN();
