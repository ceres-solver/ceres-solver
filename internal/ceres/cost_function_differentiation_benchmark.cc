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

#include <cmath>
#include <memory>
#include <type_traits>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/constant_cost_function.h"
#include "ceres/autodiff_benchmarks/cost_function_factory.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/photometric_error.h"
#include "ceres/autodiff_benchmarks/relative_pose_error.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace ceres {
namespace {

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
void BM_Rat43(benchmark::State& state) {
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

// ============================================================================
// CostFunctionToFunctor / DynamicCostFunctionToFunctor Benchmarks
// ============================================================================

struct InnerProjectionFunctor {
  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    const T xp = point[0] / point[2];
    const T yp = point[1] / point[2];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (intrinsics[1] + intrinsics[2] * r2);
    residuals[0] = intrinsics[0] * distortion * xp;
    residuals[1] = intrinsics[0] * distortion * yp;
    return true;
  }
};

template <DiffType kDiffType>
struct OuterProjectionFunctor {
  OuterProjectionFunctor()
      : inner_(std::make_unique<
               AutoDiffCostFunction<InnerProjectionFunctor, 2, 3, 3>>(
            std::make_unique<InnerProjectionFunctor>())) {}

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    T p[3];
    AngleAxisRotatePoint(rotation, point, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];
    if constexpr (kDiffType == kAutoDiff) {
      return inner_(intrinsics, p, residuals);
    } else {
      const T* params[2] = {intrinsics, p};
      return inner_(params, residuals);
    }
  }

  std::conditional_t<kDiffType == kAutoDiff,
                     CostFunctionToFunctor<2, 3, 3>,
                     DynamicCostFunctionToFunctor>
      inner_;
};

template <DiffType kDiffType>
void BM_CostFunctionToFunctor(benchmark::State& state) {
  std::unique_ptr<CostFunction> cost_function = std::make_unique<
      AutoDiffCostFunction<OuterProjectionFunctor<kDiffType>, 2, 3, 3, 3, 3>>(
      std::make_unique<OuterProjectionFunctor<kDiffType>>());
  double rot[3] = {0.1, -0.2, 0.05};
  double trans[3] = {0.5, -0.1, 2.0};
  double intr[3] = {500.0, -0.01, 0.001};
  double pt[3] = {0.3, -0.4, 5.0};
  const double* params[4] = {rot, trans, intr, pt};
  double residuals[2];
  double jacobian[4 * 2 * 3];
  double* jacobians[4] = {
      jacobian + 0, jacobian + 6, jacobian + 12, jacobian + 18};

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(params, residuals, jacobians));
  }
}

BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kDynamicAutoDiff);

}  // namespace
}  // namespace ceres

BENCHMARK_MAIN();
