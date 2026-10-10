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

#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_TO_FUNCTOR_BENCHMARK_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_TO_FUNCTOR_BENCHMARK_H_

#include <memory>
#include <type_traits>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/cost_function_benchmark_utils.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace ceres {

// Benchmarks for CostFunctionToFunctor (static sizes) and
// DynamicCostFunctionToFunctor (dynamic sizes).
//
// CostFunctionToFunctor and DynamicCostFunctionToFunctor allow an existing
// CostFunction object (which only accepts `double` inputs and computes `double`
// residuals and Jacobians via CostFunction::Evaluate) to be used as a templated
// functor inside another cost functor. When called with `Jet<double, N>`
// arguments during automatic differentiation of the outer functor, the adapter:
//   1. Extracts the scalar (`double`) parts of the input Jets.
//   2. Calls the inner CostFunction::Evaluate to compute the inner residuals
//      and Jacobians with respect to the inner inputs.
//   3. Applies the chain rule to multiply the inner Jacobians by the derivative
//      (`.v`) parts of the input Jets to form the output residual Jets.
//
// To benchmark this composition realistically, we split a standard pinhole
// camera projection with radial distortion into two stages:
//
//   - Stage 1 (`InnerProjectionFunctor`): Projects a 3D point already in the
//     camera coordinate frame, `p_cam = [X, Y, Z]`, into 2D pixel coordinates
//     using camera `intrinsics = [focal, k1, k2]`:
//       xp = X / Z,  yp = Y / Z,  r^2 = xp^2 + yp^2
//       distortion = 1 + k1 * r^2 + k2 * r^4
//       residuals  = [focal * distortion * xp, focal * distortion * yp]
//     This inner functor is wrapped in an `AutoDiffCostFunction<..., 2, 3, 3>`,
//     representing a pre-existing CostFunction object.
//
//   - Stage 2 (`OuterProjectionFunctor`): Takes four 3D parameter blocks
//     (`rotation` angle-axis, `translation`, `intrinsics`, and a world-frame
//     `point`), transforms the world point into the camera frame:
//       p_cam = AngleAxisRotatePoint(rotation, point) + translation
//     and then invokes the inner CostFunction via either:
//       * `CostFunctionToFunctor<2, 3, 3>` (when `kDiffType == kAutoDiff`), or
//       * `DynamicCostFunctionToFunctor`   (when `kDiffType == kDynamicAutoDiff`).
//
//   - Benchmark (`BM_CostFunctionToFunctor`): Wraps `OuterProjectionFunctor`
//     in an outer `AutoDiffCostFunction<..., 2, 3, 3, 3, 3>` and measures its
//     evaluation time.

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
    // Transform the 3D point from the world frame into the camera frame.
    T p_cam[3];
    AngleAxisRotatePoint(rotation, point, p_cam);
    p_cam[0] += translation[0];
    p_cam[1] += translation[1];
    p_cam[2] += translation[2];

    // Project the camera-frame point into 2D pixel coordinates by invoking the
    // wrapped inner CostFunction through CostFunctionToFunctor (static) or
    // DynamicCostFunctionToFunctor (dynamic).
    if constexpr (kDiffType == kAutoDiff) {
      return inner_(intrinsics, p_cam, residuals);
    } else {
      const T* params[2] = {intrinsics, p_cam};
      return inner_(params, residuals);
    }
  }

  std::conditional_t<kDiffType == kAutoDiff,
                     CostFunctionToFunctor<2, 3, 3>,
                     DynamicCostFunctionToFunctor>
      inner_;
};

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_CostFunctionToFunctor(benchmark::State& state) {
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
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(params, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor,
                   kDynamicAutoDiff,
                   kResidualsAndJacobians);

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_TO_FUNCTOR_BENCHMARK_H_
