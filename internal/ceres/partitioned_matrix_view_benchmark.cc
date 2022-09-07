// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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

#include <memory>
#include <random>

#include "benchmark/benchmark.h"
#include "ceres/context_impl.h"
#include "ceres/fake_bundle_adjustment_jacobian.h"
#include "ceres/partitioned_matrix_view.h"

constexpr int kNumCameras = 1000;
constexpr int kNumPoints = 10000;
constexpr int kCameraSize = 6;
constexpr int kPointSize = 3;
constexpr double kVisibility = 0.1;

namespace ceres::internal {

static void BM_PatitionedViewRightMultiplyAndAccumulateE_Static(
    benchmark::State& state) {
  const int num_threads = state.range(0);
  std::mt19937 prng;
  auto [partitioned_view, jacobian] =
      CreateFakeBundleAdjustmentPartitionedJacobian<kPointSize, kCameraSize>(
          kNumCameras, kNumPoints, kVisibility, prng);

  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Vector x(partitioned_view->num_cols_e());
  Vector y(partitioned_view->num_rows());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    partitioned_view->RightMultiplyAndAccumulateE(
        x.data(), y.data(), &context, num_threads);
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}
BENCHMARK(BM_PatitionedViewRightMultiplyAndAccumulateE_Static)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

static void BM_PatitionedViewRightMultiplyAndAccumulateE_Dynamic(
    benchmark::State& state) {
  std::mt19937 prng;
  auto [partitioned_view, jacobian] =
      CreateFakeBundleAdjustmentPartitionedJacobian(
          kNumCameras, kNumPoints, kCameraSize, kPointSize, kVisibility, prng);

  Vector x(partitioned_view->num_cols_e());
  Vector y(partitioned_view->num_rows());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    partitioned_view->RightMultiplyAndAccumulateE(x.data(), y.data());
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}
BENCHMARK(BM_PatitionedViewRightMultiplyAndAccumulateE_Dynamic);

static void BM_PatitionedViewRightMultiplyAndAccumulateF_Static(
    benchmark::State& state) {
  const int num_threads = state.range(0);
  std::mt19937 prng;
  auto [partitioned_view, jacobian] =
      CreateFakeBundleAdjustmentPartitionedJacobian<kPointSize, kCameraSize>(
          kNumCameras, kNumPoints, kVisibility, prng);

  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Vector x(partitioned_view->num_cols_f());
  Vector y(partitioned_view->num_rows());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    partitioned_view->RightMultiplyAndAccumulateF(
        x.data(), y.data(), &context, num_threads);
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}
BENCHMARK(BM_PatitionedViewRightMultiplyAndAccumulateF_Static)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

static void BM_PatitionedViewRightMultiplyAndAccumulateF_Dynamic(
    benchmark::State& state) {
  std::mt19937 prng;
  auto [partitioned_view, jacobian] =
      CreateFakeBundleAdjustmentPartitionedJacobian(
          kNumCameras, kNumPoints, kCameraSize, kPointSize, kVisibility, prng);

  Vector x(partitioned_view->num_cols_f());
  Vector y(partitioned_view->num_rows());
  x.setRandom();
  y.setRandom();
  double sum = 0;
  for (auto _ : state) {
    partitioned_view->RightMultiplyAndAccumulateF(x.data(), y.data());
    sum += y.norm();
  }
  CHECK_NE(sum, 0.0);
}
BENCHMARK(BM_PatitionedViewRightMultiplyAndAccumulateF_Dynamic);

}  // namespace ceres::internal

BENCHMARK_MAIN();
