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
#include <cmath>
#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_manifold.h"
#include "ceres/manifold.h"
#include "ceres/rotation.h"

namespace ceres::internal {
namespace {

struct QuaternionFunctor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    T q_delta[4];
    if (squared_norm_delta > T(0.0)) {
      const T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    const T minus_x[4] = {x[0], -x[1], -x[2], -x[3]};
    T ambient_y_minus_x[4];
    QuaternionProduct(y, minus_x, ambient_y_minus_x);
    const T u_sq = ambient_y_minus_x[1] * ambient_y_minus_x[1] +
                   ambient_y_minus_x[2] * ambient_y_minus_x[2] +
                   ambient_y_minus_x[3] * ambient_y_minus_x[3];
    if (u_sq > T(0.0)) {
      const T u_norm = sqrt(u_sq);
      const T theta = atan2(u_norm, ambient_y_minus_x[0]);
      y_minus_x[0] = theta * ambient_y_minus_x[1] / u_norm;
      y_minus_x[1] = theta * ambient_y_minus_x[2] / u_norm;
      y_minus_x[2] = theta * ambient_y_minus_x[3] / u_norm;
    } else {
      y_minus_x[0] = ambient_y_minus_x[1];
      y_minus_x[1] = ambient_y_minus_x[2];
      y_minus_x[2] = ambient_y_minus_x[3];
    }
    return true;
  }
};

struct Pose3Functor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    QuaternionFunctor q;
    q.Plus(x, delta, x_plus_delta);
    x_plus_delta[4] = x[4] + delta[3];
    x_plus_delta[5] = x[5] + delta[4];
    x_plus_delta[6] = x[6] + delta[5];
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    QuaternionFunctor q;
    q.Minus(y, x, y_minus_x);
    y_minus_x[3] = y[4] - x[4];
    y_minus_x[4] = y[5] - x[5];
    y_minus_x[5] = y[6] - x[6];
    return true;
  }
};

template <typename Functor, int kAmbientSize, int kTangentSize>
void BM_AutoDiffManifoldPlusJacobian(benchmark::State& state) {
  std::unique_ptr<Manifold> manifold =
      std::make_unique<AutoDiffManifold<Functor, kAmbientSize, kTangentSize>>();
  std::array<double, kAmbientSize> x{};
  x[0] = 0.5;
  x[1] = 0.5;
  x[2] = 0.5;
  x[3] = 0.5;
  for (int i = 4; i < kAmbientSize; ++i) {
    x[i] = static_cast<double>(i - 3);
  }
  std::array<double, kAmbientSize * kTangentSize> jacobian{};
  for (auto _ : state) {
    benchmark::DoNotOptimize(manifold->PlusJacobian(x.data(), jacobian.data()));
  }
}

template <typename Functor, int kAmbientSize, int kTangentSize>
void BM_AutoDiffManifoldMinusJacobian(benchmark::State& state) {
  std::unique_ptr<Manifold> manifold =
      std::make_unique<AutoDiffManifold<Functor, kAmbientSize, kTangentSize>>();
  std::array<double, kAmbientSize> x{};
  x[0] = 0.5;
  x[1] = 0.5;
  x[2] = 0.5;
  x[3] = 0.5;
  for (int i = 4; i < kAmbientSize; ++i) {
    x[i] = static_cast<double>(i - 3);
  }
  std::array<double, kTangentSize * kAmbientSize> jacobian{};
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        manifold->MinusJacobian(x.data(), jacobian.data()));
  }
}

BENCHMARK_TEMPLATE(BM_AutoDiffManifoldPlusJacobian, QuaternionFunctor, 4, 3);
BENCHMARK_TEMPLATE(BM_AutoDiffManifoldMinusJacobian, QuaternionFunctor, 4, 3);
BENCHMARK_TEMPLATE(BM_AutoDiffManifoldPlusJacobian, Pose3Functor, 7, 6);
BENCHMARK_TEMPLATE(BM_AutoDiffManifoldMinusJacobian, Pose3Functor, 7, 6);

}  // namespace
}  // namespace ceres::internal

BENCHMARK_MAIN();
