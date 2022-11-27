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
#include "benchmark/benchmark.h"
#include "ceres/parallel_for.h"

namespace ceres::internal {

const int kVectorSize = 64 * 1024 * 1024 / sizeof(double);
const double kEpsilon = 1e-13;

static void SetZero(benchmark::State& state) {
  Vector x = Vector::Random(kVectorSize);
  for (auto _ : state) {
    x.setZero();
  }
  CHECK_EQ(x.squaredNorm(), 0.);
}
BENCHMARK(SetZero);

static void SetZeroParallel(benchmark::State& state) {
  const int num_threads = state.range(0);
  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Vector x = Vector::Random(kVectorSize);
  for (auto _ : state) {
    ParallelSetZero(&context, num_threads, x);
  }
  CHECK_EQ(x.squaredNorm(), 0.);
}
BENCHMARK(SetZeroParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

static void Negate(benchmark::State& state) {
  Vector x = Vector::Random(kVectorSize).normalized();
  const Vector x_init = x;

  for (auto _ : state) {
    x = -x;
  }
  CHECK((x - x_init).squaredNorm() == 0. || (x + x_init).squaredNorm() == 0);
}
BENCHMARK(Negate);

static void NegateParallel(benchmark::State& state) {
  const int num_threads = state.range(0);
  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Vector x = Vector::Random(kVectorSize).normalized();
  const Vector x_init = x;

  for (auto _ : state) {
    ParallelEvaluate(&context, num_threads, x, -x);
  }
  CHECK((x - x_init).squaredNorm() == 0. || (x + x_init).squaredNorm() == 0);
}
BENCHMARK(NegateParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

static void Assign(benchmark::State& state) {
  Vector x = Vector::Random(kVectorSize);
  Vector y = Vector(kVectorSize);
  for (auto _ : state) {
    y = x;
  }
  CHECK_EQ((y - x).squaredNorm(), 0.);
}
BENCHMARK(Assign);

static void AssignParallel(benchmark::State& state) {
  const int num_threads = state.range(0);
  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Vector x = Vector::Random(kVectorSize);
  Vector y = Vector(kVectorSize);

  for (auto _ : state) {
    ParallelEvaluate(&context, num_threads, y, x);
  }
  CHECK_EQ((y - x).squaredNorm(), 0.);
}
BENCHMARK(AssignParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

static void D2X(benchmark::State& state) {
  const Vector x = Vector::Random(kVectorSize);
  const Vector D = Vector::Random(kVectorSize);
  Vector y = Vector::Zero(kVectorSize);
  for (auto _ : state) {
    y = D.array().square() * x.array();
  }
  CHECK_GT(y.squaredNorm(), 0.);
}
BENCHMARK(D2X);

static void D2XParallel(benchmark::State& state) {
  const int num_threads = state.range(0);
  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  const Vector x = Vector::Random(kVectorSize);
  const Vector D = Vector::Random(kVectorSize);
  Vector y = Vector(kVectorSize);

  for (auto _ : state) {
    ParallelEvaluate(&context, num_threads, y, D.array().square() * x.array());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}
BENCHMARK(D2XParallel)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

}  // namespace ceres::internal

BENCHMARK_MAIN();
