// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2021 Google Inc. All rights reserved.
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
// Author: alex@karatarakis.com (Alexander Karatarakis)

#include <array>

#include "benchmark/benchmark.h"
#include "ceres/jet.h"

namespace ceres {

// Cycle the Jets to avoid caching effects in the benchmark.
template <class JetType>
class JetInputData {
  static constexpr std::size_t SIZE = 20;

 public:
  JetInputData() : index_{0}, a_{}, b_{}, c_{}, d_{}, e_{} {
    using T = typename JetType::Scalar;
    for (int i = 0; i < static_cast<int>(SIZE); i++) {
      const T ti = static_cast<T>(i + 1);

      a_[i].a = T(1.1) * ti;
      a_[i].v.setRandom();

      b_[i].a = T(2.2) * ti;
      b_[i].v.setRandom();

      c_[i].a = T(3.3) * ti;
      c_[i].v.setRandom();

      d_[i].a = T(4.4) * ti;
      d_[i].v.setRandom();

      e_[i].a = T(5.5) * ti;
      e_[i].v.setRandom();
    }
  }

  void advance() { index_ = (index_ + 1) % SIZE; }

  const JetType& a() { return a_[index_]; }
  const JetType& b() { return b_[index_]; }
  const JetType& c() { return c_[index_]; }
  const JetType& d() { return d_[index_]; }
  const JetType& e() { return e_[index_]; }

 private:
  std::size_t index_;
  std::array<JetType, SIZE> a_;
  std::array<JetType, SIZE> b_;
  std::array<JetType, SIZE> c_;
  std::array<JetType, SIZE> d_;
  std::array<JetType, SIZE> e_;
};

template <std::size_t JET_SIZE>
static void addition(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetInputData<JetType> data{};
  JetType out{};
  const int iterations = static_cast<int>(state.range(0));
  for (auto _ : state) {
    for (int i = 0; i < iterations; i++) {
      out += data.a() + data.b() + data.c() + data.d() + data.e();
      data.advance();
    }
  }
  benchmark::DoNotOptimize(out);
}

BENCHMARK_TEMPLATE(addition, 3)->Arg(1000);
BENCHMARK_TEMPLATE(addition, 10)->Arg(1000);
BENCHMARK_TEMPLATE(addition, 15)->Arg(1000);
BENCHMARK_TEMPLATE(addition, 25)->Arg(1000);
BENCHMARK_TEMPLATE(addition, 32)->Arg(1000);
BENCHMARK_TEMPLATE(addition, 200)->Arg(160);

template <std::size_t JET_SIZE>
static void multiply_add_operation(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetInputData<JetType> data{};
  JetType out{};
  const int iterations = static_cast<int>(state.range(0));
  for (auto _ : state) {
    for (int i = 0; i < iterations; i++) {
      out += data.a().a * data.a() + data.b().a * data.b() +
             data.c().a * data.c() + data.d().a * data.d() +
             data.e().a * data.e();
      data.advance();
    }
  }
  benchmark::DoNotOptimize(out);
}

BENCHMARK_TEMPLATE(multiply_add_operation, 3)->Arg(1000);
BENCHMARK_TEMPLATE(multiply_add_operation, 10)->Arg(1000);
BENCHMARK_TEMPLATE(multiply_add_operation, 15)->Arg(1000);
BENCHMARK_TEMPLATE(multiply_add_operation, 25)->Arg(1000);
BENCHMARK_TEMPLATE(multiply_add_operation, 32)->Arg(1000);
BENCHMARK_TEMPLATE(multiply_add_operation, 200)->Arg(160);

template <std::size_t JET_SIZE>
static void subtraction_division_operation(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetInputData<JetType> data{};
  JetType out{};
  const int iterations = static_cast<int>(state.range(0));
  for (auto _ : state) {
    for (int i = 0; i < iterations; i++) {
      out -= data.a() / data.a().a - data.b() / data.b().a -
             data.c() / data.c().a - data.d() / data.d().a -
             data.e() / data.e().a;
      data.advance();
    }
  }
  benchmark::DoNotOptimize(out);
}

BENCHMARK_TEMPLATE(subtraction_division_operation, 3)->Arg(1000);
BENCHMARK_TEMPLATE(subtraction_division_operation, 10)->Arg(1000);
BENCHMARK_TEMPLATE(subtraction_division_operation, 15)->Arg(1000);
BENCHMARK_TEMPLATE(subtraction_division_operation, 25)->Arg(1000);
BENCHMARK_TEMPLATE(subtraction_division_operation, 32)->Arg(1000);
BENCHMARK_TEMPLATE(subtraction_division_operation, 200)->Arg(160);

}  // namespace ceres

BENCHMARK_MAIN();
