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

#include "benchmark/benchmark.h"
#include "ceres/jet.h"

namespace ceres {

// Cycle the Jets to avoid caching effects in the benchmark.
template <class JetType>
class JetInputData {
 public:
  JetInputData()
      : size_{20},
        index_{0},
        a_(size_),
        b_(size_),
        c_(size_),
        d_(size_),
        e_(size_) {
    using T = typename JetType::Scalar;
    for (std::size_t i = 0; i < size_; i++) {
      const T ti = static_cast<T>(i + 1);
      a_[i].a = T(1.1) * ti;
      a_[i].v.fill(T(1.1) * ti);

      b_[i].a = T(2.2) * ti;
      b_[i].v.fill(T(2.2) * ti);

      c_[i].a = T(3.3) * ti;
      c_[i].v.fill(T(3.3) * ti);

      d_[i].a = T(4.4) * ti;
      d_[i].v.fill(T(4.4) * ti);

      e_[i].a = T(5.5) * ti;
      e_[i].v.fill(T(5.5) * ti);
    }
  }

  void advance() { index_ = (index_ + 1) % size_; }
  const JetType& a() { return a_[index_]; }
  const JetType& b() { return a_[index_]; }
  const JetType& c() { return a_[index_]; }
  const JetType& d() { return a_[index_]; }
  const JetType& e() { return a_[index_]; }

 private:
  std::size_t size_;
  std::size_t index_;
  std::vector<JetType> a_;
  std::vector<JetType> b_;
  std::vector<JetType> c_;
  std::vector<JetType> d_;
  std::vector<JetType> e_;
};

class JetFixture : public benchmark::Fixture {
 public:
  ~JetFixture() override;
};
JetFixture::~JetFixture() = default;

template <std::size_t JET_SIZE, std::size_t ITERATIONS>
void addition_helper(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetInputData<JetType> data{};
  JetType out{};
  for (auto _ : state) {
    for (std::size_t i = 0; i < ITERATIONS; i++) {
      out += data.a() + data.b() + data.c() + data.d() + data.e();
      data.advance();
      benchmark::DoNotOptimize(out);
    }
  }
}

BENCHMARK_F(JetFixture, Addition10x1000)(benchmark::State& state) {
  addition_helper<10, 1000>(state);
}
BENCHMARK_F(JetFixture, Addition25x1000)(benchmark::State& state) {
  addition_helper<25, 1000>(state);
}
BENCHMARK_F(JetFixture, Addition32x1000)(benchmark::State& state) {
  addition_helper<32, 1000>(state);
}
BENCHMARK_F(JetFixture, Addition1000x32)(benchmark::State& state) {
  addition_helper<1000, 32>(state);
}

}  // namespace ceres

BENCHMARK_MAIN();
