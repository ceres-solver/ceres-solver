// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <array>

#include "benchmark/benchmark.h"
#include "ceres/jet.h"

namespace ceres {

// Cycle the Jets to avoid caching effects in the benchmark.
template <class JetType>
class JetInputData {
  using T = typename JetType::Scalar;
  static constexpr std::size_t SIZE = 20;

 public:
  JetInputData() {
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

      // Values strictly inside (-1, 1) for inverse trigonometric functions.
      unit_a_[i].a =
          T(-0.9) + T(1.8) * static_cast<T>(i) / static_cast<T>(SIZE);
      unit_a_[i].v.setRandom();

      unit_b_[i].a =
          T(-0.85) + T(1.7) * static_cast<T>(i) / static_cast<T>(SIZE);
      unit_b_[i].v.setRandom();

      scalar_a_[i] = T(1.1) * ti;
      scalar_b_[i] = T(2.2) * ti;
      scalar_c_[i] = T(3.3) * ti;
      scalar_d_[i] = T(4.4) * ti;
      scalar_e_[i] = T(5.5) * ti;
    }
  }

  void advance() { index_ = (index_ + 1) % SIZE; }

  const JetType& a() const { return a_[index_]; }
  const JetType& b() const { return b_[index_]; }
  const JetType& c() const { return c_[index_]; }
  const JetType& d() const { return d_[index_]; }
  const JetType& e() const { return e_[index_]; }
  const JetType& unit_a() const { return unit_a_[index_]; }
  const JetType& unit_b() const { return unit_b_[index_]; }
  T scalar_a() const { return scalar_a_[index_]; }
  T scalar_b() const { return scalar_b_[index_]; }
  T scalar_c() const { return scalar_c_[index_]; }
  T scalar_d() const { return scalar_d_[index_]; }
  T scalar_e() const { return scalar_e_[index_]; }

 private:
  std::size_t index_{0};
  std::array<JetType, SIZE> a_{};
  std::array<JetType, SIZE> b_{};
  std::array<JetType, SIZE> c_{};
  std::array<JetType, SIZE> d_{};
  std::array<JetType, SIZE> e_{};
  std::array<JetType, SIZE> unit_a_{};
  std::array<JetType, SIZE> unit_b_{};
  std::array<T, SIZE> scalar_a_;
  std::array<T, SIZE> scalar_b_;
  std::array<T, SIZE> scalar_c_;
  std::array<T, SIZE> scalar_d_;
  std::array<T, SIZE> scalar_e_;
};

template <std::size_t JET_SIZE, class Function>
static void JetBenchmarkHelper(benchmark::State& state, const Function& func) {
  using JetType = Jet<double, JET_SIZE>;
  JetInputData<JetType> data{};
  JetType out{};
  const int iterations = static_cast<int>(state.range(0));
  for (auto _ : state) {
    for (int i = 0; i < iterations; i++) {
      func(data, out);
      data.advance();
    }
  }
  benchmark::DoNotOptimize(out);
}

#define CERES_REGISTER_JET_BENCHMARK(Name) \
  BENCHMARK_TEMPLATE(Name, 3)->Arg(1000);  \
  BENCHMARK_TEMPLATE(Name, 10)->Arg(1000); \
  BENCHMARK_TEMPLATE(Name, 15)->Arg(1000); \
  BENCHMARK_TEMPLATE(Name, 25)->Arg(1000); \
  BENCHMARK_TEMPLATE(Name, 32)->Arg(1000)

// ---------------------------------------------------------------------------
// Arithmetic & compound assignment operators
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Addition(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += +d.a() + d.b() + d.c() + d.d() + d.e();
      });
}
CERES_REGISTER_JET_BENCHMARK(Addition);

template <std::size_t JET_SIZE>
static void AdditionScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out +=
            d.scalar_a() + d.scalar_b() + d.c() + d.scalar_d() + d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(AdditionScalar);

template <std::size_t JET_SIZE>
static void Subtraction(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out -= -d.a() - d.b() - d.c() - d.d() - d.e();
      });
}
CERES_REGISTER_JET_BENCHMARK(Subtraction);

template <std::size_t JET_SIZE>
static void SubtractionScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out -=
            -d.scalar_a() - d.scalar_b() - d.c() - d.scalar_d() - d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(SubtractionScalar);

template <std::size_t JET_SIZE>
static void Multiplication(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.a() * d.b() * d.c() * d.d() * d.e();
      });
}
CERES_REGISTER_JET_BENCHMARK(Multiplication);

template <std::size_t JET_SIZE>
static void MultiplicationLeftScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.scalar_a() *
               (d.scalar_b() * (d.scalar_c() * (d.scalar_d() * d.e())));
      });
}
CERES_REGISTER_JET_BENCHMARK(MultiplicationLeftScalar);

template <std::size_t JET_SIZE>
static void MultiplicationRightScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += (((d.a() * d.scalar_b()) * d.scalar_c()) * d.scalar_d()) *
               d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(MultiplicationRightScalar);

template <std::size_t JET_SIZE>
static void Division(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.a() / d.b() / d.c() / d.d() / d.e();
      });
}
CERES_REGISTER_JET_BENCHMARK(Division);

template <std::size_t JET_SIZE>
static void CompoundAdditionScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.scalar_a();
        out += d.scalar_b();
        out += d.scalar_c();
        out += d.scalar_d();
        out += d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundAdditionScalar);

template <std::size_t JET_SIZE>
static void CompoundSubtractionScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out -= d.scalar_a();
        out -= d.scalar_b();
        out -= d.scalar_c();
        out -= d.scalar_d();
        out -= d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundSubtractionScalar);

template <std::size_t JET_SIZE>
static void CompoundMultiplication(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        JetType tmp = d.a();
        tmp *= d.b();
        tmp *= d.c();
        tmp *= d.d();
        tmp *= d.e();
        out += tmp;
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundMultiplication);

template <std::size_t JET_SIZE>
static void CompoundMultiplicationScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        JetType tmp = d.a();
        tmp *= d.scalar_b();
        tmp *= d.scalar_c();
        tmp *= d.scalar_d();
        tmp *= d.scalar_e();
        out += tmp;
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundMultiplicationScalar);

template <std::size_t JET_SIZE>
static void CompoundDivision(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        JetType tmp = d.a();
        tmp /= d.b();
        tmp /= d.c();
        tmp /= d.d();
        tmp /= d.e();
        out += tmp;
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundDivision);

template <std::size_t JET_SIZE>
static void CompoundDivisionScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        JetType tmp = d.a();
        tmp /= d.scalar_b();
        tmp /= d.scalar_c();
        tmp /= d.scalar_d();
        tmp /= d.scalar_e();
        out += tmp;
      });
}
CERES_REGISTER_JET_BENCHMARK(CompoundDivisionScalar);

template <std::size_t JET_SIZE>
static void DivisionLeftScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.scalar_a() /
               (d.scalar_b() / (d.scalar_c() / (d.scalar_d() / d.e())));
      });
}
CERES_REGISTER_JET_BENCHMARK(DivisionLeftScalar);

template <std::size_t JET_SIZE>
static void DivisionRightScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += (((d.a() / d.scalar_b()) / d.scalar_c()) / d.scalar_d()) /
               d.scalar_e();
      });
}
CERES_REGISTER_JET_BENCHMARK(DivisionRightScalar);

template <std::size_t JET_SIZE>
static void MultiplyAndAdd(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += d.scalar_a() * d.a() + d.scalar_b() * d.b() +
               d.scalar_c() * d.c() + d.scalar_d() * d.d() +
               d.scalar_e() * d.e();
      });
}
CERES_REGISTER_JET_BENCHMARK(MultiplyAndAdd);

// ---------------------------------------------------------------------------
// Unary algebraic & rounding functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Abs(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += abs(d.unit_a()) + abs(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Abs);

template <std::size_t JET_SIZE>
static void Sqrt(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += sqrt(d.a()) + sqrt(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Sqrt);

template <std::size_t JET_SIZE>
static void Cbrt(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += cbrt(d.a()) + cbrt(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Cbrt);

template <std::size_t JET_SIZE>
static void Norm(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += norm(d.a()) + norm(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Norm);

template <std::size_t JET_SIZE>
static void Floor(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += floor(d.a()) + floor(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Floor);

template <std::size_t JET_SIZE>
static void Ceil(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += ceil(d.a()) + ceil(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Ceil);

// ---------------------------------------------------------------------------
// Exponential & logarithmic functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Exp(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += exp(d.a()) + exp(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Exp);

template <std::size_t JET_SIZE>
static void Exp2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += exp2(d.a()) + exp2(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Exp2);

template <std::size_t JET_SIZE>
static void Expm1(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += expm1(d.unit_a()) + expm1(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Expm1);

template <std::size_t JET_SIZE>
static void Log(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += log(d.a()) + log(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Log);

template <std::size_t JET_SIZE>
static void Log2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += log2(d.a()) + log2(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Log2);

template <std::size_t JET_SIZE>
static void Log10(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += log10(d.a()) + log10(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Log10);

template <std::size_t JET_SIZE>
static void Log1p(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += log1p(d.unit_a()) + log1p(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Log1p);

// ---------------------------------------------------------------------------
// Trigonometric & inverse trigonometric functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Sin(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += sin(d.a()) + sin(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Sin);

template <std::size_t JET_SIZE>
static void Cos(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += cos(d.a()) + cos(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Cos);

template <std::size_t JET_SIZE>
static void Tan(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += tan(d.a()) + tan(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Tan);

template <std::size_t JET_SIZE>
static void Asin(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += asin(d.unit_a()) + asin(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Asin);

template <std::size_t JET_SIZE>
static void Acos(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += acos(d.unit_a()) + acos(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Acos);

template <std::size_t JET_SIZE>
static void Atan(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += atan(d.a()) + atan(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Atan);

template <std::size_t JET_SIZE>
static void Atan2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += atan2(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Atan2);

// ---------------------------------------------------------------------------
// Hyperbolic functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Sinh(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += sinh(d.a()) + sinh(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Sinh);

template <std::size_t JET_SIZE>
static void Cosh(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += cosh(d.a()) + cosh(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Cosh);

template <std::size_t JET_SIZE>
static void Tanh(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += tanh(d.a()) + tanh(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Tanh);

// ---------------------------------------------------------------------------
// Power, norm & hypotenuse functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void PowScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += pow(d.a(), 2.3) + pow(d.b(), -0.7);
      });
}
CERES_REGISTER_JET_BENCHMARK(PowScalar);

template <std::size_t JET_SIZE>
static void PowScalarBase(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += pow(d.scalar_a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(PowScalarBase);

template <std::size_t JET_SIZE>
static void PowJet(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += pow(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(PowJet);

template <std::size_t JET_SIZE>
static void Hypot2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += hypot(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Hypot2);

template <std::size_t JET_SIZE>
static void Hypot3(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += hypot(d.a(), d.b(), d.c());
      });
}
CERES_REGISTER_JET_BENCHMARK(Hypot3);

template <std::size_t JET_SIZE>
static void AccurateNorm2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += AccurateNorm(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(AccurateNorm2);

template <std::size_t JET_SIZE>
static void AccurateNorm3(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += AccurateNorm(d.a(), d.b(), d.c());
      });
}
CERES_REGISTER_JET_BENCHMARK(AccurateNorm3);

template <std::size_t JET_SIZE>
static void AccurateRNorm2(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += AccurateRNorm(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(AccurateRNorm2);

template <std::size_t JET_SIZE>
static void AccurateRNorm3(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += AccurateRNorm(d.a(), d.b(), d.c());
      });
}
CERES_REGISTER_JET_BENCHMARK(AccurateRNorm3);

// ---------------------------------------------------------------------------
// Special functions (error functions & Bessel functions)
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Erf(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += erf(d.unit_a()) + erf(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Erf);

template <std::size_t JET_SIZE>
static void Erfc(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += erfc(d.unit_a()) + erfc(d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Erfc);

#if defined(CERES_HAS_CPP17_BESSEL_FUNCTIONS) || \
    defined(CERES_HAS_POSIX_BESSEL_FUNCTIONS)

template <std::size_t JET_SIZE>
static void BesselJ0Bench(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += BesselJ0(d.a()) + BesselJ0(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(BesselJ0Bench);

template <std::size_t JET_SIZE>
static void BesselJ1Bench(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += BesselJ1(d.a()) + BesselJ1(d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(BesselJ1Bench);

template <std::size_t JET_SIZE>
static void BesselJnBench(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += BesselJn(2, d.a()) + BesselJn(3, d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(BesselJnBench);

#ifdef CERES_HAS_CPP17_BESSEL_FUNCTIONS

template <std::size_t JET_SIZE>
static void CylBesselJBench(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += cyl_bessel_j(0.0, d.a()) + cyl_bessel_j(2.5, d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(CylBesselJBench);

#endif  // defined(CERES_HAS_CPP17_BESSEL_FUNCTIONS)

#endif  // defined(CERES_HAS_CPP17_BESSEL_FUNCTIONS) ||
        // defined(CERES_HAS_POSIX_BESSEL_FUNCTIONS)

// ---------------------------------------------------------------------------
// Multi-argument, branching, & interpolation functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void Copysign(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += copysign(d.a(), d.unit_a()) + copysign(d.b(), d.unit_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Copysign);

template <std::size_t JET_SIZE>
static void Fma(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fma(d.a(), d.b(), d.c());
      });
}
CERES_REGISTER_JET_BENCHMARK(Fma);

template <std::size_t JET_SIZE>
static void Fmax(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fmax(d.a(), d.b()) + fmax(d.b(), d.a());
      });
}
CERES_REGISTER_JET_BENCHMARK(Fmax);

template <std::size_t JET_SIZE>
static void FmaxScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fmax(d.a(), d.scalar_b()) + fmax(d.b(), d.scalar_a());
      });
}
CERES_REGISTER_JET_BENCHMARK(FmaxScalar);

template <std::size_t JET_SIZE>
static void Fmin(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fmin(d.a(), d.b()) + fmin(d.b(), d.a());
      });
}
CERES_REGISTER_JET_BENCHMARK(Fmin);

template <std::size_t JET_SIZE>
static void FminScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fmin(d.a(), d.scalar_b()) + fmin(d.b(), d.scalar_a());
      });
}
CERES_REGISTER_JET_BENCHMARK(FminScalar);

template <std::size_t JET_SIZE>
static void Fdim(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fdim(d.b(), d.a()) + fdim(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Fdim);

template <std::size_t JET_SIZE>
static void FdimScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += fdim(d.b(), d.scalar_a()) + fdim(d.a(), d.scalar_b());
      });
}
CERES_REGISTER_JET_BENCHMARK(FdimScalar);

#ifdef CERES_HAS_CPP20

template <std::size_t JET_SIZE>
static void Lerp(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += lerp(d.a(), d.b(), d.unit_a());
      });
}
CERES_REGISTER_JET_BENCHMARK(Lerp);

template <std::size_t JET_SIZE>
static void Midpoint(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        out += midpoint(d.a(), d.b());
      });
}
CERES_REGISTER_JET_BENCHMARK(Midpoint);

#endif  // defined(CERES_HAS_CPP20)

// ---------------------------------------------------------------------------
// Comparison & classification functions
// ---------------------------------------------------------------------------

template <std::size_t JET_SIZE>
static void ComparisonJet(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        const int r = static_cast<int>(d.a() < d.b()) +
                      static_cast<int>(d.a() <= d.b()) +
                      static_cast<int>(d.b() > d.a()) +
                      static_cast<int>(d.b() >= d.a()) +
                      static_cast<int>(d.a() == d.b()) +
                      static_cast<int>(d.a() != d.b());
        out.a += static_cast<double>(r);
      });
}
CERES_REGISTER_JET_BENCHMARK(ComparisonJet);

template <std::size_t JET_SIZE>
static void ComparisonScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        const int r = static_cast<int>(d.a() < d.scalar_b()) +
                      static_cast<int>(d.scalar_a() <= d.b()) +
                      static_cast<int>(d.b() > d.scalar_a()) +
                      static_cast<int>(d.scalar_b() >= d.a()) +
                      static_cast<int>(d.a() == d.scalar_b()) +
                      static_cast<int>(d.scalar_a() != d.b());
        out.a += static_cast<double>(r);
      });
}
CERES_REGISTER_JET_BENCHMARK(ComparisonScalar);

template <std::size_t JET_SIZE>
static void QuietComparisonJet(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        const int r = static_cast<int>(isless(d.a(), d.b())) +
                      static_cast<int>(islessequal(d.a(), d.b())) +
                      static_cast<int>(isgreater(d.b(), d.a())) +
                      static_cast<int>(isgreaterequal(d.b(), d.a())) +
                      static_cast<int>(islessgreater(d.a(), d.b())) +
                      static_cast<int>(isunordered(d.a(), d.b()));
        out.a += static_cast<double>(r);
      });
}
CERES_REGISTER_JET_BENCHMARK(QuietComparisonJet);

template <std::size_t JET_SIZE>
static void QuietComparisonScalar(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        const int r = static_cast<int>(isless(d.a(), d.scalar_b())) +
                      static_cast<int>(islessequal(d.scalar_a(), d.b())) +
                      static_cast<int>(isgreater(d.b(), d.scalar_a())) +
                      static_cast<int>(isgreaterequal(d.scalar_b(), d.a())) +
                      static_cast<int>(islessgreater(d.a(), d.scalar_b())) +
                      static_cast<int>(isunordered(d.scalar_a(), d.b()));
        out.a += static_cast<double>(r);
      });
}
CERES_REGISTER_JET_BENCHMARK(QuietComparisonScalar);

template <std::size_t JET_SIZE>
static void Classification(benchmark::State& state) {
  using JetType = Jet<double, JET_SIZE>;
  JetBenchmarkHelper<JET_SIZE>(
      state, [](const JetInputData<JetType>& d, JetType& out) {
        const int r = static_cast<int>(isfinite(d.a())) +
                      static_cast<int>(isinf(d.a())) +
                      static_cast<int>(isnan(d.a())) +
                      static_cast<int>(isnormal(d.a())) + fpclassify(d.a()) +
                      static_cast<int>(signbit(d.unit_a()));
        out.a += static_cast<double>(r);
      });
}
CERES_REGISTER_JET_BENCHMARK(Classification);

#undef CERES_REGISTER_JET_BENCHMARK

}  // namespace ceres

BENCHMARK_MAIN();
