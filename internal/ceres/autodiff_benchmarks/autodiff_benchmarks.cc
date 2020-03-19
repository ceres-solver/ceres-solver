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
// Author: darius.rueckert@fau.de (Darius Rueckert)

#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"
#include "ceres/ceres.h"
#include "ceres/codegen/test_utils.h"

namespace ceres {

#ifdef WITH_CODE_GENERATION
static void BM_Linear1CodeGen(benchmark::State& state) {
  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(new Linear1CostFunction());

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear1CodeGen)->Arg(0)->Arg(1);
#endif

static void BM_Linear1AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<Linear1CostFunction>;

  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 1, 1>(new FunctorType()));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear1AutoDiff)->Arg(0)->Arg(1);
;

#ifdef WITH_CODE_GENERATION
static void BM_Linear10CodeGen(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new Linear10CostFunction());

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear10CodeGen)->Arg(0)->Arg(1);
;
#endif

static void BM_Linear10AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<Linear10CostFunction>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 10, 10>(new FunctorType()));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear10AutoDiff)->Arg(0)->Arg(1);
;

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
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

static void BM_Rat43AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(
          new Rat43CostFunctor(x, y)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Rat43AutoDiff)->Arg(0)->Arg(1);

#ifdef WITH_CODE_GENERATION
static void BM_SnavelyReprojectionCodeGen(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;

  std::unique_ptr<ceres::CostFunction> cost_function(
      new SnavelyReprojectionError(x, y));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_SnavelyReprojectionCodeGen)->Arg(0)->Arg(1);
;
#endif

static void BM_SnavelyReprojectionAutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<SnavelyReprojectionError>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 2, 9, 3>(
          new FunctorType(x, y)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_SnavelyReprojectionAutoDiff)->Arg(0)->Arg(1);
;

#ifdef WITH_CODE_GENERATION
static void BM_BrdfCodeGen(benchmark::State& state) {
  using FunctorType = ceres::internal::CostFunctionToFunctor<Brdf>;

  double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  auto c = Eigen::Vector3d(0.1, 0.2, 0.3);
  auto n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  auto v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  auto l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  auto x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  auto y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  double* parameters[7] = {
      material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()};

  double jacobian[(10 + 6 * 3) * 3];
  double residuals[3];
  double* jacobians[7] = {
      jacobian + 0,
      jacobian + 10 * 3,
      jacobian + 13 * 3,
      jacobian + 16 * 3,
      jacobian + 19 * 3,
      jacobian + 22 * 3,
      jacobian + 25 * 3,
  };

  std::unique_ptr<ceres::CostFunction> cost_function(new Brdf());

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_BrdfCodeGen)->Arg(0)->Arg(1);
;
#endif

static void BM_BrdfAutoDiff(benchmark::State& state) {
  using FunctorType = ceres::internal::CostFunctionToFunctor<Brdf>;

  double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  auto c = Eigen::Vector3d(0.1, 0.2, 0.3);
  auto n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  auto v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  auto l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  auto x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  auto y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  double* parameters[7] = {
      material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()};

  double jacobian[(10 + 6 * 3) * 3];
  double residuals[3];
  double* jacobians[7] = {
      jacobian + 0,
      jacobian + 10 * 3,
      jacobian + 13 * 3,
      jacobian + 16 * 3,
      jacobian + 19 * 3,
      jacobian + 22 * 3,
      jacobian + 25 * 3,
  };

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 3, 10, 3, 3, 3, 3, 3, 3>(
          new FunctorType));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_BrdfAutoDiff)->Arg(0)->Arg(1);
;

}  // namespace ceres

BENCHMARK_MAIN();
