// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Authors: sameeragarwal@google.com (Sameer Agarwal)
//#define EIGEN_DONT_VECTORIZE
#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/ceres.h"
#include "codegen/test_utils.h"
#include "trivial_cost_functions.h"

namespace ceres {

#ifdef WITH_CODE_GENERATION
static void BM_Linear1CodeGen(benchmark::State& state) {
  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new benchmark::Linear1CostFunction());

  while (state.KeepRunning()) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_Linear1CodeGen);
#endif
static void BM_Linear1AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<benchmark::Linear1CostFunction>;

  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 1, 1>(new FunctorType()));

  while (state.KeepRunning()) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_Linear1AutoDiff);
#ifdef WITH_CODE_GENERATION
static void BM_Linear10CodeGen(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new benchmark::Linear10CostFunction());

  while (state.KeepRunning()) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_Linear10CodeGen);
#endif
static void BM_Linear10AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<benchmark::Linear10CostFunction>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 10, 10>(new FunctorType()));

  while (state.KeepRunning()) {
    cost_function->Evaluate(parameters, residuals, jacobians);
  }
}
BENCHMARK(BM_Linear10AutoDiff);

}  // namespace ceres

BENCHMARK_MAIN();
