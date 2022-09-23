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
//
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "ceres/bundle_adjustment_test_util.h"
#include "ceres/evaluator.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sparse_matrix.h"
#include "gflags/gflags.h"

namespace ceres::internal {

// Benchmark library might invoke benchmark function multiple times.
// In order to save time required to parse BAL data, we ensure that
// each dataset is being loaded at most once.
struct BALData {
  explicit BALData(const std::string& path) {
    bal_problem = std::make_unique<BundleAdjustmentProblem>(path);
    CHECK(bal_problem != nullptr);

    auto problem_impl = bal_problem->mutable_problem()->mutable_impl();
    auto program = problem_impl->mutable_program();
    parameters.resize(program->NumParameters());
    program->ParameterBlocksToStateVector(parameters.data());
  }

  Vector parameters;
  std::unique_ptr<BundleAdjustmentProblem> bal_problem;
};

static void Residuals(benchmark::State& state,
                      BALData* data,
                      ContextImpl* context) {
  auto problem = data->bal_problem->mutable_problem();
  auto problem_impl = problem->mutable_impl();
  CHECK(problem_impl != nullptr);
  const int num_threads = state.range(0);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.num_threads = num_threads;
  options.context = context;
  options.num_eliminate_blocks = 0;

  std::string error;
  auto program = problem_impl->mutable_program();
  auto evaluator = Evaluator::Create(options, program, &error);
  CHECK(evaluator != nullptr);

  double cost = 0.;
  Vector residuals = Vector::Zero(program->NumResiduals());

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->parameters.data(),
                              &cost,
                              residuals.data(),
                              nullptr,
                              nullptr));
  }
}

static void ResidualsAndJacobian(benchmark::State& state,
                                 BALData* data,
                                 ContextImpl* context) {
  auto problem = data->bal_problem->mutable_problem();
  auto problem_impl = problem->mutable_impl();
  CHECK(problem_impl != nullptr);
  const int num_threads = state.range(0);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.num_threads = num_threads;
  options.context = context;
  options.num_eliminate_blocks = 0;

  std::string error;
  auto program = problem_impl->mutable_program();
  auto evaluator = Evaluator::Create(options, program, &error);
  CHECK(evaluator != nullptr);

  double cost = 0.;
  Vector residuals = Vector::Zero(program->NumResiduals());
  auto jacobian = evaluator->CreateJacobian();

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->parameters.data(),
                              &cost,
                              residuals.data(),
                              nullptr,
                              jacobian.get()));
  }
}

}  // namespace ceres::internal

// Older versions of benchmark library might come without ::benchmark::Shutdown
// function. We provide an empty fallback variant of Shutdown function in
// order to support both older and newer versions
namespace benchmark_shutdown_fallback {
template <typename... Args>
void Shutdown(Args... args) {}
};  // namespace benchmark_shutdown_fallback

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);

  std::vector<std::unique_ptr<ceres::internal::BALData>> benchmark_data;
  if (argc == 1) {
    LOG(FATAL) << "No input datasets specified. Usage: " << argv[0]
               << " [benchmark flags] path_to_BAL_data_1.txt ... "
                  "path_to_BAL_data_N.txt";
    return -1;
  }

  ceres::internal::ContextImpl context;
  context.EnsureMinimumThreads(16);

  for (int i = 1; i < argc; ++i) {
    const std::string path(argv[i]);
    const std::string name_residuals = "Residuals<" + path + ">";
    benchmark_data.emplace_back(
        std::make_unique<ceres::internal::BALData>(path));
    auto data = benchmark_data.back().get();
    ::benchmark::RegisterBenchmark(
        name_residuals.c_str(), ceres::internal::Residuals, data, &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);
    const std::string name_jacobians = "ResidualsAndJacobian<" + path + ">";
    ::benchmark::RegisterBenchmark(name_jacobians.c_str(),
                                   ceres::internal::ResidualsAndJacobian,
                                   data,
                                   &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);
  }

  ::benchmark::RunSpecifiedBenchmarks();

  using namespace ::benchmark;
  using namespace benchmark_shutdown_fallback;
  Shutdown();
  return 0;
}
