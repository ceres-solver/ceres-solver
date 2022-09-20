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

#include <benchmark/benchmark.h>
#include <gflags/gflags.h>

#include "ceres/bundle_adjustment_test_util.h"
#include "ceres/evaluator.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sparse_matrix.h"

namespace ceres::internal {

// Benchmark library might invoke benchmark function multiple times.
// In order to save time required to parse BAL data, we ensure that
// each dataset is being loaded at most once.
struct BAL_Data {
  BAL_Data(const std::string& path) {
    auto f = fopen(path.c_str(), "r");
    if (!f) {
      LOG(FATAL) << "Unable to open " << path;
    }
    fclose(f);

    bal_problem_ = std::make_unique<BundleAdjustmentProblem>(path);

    ContextImpl context;
    auto problem_impl = bal_problem_->mutable_problem()->MutableProblemImpl();
    Evaluator::Options options;
    options.linear_solver_type = SPARSE_SCHUR;
    options.num_threads = 1;
    options.num_eliminate_blocks = 0;
    options.context = &context;
    auto program = problem_impl->mutable_program();
    std::string error;
    auto evaluator = Evaluator::Create(options, program, &error);
    CHECK(evaluator != nullptr);

    residuals_.resize(evaluator->NumResiduals());
    parameters_.resize(evaluator->NumParameters());
    program->ParameterBlocksToStateVector(parameters_.data());
    jacobian_ = evaluator->CreateJacobian();
  }

  std::unique_ptr<BundleAdjustmentProblem> bal_problem_;
  Vector residuals_;
  Vector parameters_;
  std::unique_ptr<SparseMatrix> jacobian_;
};

static void Residuals(benchmark::State& state, BAL_Data* data) {
  auto problem = data->bal_problem_->mutable_problem();
  auto problem_impl = problem->MutableProblemImpl();
  CHECK(problem_impl != nullptr);
  const int num_threads = state.range(0);

  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_SCHUR;
  options.num_threads = num_threads;
  options.context = &context;
  options.num_eliminate_blocks = 0;

  std::string error;
  auto program = problem_impl->mutable_program();
  auto evaluator = Evaluator::Create(options, program, &error);
  CHECK(evaluator != nullptr);
  double cost = 0.;

  data->residuals_.setZero();

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->parameters_.data(),
                              &cost,
                              data->residuals_.data(),
                              nullptr,
                              nullptr));
  }
}

static void ResidualsAndJacobian(benchmark::State& state, BAL_Data* data) {
  auto problem = data->bal_problem_->mutable_problem();
  auto problem_impl = problem->MutableProblemImpl();
  CHECK(problem_impl != nullptr);
  const int num_threads = state.range(0);

  ContextImpl context;
  context.EnsureMinimumThreads(num_threads);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_SCHUR;
  options.num_threads = num_threads;
  options.context = &context;
  options.num_eliminate_blocks = 0;

  std::string error;
  auto program = problem_impl->mutable_program();
  auto evaluator = Evaluator::Create(options, program, &error);
  CHECK(evaluator != nullptr);
  double cost = 0.;

  data->residuals_.setZero();

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->parameters_.data(),
                              &cost,
                              data->residuals_.data(),
                              nullptr,
                              data->jacobian_.get()));
  }
}

}  // namespace ceres::internal

int main(int argc, char** argv) {
  using namespace ceres::internal;
  ::benchmark::Initialize(&argc, argv);

  std::vector<std::unique_ptr<BAL_Data>> benchmark_data;
  if (argc == 1) {
    LOG(FATAL) << "No input datasets specified. Usage: " << argv[0]
               << " [benchmark flags] path_to_BAL_data_1.txt ... "
                  "path_to_BAL_data_N.txt";
    return -1;
  }

  for (int i = 1; i < argc; ++i) {
    const std::string path(argv[i]);
    const std::string name_residuals = "Residuals<" + path + ">";
    benchmark_data.emplace_back(std::make_unique<BAL_Data>(path));
    auto data = benchmark_data.back().get();
    ::benchmark::RegisterBenchmark(name_residuals.c_str(), Residuals, data)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);
    const std::string name_jacobians = "ResidualsAndJacobian<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_jacobians.c_str(), ResidualsAndJacobian, data)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);
  }

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
