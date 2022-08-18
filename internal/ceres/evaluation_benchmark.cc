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

DEFINE_string(bal_root, "", "Path to `data` directory of BAL dataset");

namespace ceres {
class BenchmarkProblem : public Problem {
 public:
  BenchmarkProblem(Problem&& problem) : Problem(std::move(problem)) {}
  internal::ProblemImpl* problem_impl() const { return impl_.get(); }
};
}  // namespace ceres

namespace ceres::internal {

// Data paths relative to BAL dataset root
const char* paths[] = {"/dubrovnik/problem-356-226730-pre.txt",
                       "/final/problem-13682-4456117-pre.txt",
                       "/ladybug/problem-1723-156502-pre.txt",
                       "/trafalgar/problem-257-65132-pre.txt",
                       "/venice/problem-1778-993923-pre.txt"};

// Evaluation time per one iteration will be quite large, especially for the
// largest input, which might lead to benchmark library running function only
// once. Thus we force a specific number of iterations.
const int kNumIterations = 20;

// Benchmark library might invoke benchmark function multiple times.
// In order to save time required to parse BAL data, we ensure that
// each dataset is being loaded at most once.
template <int id>
class BAL_Data {
 private:
  BAL_Data() {
    const std::string problem = FLAGS_bal_root + paths[id];
    auto f = fopen(problem.c_str(), "r");
    if (!f) {
      LOG(FATAL) << "Unable to open " << problem
                 << "; set --bal_root to the data directory of BAL dataset";
    }
    fclose(f);

    bal_problem_ = std::make_unique<BundleAdjustmentProblem>(problem);
    benchmark_problem_ = std::make_unique<BenchmarkProblem>(
        std::move(*bal_problem_->mutable_problem()));

    ContextImpl context;
    auto problem_impl = benchmark_problem_->problem_impl();
    Evaluator::Options options;
    options.linear_solver_type = SPARSE_SCHUR;
    options.num_threads = 1;
    options.num_eliminate_blocks = 0;
    options.context = &context;
    auto program = problem_impl->mutable_program();
    std::string error;
    auto evaluator = Evaluator::Create(options, program, &error);
    CHECK(evaluator);

    res_.resize(evaluator->NumResiduals());
    at_.resize(evaluator->NumParameters());
    program->ParameterBlocksToStateVector(at_.data());
    jacobian_ = evaluator->CreateJacobian();
  }

 public:
  static BAL_Data& Create() {
    static BAL_Data data;
    return data;
  }

  std::unique_ptr<BundleAdjustmentProblem> bal_problem_;
  std::unique_ptr<BenchmarkProblem> benchmark_problem_;
  Vector res_;
  Vector at_;
  std::unique_ptr<SparseMatrix> jacobian_;
};

using Dubrovnik356 = BAL_Data<0>;
using Final13682 = BAL_Data<1>;
using Ladybug1723 = BAL_Data<2>;
using Trafalgar257 = BAL_Data<3>;
using Venice1778 = BAL_Data<4>;

template <typename T>
static void residuals(benchmark::State& state) {
  auto data = &T::Create();
  auto problem_impl = data->benchmark_problem_->problem_impl();
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

  data->res_.setZero();

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->at_.data(),
                              &cost,
                              data->res_.data(),
                              nullptr,
                              nullptr));
  }
}

template <typename T>
static void residuals_and_jacobian(benchmark::State& state) {
  auto data = &T::Create();
  auto problem_impl = data->benchmark_problem_->problem_impl();
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

  data->res_.setZero();

  Evaluator::EvaluateOptions eval_options;
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              data->at_.data(),
                              &cost,
                              data->res_.data(),
                              nullptr,
                              data->jacobian_.get()));
  }
  CHECK(problem_impl);
}

#define DEFINE_BENCHMARK(benchmark, dataset) \
  BENCHMARK(benchmark<dataset>) \
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Iterations(kNumIterations);

#define DEFINE_BENCHMARKS(benchmark) \
  DEFINE_BENCHMARK(benchmark, Trafalgar257) \
  DEFINE_BENCHMARK(benchmark, Dubrovnik356) \
  DEFINE_BENCHMARK(benchmark, Ladybug1723) \
  DEFINE_BENCHMARK(benchmark, Venice1778) \
  DEFINE_BENCHMARK(benchmark, Final13682)

DEFINE_BENCHMARKS(residuals)
DEFINE_BENCHMARKS(residuals_and_jacobian)

}  // namespace ceres::internal

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
