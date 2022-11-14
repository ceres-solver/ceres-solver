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
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/bundle_adjustment_test_util.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "ceres/evaluator.h"
#include "ceres/partitioned_matrix_view.h"
#include "ceres/preprocessor.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sparse_matrix.h"

namespace ceres::internal {

template <typename Derived, typename Base>
std::unique_ptr<Derived> downcast_unique_ptr(std::unique_ptr<Base>& base) {
  return std::unique_ptr<Derived>(dynamic_cast<Derived*>(base.release()));
}

// Benchmark library might invoke benchmark function multiple times.
// In order to save time required to parse BAL data, we ensure that
// each dataset is being loaded at most once.
// Each type of jacobians is also cached after first creation
struct BALData {
  using PartitionedView = PartitionedMatrixView<2, 3, 9>;
  explicit BALData(const std::string& path) {
    bal_problem = std::make_unique<BundleAdjustmentProblem>(path);
    CHECK(bal_problem != nullptr);

    auto problem_impl = bal_problem->mutable_problem()->mutable_impl();
    auto preprocessor = Preprocessor::Create(MinimizerType::TRUST_REGION);

    preprocessed_problem = std::make_unique<PreprocessedProblem>();
    Solver::Options options = bal_problem->options();
    options.linear_solver_type = ITERATIVE_SCHUR;
    CHECK(preprocessor->Preprocess(
        options, problem_impl, preprocessed_problem.get()));

    auto program = preprocessed_problem->reduced_program.get();

    parameters.resize(program->NumParameters());
    program->ParameterBlocksToStateVector(parameters.data());
  }

  std::unique_ptr<BlockSparseMatrix> CreateBlockSparseJacobian(
      ContextImpl* context) {
    auto problem = bal_problem->mutable_problem();
    auto problem_impl = problem->mutable_impl();
    CHECK(problem_impl != nullptr);

    Evaluator::Options options;
    options.linear_solver_type = ITERATIVE_SCHUR;
    options.num_threads = 1;
    options.context = context;
    options.num_eliminate_blocks = bal_problem->num_points();

    std::string error;
    auto program = preprocessed_problem->reduced_program.get();
    auto evaluator = Evaluator::Create(options, program, &error);
    CHECK(evaluator != nullptr);

    auto jacobian = evaluator->CreateJacobian();
    auto block_sparse = downcast_unique_ptr<BlockSparseMatrix>(jacobian);
    CHECK(block_sparse != nullptr);

    std::mt19937 rng;
    std::normal_distribution<double> rnorm;
    const int nnz = block_sparse->num_nonzeros();
    auto values = block_sparse->mutable_values();
    for (int i = 0; i < nnz; ++i) {
      values[i] = rnorm(rng);
    }
    return block_sparse;
  }

  std::unique_ptr<CompressedRowSparseMatrix> CreateCompressedRowSparseJacobian(
      ContextImpl* context) {
    auto block_sparse = BlockSparseJacobian(context);
    auto crs_jacobian = std::make_unique<CompressedRowSparseMatrix>(
        block_sparse->num_rows(),
        block_sparse->num_cols(),
        block_sparse->num_nonzeros());

    block_sparse->ToCompressedRowSparseMatrix(crs_jacobian.get());
    return crs_jacobian;
  }

  const BlockSparseMatrix* BlockSparseJacobian(ContextImpl* context) {
    if (!block_sparse_jacobian) {
      block_sparse_jacobian = CreateBlockSparseJacobian(context);
    }
    return block_sparse_jacobian.get();
  }

  const BlockSparseMatrix* BlockSparseJacobianWithTranspose(
      ContextImpl* context) {
    if (!block_sparse_jacobian_with_transpose) {
      block_sparse_jacobian_with_transpose = CreateBlockSparseJacobian(context);
      block_sparse_jacobian_with_transpose->AddTransposeBlockStructure();
    }
    return block_sparse_jacobian_with_transpose.get();
  }

  const CompressedRowSparseMatrix* CompressedRowSparseJacobian(
      ContextImpl* context) {
    if (!crs_jacobian) {
      crs_jacobian = CreateCompressedRowSparseJacobian(context);
    }
    return crs_jacobian.get();
  }

  const PartitionedView* PartitionedMatrixViewJacobian(
      const LinearSolver::Options& options) {
    auto block_sparse = BlockSparseJacobian(options.context);
    partitioned_view_jacobian =
        std::make_unique<PartitionedView>(options, *block_sparse);
    return partitioned_view_jacobian.get();
  }

  const PartitionedView* PartitionedMatrixViewJacobianWithTranspose(
      const LinearSolver::Options& options) {
    auto block_sparse_transpose =
        BlockSparseJacobianWithTranspose(options.context);
    partitioned_view_jacobian_with_transpose =
        std::make_unique<PartitionedView>(options, *block_sparse_transpose);
    return partitioned_view_jacobian_with_transpose.get();
  }

  Vector parameters;
  std::unique_ptr<BundleAdjustmentProblem> bal_problem;
  std::unique_ptr<PreprocessedProblem> preprocessed_problem;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian_with_transpose;
  std::unique_ptr<CompressedRowSparseMatrix> crs_jacobian;
  std::unique_ptr<PartitionedView> partitioned_view_jacobian;
  std::unique_ptr<PartitionedView> partitioned_view_jacobian_with_transpose;
};

static void Residuals(benchmark::State& state,
                      BALData* data,
                      ContextImpl* context) {
  const int num_threads = state.range(0);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.num_threads = num_threads;
  options.context = context;
  options.num_eliminate_blocks = 0;

  std::string error;
  CHECK(data->preprocessed_problem != nullptr);
  auto program = data->preprocessed_problem->reduced_program.get();
  CHECK(program != nullptr);
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
  const int num_threads = state.range(0);

  Evaluator::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.num_threads = num_threads;
  options.context = context;
  options.num_eliminate_blocks = 0;

  std::string error;
  CHECK(data->preprocessed_problem != nullptr);
  auto program = data->preprocessed_problem->reduced_program.get();
  CHECK(program != nullptr);
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

static void PMVRightMultiplyAndAccumulateF(benchmark::State& state,
                                           BALData* data,
                                           ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = state.range(0);
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);

  Vector y = Vector::Zero(jacobian->num_rows());
  Vector x = Vector::Random(jacobian->num_cols_f());

  for (auto _ : state) {
    jacobian->RightMultiplyAndAccumulateF(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void PMVLeftMultiplyAndAccumulateF(benchmark::State& state,
                                          BALData* data,
                                          ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = state.range(0);
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobianWithTranspose(options);

  Vector y = Vector::Zero(jacobian->num_cols_f());
  Vector x = Vector::Random(jacobian->num_rows());

  for (auto _ : state) {
    jacobian->LeftMultiplyAndAccumulateF(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void PMVRightMultiplyAndAccumulateE(benchmark::State& state,
                                           BALData* data,
                                           ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = state.range(0);
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);

  Vector y = Vector::Zero(jacobian->num_rows());
  Vector x = Vector::Random(jacobian->num_cols_e());

  for (auto _ : state) {
    jacobian->RightMultiplyAndAccumulateE(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void PMVLeftMultiplyAndAccumulateE(benchmark::State& state,
                                          BALData* data,
                                          ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = state.range(0);
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobianWithTranspose(options);

  Vector y = Vector::Zero(jacobian->num_cols_e());
  Vector x = Vector::Random(jacobian->num_rows());

  for (auto _ : state) {
    jacobian->LeftMultiplyAndAccumulateE(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void JacobianRightMultiplyAndAccumulate(benchmark::State& state,
                                               BALData* data,
                                               ContextImpl* context) {
  const int num_threads = state.range(0);

  auto jacobian = data->BlockSparseJacobian(context);

  Vector y = Vector::Zero(jacobian->num_rows());
  Vector x = Vector::Random(jacobian->num_cols());

  for (auto _ : state) {
    jacobian->RightMultiplyAndAccumulate(
        x.data(), y.data(), context, num_threads);
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void JacobianLeftMultiplyAndAccumulate(benchmark::State& state,
                                              BALData* data,
                                              ContextImpl* context) {
  const int num_threads = state.range(0);

  auto jacobian = data->BlockSparseJacobianWithTranspose(context);

  Vector y = Vector::Zero(jacobian->num_cols());
  Vector x = Vector::Random(jacobian->num_rows());

  for (auto _ : state) {
    jacobian->LeftMultiplyAndAccumulate(
        x.data(), y.data(), context, num_threads);
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

#ifndef CERES_NO_CUDA
static void JacobianRightMultiplyAndAccumulateCuda(benchmark::State& state,
                                                   BALData* data,
                                                   ContextImpl* context) {
  auto crs_jacobian = data->CompressedRowSparseJacobian(context);
  CudaSparseMatrix cuda_jacobian(context, *crs_jacobian);
  CudaVector cuda_x(context, 0);
  CudaVector cuda_y(context, 0);

  Vector x(crs_jacobian->num_cols());
  Vector y(crs_jacobian->num_rows());
  x.setRandom();
  y.setRandom();

  cuda_x.CopyFromCpu(x);
  cuda_y.CopyFromCpu(y);
  double sum = 0;
  for (auto _ : state) {
    cuda_jacobian.RightMultiplyAndAccumulate(cuda_x, &cuda_y);
    sum += cuda_y.Norm();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_NE(sum, 0.0);
}

static void JacobianLeftMultiplyAndAccumulateCuda(benchmark::State& state,
                                                  BALData* data,
                                                  ContextImpl* context) {
  auto crs_jacobian = data->CompressedRowSparseJacobian(context);
  CudaSparseMatrix cuda_jacobian(context, *crs_jacobian);
  CudaVector cuda_x(context, 0);
  CudaVector cuda_y(context, 0);

  Vector x(crs_jacobian->num_rows());
  Vector y(crs_jacobian->num_cols());
  x.setRandom();
  y.setRandom();

  cuda_x.CopyFromCpu(x);
  cuda_y.CopyFromCpu(y);
  double sum = 0;
  for (auto _ : state) {
    cuda_jacobian.LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
    sum += cuda_y.Norm();
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
  CHECK_NE(sum, 0.0);
}
#endif

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
#ifndef CERES_NO_CUDA
  std::string message;
  context.InitCuda(&message);
#endif

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

    const std::string name_right_product =
        "JacobianRightMultiplyAndAccumulate<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_right_product.c_str(),
        ceres::internal::JacobianRightMultiplyAndAccumulate,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

    const std::string name_right_product_partitioned_f =
        "PMVRightMultiplyAndAccumulateF<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_right_product_partitioned_f.c_str(),
        ceres::internal::PMVRightMultiplyAndAccumulateF,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

    const std::string name_right_product_partitioned_e =
        "PMVRightMultiplyAndAccumulateE<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_right_product_partitioned_e.c_str(),
        ceres::internal::PMVRightMultiplyAndAccumulateE,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

#ifndef CERES_NO_CUDA
    const std::string name_right_product_cuda =
        "JacobianRightMultiplyAndAccumulateCuda<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_right_product_cuda.c_str(),
        ceres::internal::JacobianRightMultiplyAndAccumulateCuda,
        data,
        &context)
        ->Arg(1);
#endif

    const std::string name_left_product =
        "JacobianLeftMultiplyAndAccumulate<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_left_product.c_str(),
        ceres::internal::JacobianLeftMultiplyAndAccumulate,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

    const std::string name_left_product_partitioned_f =
        "PMVLeftMultiplyAndAccumulateF<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_left_product_partitioned_f.c_str(),
        ceres::internal::PMVLeftMultiplyAndAccumulateF,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

    const std::string name_left_product_partitioned_e =
        "PMVLeftMultiplyAndAccumulateE<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_left_product_partitioned_e.c_str(),
        ceres::internal::PMVLeftMultiplyAndAccumulateE,
        data,
        &context)
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16);

#ifndef CERES_NO_CUDA
    const std::string name_left_product_cuda =
        "JacobianLeftMultiplyAndAccumulateCuda<" + path + ">";
    ::benchmark::RegisterBenchmark(
        name_left_product_cuda.c_str(),
        ceres::internal::JacobianLeftMultiplyAndAccumulateCuda,
        data,
        &context)
        ->Arg(1);
#endif
  }
  ::benchmark::RunSpecifiedBenchmarks();

  using namespace ::benchmark;
  using namespace benchmark_shutdown_fallback;
  Shutdown();
  return 0;
}
