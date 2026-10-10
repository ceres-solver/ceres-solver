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
// Authors: dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#include <algorithm>
#include <array>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "benchmark/benchmark.h"
#include "ceres/benchmark_cost_functions.h"
#include "ceres/block_evaluate_preparer.h"
#include "ceres/block_jacobian_writer.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/bundle_adjustment_test_util.h"
#include "ceres/compressed_row_jacobian_writer.h"
#include "ceres/cuda_block_sparse_crs_view.h"
#include "ceres/cuda_partitioned_block_sparse_crs_view.h"
#include "ceres/cuda_sparse_matrix.h"
#include "ceres/cuda_vector.h"
#include "ceres/evaluator.h"
#include "ceres/implicit_schur_complement.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/partitioned_matrix_view.h"
#include "ceres/power_series_expansion_preconditioner.h"
#include "ceres/preprocessor.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/program_evaluator.h"
#include "ceres/scratch_evaluate_preparer.h"
#include "ceres/sparse_matrix.h"

namespace ceres::internal {

template <typename Derived, typename Base>
std::unique_ptr<Derived> downcast_unique_ptr(std::unique_ptr<Base>& base) {
  return std::unique_ptr<Derived>(dynamic_cast<Derived*>(base.release()));
}

// Builds and preprocesses a bundle adjustment Problem from a BAL dataset for
// any reprojection `CostFunctor` in benchmark_cost_functions.h.
//
// Using `CostFunctor::ParameterDims`, `CostFunctor::kPointBlockIndex`,
// `CostFunctor::MakeParametersFromBAL`, `CostFunctor::MakeManifold`, and
// `CostFunctor::MakeLossFunction`, this template:
//   1. Allocates `num_points` blocks for `b == CostFunctor::kPointBlockIndex`
//      and `num_cameras` blocks for all other camera block indices `b`,
//      initializing their values from the BAL cameras and 3D points.
//   2. Connects the point and camera blocks along the BAL bipartite edges
//      `(camera_index[i], point_index[i])` with `AutoDiffCostFunction` and
//      `CostFunctor::MakeLossFunction()`.
//   3. Attaches `CostFunctor::MakeManifold(b)` to each block at index `b`,
//      places points in Schur elimination group 0 and camera blocks in group 1,
//      and runs `Preprocessor::Preprocess`.
template <typename CostFunctor,
          typename ParameterDims = typename CostFunctor::ParameterDims>
struct EvaluatorTestProblem;

template <typename CostFunctor, int... Ns>
struct EvaluatorTestProblem<CostFunctor, StaticParameterDims<Ns...>> {
  static constexpr int kNumResiduals = CostFunctor::kNumResiduals;
  static constexpr int kNumBlocks = sizeof...(Ns);
  static constexpr std::array<int, kNumBlocks> kBlockSizes = {Ns...};

  explicit EvaluatorTestProblem(BundleAdjustmentProblem& bal_problem)
      : problem(MakeProblemOptions()),
        loss_function(CostFunctor::MakeLossFunction()) {
    const int num_cams = bal_problem.num_cameras();
    const int num_pts = bal_problem.num_points();
    const int num_obs = bal_problem.num_observations();
    const double* orig_cams = bal_problem.mutable_cameras();
    const double* orig_pts = bal_problem.mutable_points();
    const int* cam_idx = bal_problem.camera_index();
    const int* pt_idx = bal_problem.point_index();
    const double* obs = bal_problem.observations();

    std::array<double*, kNumBlocks> block_bases{};
    for (int b = 0; b < kNumBlocks; ++b) {
      const int count =
          (b == CostFunctor::kPointBlockIndex) ? num_pts : num_cams;
      block_storage[b].resize(count * kBlockSizes[b], 0.0);
      block_bases[b] = block_storage[b].data();
      manifolds[b] = CostFunctor::MakeManifold(b);
    }

    for (int p = 0; p < num_pts; ++p) {
      const auto params =
          CostFunctor::MakeParametersFromBAL(orig_cams, orig_pts + 3 * p);
      const auto ptrs = params.Pointers();
      constexpr int b = CostFunctor::kPointBlockIndex;
      std::copy_n(
          ptrs[b], kBlockSizes[b], block_bases[b] + p * kBlockSizes[b]);
    }

    for (int c = 0; c < num_cams; ++c) {
      const auto params =
          CostFunctor::MakeParametersFromBAL(orig_cams + 9 * c, orig_pts);
      const auto ptrs = params.Pointers();
      for (int b = 0; b < kNumBlocks; ++b) {
        if (b != CostFunctor::kPointBlockIndex) {
          std::copy_n(
              ptrs[b], kBlockSizes[b], block_bases[b] + c * kBlockSizes[b]);
        }
      }
    }

    std::array<double*, kNumBlocks> residual_blocks{};
    for (int i = 0; i < num_obs; ++i) {
      for (int b = 0; b < kNumBlocks; ++b) {
        const int idx =
            (b == CostFunctor::kPointBlockIndex) ? pt_idx[i] : cam_idx[i];
        residual_blocks[b] = block_bases[b] + idx * kBlockSizes[b];
      }
      CostFunction* cost_function =
          new AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>(
              new CostFunctor(obs[2 * i + 0], obs[2 * i + 1]));
      problem.AddResidualBlock(cost_function,
                               loss_function.get(),
                               residual_blocks.data(),
                               kNumBlocks);
    }

    Solver::Options options = bal_problem.options();
    options.linear_solver_type = ITERATIVE_SCHUR;
    options.linear_solver_ordering =
        std::make_shared<ParameterBlockOrdering>();
    for (int b = 0; b < kNumBlocks; ++b) {
      const int count =
          (b == CostFunctor::kPointBlockIndex) ? num_pts : num_cams;
      const int group = (b == CostFunctor::kPointBlockIndex) ? 0 : 1;
      for (int idx = 0; idx < count; ++idx) {
        double* block_ptr = block_bases[b] + idx * kBlockSizes[b];
        if (manifolds[b] != nullptr) {
          problem.SetManifold(block_ptr, manifolds[b].get());
        }
        options.linear_solver_ordering->AddElementToGroup(block_ptr, group);
      }
    }

    auto preprocessor = Preprocessor::Create(MinimizerType::TRUST_REGION);
    preprocessed_problem = std::make_unique<PreprocessedProblem>();
    CHECK(preprocessor->Preprocess(
        options, problem.mutable_impl(), preprocessed_problem.get()));
    auto* program = preprocessed_problem->reduced_program.get();
    parameters.resize(program->NumParameters());
    program->ParameterBlocksToStateVector(parameters.data());
  }

  static Problem::Options MakeProblemOptions() {
    Problem::Options options;
    options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
    options.manifold_ownership = DO_NOT_TAKE_OWNERSHIP;
    return options;
  }

  Problem problem;
  std::unique_ptr<LossFunction> loss_function;
  std::array<std::unique_ptr<Manifold>, kNumBlocks> manifolds;
  std::array<std::vector<double>, kNumBlocks> block_storage;
  std::unique_ptr<PreprocessedProblem> preprocessed_problem;
  Vector parameters;
};

// Benchmark library might invoke benchmark function multiple times.
// In order to save time required to parse BAL data, we ensure that
// each dataset is being loaded at most once.
// Each type of jacobians is also cached after first creation.
struct BALData {
  using PartitionedView = PartitionedMatrixView<2, 3, 9>;
  explicit BALData(const std::string& path) {
    bal_problem = std::make_unique<BundleAdjustmentProblem>(path);
    CHECK(bal_problem != nullptr);

    auto* program = ProblemFor<SnavelyReprojectionError>()
                        ->preprocessed_problem->reduced_program.get();

    const int num_residuals = program->NumResiduals();
    b.resize(num_residuals);

    std::mt19937 rng;
    std::normal_distribution<double> rnorm;
    for (int i = 0; i < num_residuals; ++i) {
      b[i] = rnorm(rng);
    }

    const int num_parameters = program->NumParameters();
    D.resize(num_parameters);
    for (int i = 0; i < num_parameters; ++i) {
      D[i] = rnorm(rng);
    }
  }

  template <typename CostFunctor>
  EvaluatorTestProblem<CostFunctor>* ProblemFor() {
    auto& slot =
        std::get<std::unique_ptr<EvaluatorTestProblem<CostFunctor>>>(
            test_problems);
    if (!slot) {
      slot = std::make_unique<EvaluatorTestProblem<CostFunctor>>(*bal_problem);
    }
    return slot.get();
  }

  std::unique_ptr<BlockSparseMatrix> CreateBlockSparseJacobian(
      ContextImpl* context, bool sequential) {
    Evaluator::Options options;
    options.linear_solver_type = ITERATIVE_SCHUR;
    options.num_threads = 1;
    options.context = context;
    options.num_eliminate_blocks = bal_problem->num_points();

    std::string error;
    auto* program = ProblemFor<SnavelyReprojectionError>()
                        ->preprocessed_problem->reduced_program.get();
    auto evaluator = Evaluator::Create(options, program, &error);
    CHECK(evaluator != nullptr);

    auto jacobian = evaluator->CreateJacobian();
    auto block_sparse = downcast_unique_ptr<BlockSparseMatrix>(jacobian);
    CHECK(block_sparse != nullptr);

    if (sequential) {
      auto block_structure_sequential =
          std::make_unique<CompressedRowBlockStructure>(
              *block_sparse->block_structure());
      int num_nonzeros = 0;
      for (auto& row_block : block_structure_sequential->rows) {
        const int row_block_size = row_block.block.size;
        for (auto& cell : row_block.cells) {
          const int col_block_size =
              block_structure_sequential->cols[cell.block_id].size;
          cell.position = num_nonzeros;
          num_nonzeros += col_block_size * row_block_size;
        }
      }
      block_sparse = std::make_unique<BlockSparseMatrix>(
          block_structure_sequential.release(),
#ifndef CERES_NO_CUDA
          true
#else
          false
#endif
      );
    }

    std::mt19937 rng;
    std::normal_distribution<double> rnorm;
    const int nnz = block_sparse->num_nonzeros();
    auto values = block_sparse->mutable_values();
    for (int i = 0; i < nnz; ++i) {
      values[i] = rnorm(rng);
    }

    return block_sparse;
  }

  const BlockSparseMatrix* BlockSparseJacobian(ContextImpl* context) {
    if (!block_sparse_jacobian) {
      block_sparse_jacobian = CreateBlockSparseJacobian(context, true);
    }
    return block_sparse_jacobian.get();
  }

  const BlockSparseMatrix* BlockSparseJacobianPartitioned(
      ContextImpl* context) {
    if (!block_sparse_jacobian_partitioned) {
      block_sparse_jacobian_partitioned =
          CreateBlockSparseJacobian(context, false);
    }
    return block_sparse_jacobian_partitioned.get();
  }

  const CompressedRowSparseMatrix* CompressedRowSparseJacobian(
      ContextImpl* context) {
    if (!crs_jacobian) {
      crs_jacobian =
          BlockSparseJacobian(context)->ToCompressedRowSparseMatrix();
    }
    return crs_jacobian.get();
  }

  std::unique_ptr<PartitionedView> PartitionedMatrixViewJacobian(
      const LinearSolver::Options& options) {
    auto block_sparse = BlockSparseJacobianPartitioned(options.context);
    return std::make_unique<PartitionedView>(options, *block_sparse);
  }

  BlockSparseMatrix* BlockDiagonalEtE(const LinearSolver::Options& options) {
    if (!block_diagonal_ete) {
      auto partitioned_view = PartitionedMatrixViewJacobian(options);
      block_diagonal_ete = partitioned_view->CreateBlockDiagonalEtE();
    }
    return block_diagonal_ete.get();
  }

  BlockSparseMatrix* BlockDiagonalFtF(const LinearSolver::Options& options) {
    if (!block_diagonal_ftf) {
      auto partitioned_view = PartitionedMatrixViewJacobian(options);
      block_diagonal_ftf = partitioned_view->CreateBlockDiagonalFtF();
    }
    return block_diagonal_ftf.get();
  }

  const ImplicitSchurComplement* ImplicitSchurComplementWithoutDiagonal(
      const LinearSolver::Options& options) {
    if (!implicit_schur_complement) {
      auto block_sparse = BlockSparseJacobianPartitioned(options.context);
      implicit_schur_complement =
          std::make_unique<ImplicitSchurComplement>(options);
      implicit_schur_complement->Init(*block_sparse, nullptr, b.data());
    }
    return implicit_schur_complement.get();
  }

  const ImplicitSchurComplement* ImplicitSchurComplementWithDiagonal(
      const LinearSolver::Options& options) {
    if (!implicit_schur_complement_diag) {
      auto block_sparse = BlockSparseJacobianPartitioned(options.context);
      implicit_schur_complement_diag =
          std::make_unique<ImplicitSchurComplement>(options);
      implicit_schur_complement_diag->Init(*block_sparse, D.data(), b.data());
    }
    return implicit_schur_complement_diag.get();
  }

  Vector D;
  Vector b;
  std::unique_ptr<BundleAdjustmentProblem> bal_problem;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian_partitioned;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian;
  std::unique_ptr<CompressedRowSparseMatrix> crs_jacobian;
  std::unique_ptr<BlockSparseMatrix> block_diagonal_ete;
  std::unique_ptr<BlockSparseMatrix> block_diagonal_ftf;
  std::unique_ptr<ImplicitSchurComplement> implicit_schur_complement;
  std::unique_ptr<ImplicitSchurComplement> implicit_schur_complement_diag;
  std::tuple<
      std::unique_ptr<EvaluatorTestProblem<SnavelyReprojectionError>>,
      std::unique_ptr<EvaluatorTestProblem<ColmapOpenCVReprojectionError>>,
      std::unique_ptr<EvaluatorTestProblem<LibmvBrownReprojectionError>>>
      test_problems;
};

// Outputs requested from `Evaluator::Evaluate`:
//   - kResidualsOnly:                cost + residuals
//   - kResidualsAndJacobian:         cost + residuals + Jacobian
//   - kResidualsGradientAndJacobian: cost + residuals + gradient + Jacobian
enum class EvaluationOutputs {
  kResidualsOnly,
  kResidualsAndJacobian,
  kResidualsGradientAndJacobian,
};

// Sparse matrix representation written by `ProgramEvaluator`:
//   - kNone:          no Jacobian (residuals only)
//   - kBlockSparse:   BlockEvaluatePreparer + BlockJacobianWriter
//                     (used by Schur and SparseNormalCholesky solvers)
//   - kCompressedRow: ScratchEvaluatePreparer + CompressedRowJacobianWriter
//                     (used by Problem::Evaluate and CUDA CGNR)
enum class JacobianFormat {
  kNone,
  kBlockSparse,
  kCompressedRow,
};

// Benchmarks whole-problem `Evaluator::Evaluate` for `CostFunctor` across
// output combinations, sparse Jacobian storage formats, and thread counts.
template <typename CostFunctor,
          EvaluationOutputs kOutputs,
          JacobianFormat kJacobianFormat>
static void EvaluateProgram(benchmark::State& state,
                            BALData* data,
                            ContextImpl* context) {
  auto* test_problem = data->ProblemFor<CostFunctor>();
  Program* program = test_problem->preprocessed_problem->reduced_program.get();
  const double* parameters = test_problem->parameters.data();

  Evaluator::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.context = context;
  options.linear_solver_type = ITERATIVE_SCHUR;
  options.num_eliminate_blocks = data->bal_problem->num_points();

  std::unique_ptr<Evaluator> evaluator;
  if constexpr (kJacobianFormat == JacobianFormat::kCompressedRow) {
    evaluator = std::make_unique<
        ProgramEvaluator<ScratchEvaluatePreparer, CompressedRowJacobianWriter>>(
        options, program);
  } else {
    evaluator = std::make_unique<
        ProgramEvaluator<BlockEvaluatePreparer, BlockJacobianWriter>>(
        options, program);
  }

  double cost = 0.;
  Vector residuals = Vector::Zero(program->NumResiduals());
  Vector gradient;
  double* gradient_ptr = nullptr;
  if constexpr (kOutputs == EvaluationOutputs::kResidualsGradientAndJacobian) {
    gradient = Vector::Zero(program->NumEffectiveParameters());
    gradient_ptr = gradient.data();
  }

  std::unique_ptr<SparseMatrix> jacobian;
  SparseMatrix* jacobian_ptr = nullptr;
  if constexpr (kOutputs != EvaluationOutputs::kResidualsOnly) {
    jacobian = evaluator->CreateJacobian();
    jacobian_ptr = jacobian.get();
  }

  Evaluator::EvaluateOptions eval_options;
  CHECK(evaluator->Evaluate(eval_options,
                            parameters,
                            &cost,
                            residuals.data(),
                            gradient_ptr,
                            jacobian_ptr));

  for (auto _ : state) {
    benchmark::DoNotOptimize(evaluator->Evaluate(eval_options,
                                                 parameters,
                                                 &cost,
                                                 residuals.data(),
                                                 gradient_ptr,
                                                 jacobian_ptr));
    benchmark::DoNotOptimize(cost);
    benchmark::DoNotOptimize(residuals.data());
  }
}

static void Plus(benchmark::State& state, BALData* data, ContextImpl* context) {
  Evaluator::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.num_threads = static_cast<int>(state.range(0));
  options.context = context;
  options.num_eliminate_blocks = 0;

  auto* test_problem = data->ProblemFor<SnavelyReprojectionError>();
  Program* program = test_problem->preprocessed_problem->reduced_program.get();
  ProgramEvaluator<BlockEvaluatePreparer, BlockJacobianWriter> evaluator(
      options, program);

  Vector state_plus_delta = Vector::Zero(program->NumParameters());
  Vector delta = Vector::Random(program->NumEffectiveParameters());

  CHECK(evaluator.Plus(
      test_problem->parameters.data(), delta.data(), state_plus_delta.data()));

  for (auto _ : state) {
    benchmark::DoNotOptimize(evaluator.Plus(test_problem->parameters.data(),
                                            delta.data(),
                                            state_plus_delta.data()));
    benchmark::DoNotOptimize(state_plus_delta.data());
  }
  CHECK_GT(state_plus_delta.squaredNorm(), 0.);
}

enum class Submatrix { kE, kF };
enum class MultiplyDirection { kRight, kLeft };

static void PSEPreconditioner(benchmark::State& state,
                              BALData* data,
                              ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;

  auto jacobian = data->ImplicitSchurComplementWithDiagonal(options);
  Preconditioner::Options preconditioner_options(options);

  PowerSeriesExpansionPreconditioner preconditioner(
      jacobian, 10, 0, preconditioner_options);

  Vector y = Vector::Zero(jacobian->num_cols());
  Vector x = Vector::Random(jacobian->num_cols());

  for (auto _ : state) {
    preconditioner.RightMultiplyAndAccumulate(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

template <Submatrix kSubmatrix, MultiplyDirection kDirection>
static void PMVMultiply(benchmark::State& state,
                        BALData* data,
                        ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);

  const int num_cols = (kSubmatrix == Submatrix::kF) ? jacobian->num_cols_f()
                                                     : jacobian->num_cols_e();
  const int num_rows = jacobian->num_rows();
  Vector y = Vector::Zero(
      (kDirection == MultiplyDirection::kRight) ? num_rows : num_cols);
  Vector x = Vector::Random(
      (kDirection == MultiplyDirection::kRight) ? num_cols : num_rows);

  for (auto _ : state) {
    if constexpr (kSubmatrix == Submatrix::kF &&
                  kDirection == MultiplyDirection::kRight) {
      jacobian->RightMultiplyAndAccumulateF(x.data(), y.data());
    } else if constexpr (kSubmatrix == Submatrix::kF &&
                         kDirection == MultiplyDirection::kLeft) {
      jacobian->LeftMultiplyAndAccumulateF(x.data(), y.data());
    } else if constexpr (kSubmatrix == Submatrix::kE &&
                         kDirection == MultiplyDirection::kRight) {
      jacobian->RightMultiplyAndAccumulateE(x.data(), y.data());
    } else {
      jacobian->LeftMultiplyAndAccumulateE(x.data(), y.data());
    }
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

template <Submatrix kSubmatrix>
static void PMVUpdateBlockDiagonal(benchmark::State& state,
                                   BALData* data,
                                   ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);
  auto* block_diagonal = (kSubmatrix == Submatrix::kE)
                             ? data->BlockDiagonalEtE(options)
                             : data->BlockDiagonalFtF(options);

  for (auto _ : state) {
    if constexpr (kSubmatrix == Submatrix::kE) {
      jacobian->UpdateBlockDiagonalEtE(block_diagonal);
    } else {
      jacobian->UpdateBlockDiagonalFtF(block_diagonal);
    }
  }
}

template <bool kUseDiagonal>
static void ISCRightMultiply(benchmark::State& state,
                             BALData* data,
                             ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  const auto* jacobian =
      kUseDiagonal ? data->ImplicitSchurComplementWithDiagonal(options)
                   : data->ImplicitSchurComplementWithoutDiagonal(options);

  Vector y = Vector::Zero(jacobian->num_rows());
  Vector x = Vector::Random(jacobian->num_cols());
  for (auto _ : state) {
    jacobian->RightMultiplyAndAccumulate(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void JacobianToCRS(benchmark::State& state,
                          BALData* data,
                          ContextImpl* context) {
  auto jacobian = data->BlockSparseJacobian(context);

  std::unique_ptr<CompressedRowSparseMatrix> matrix;
  for (auto _ : state) {
    matrix = jacobian->ToCompressedRowSparseMatrix();
  }
  CHECK(matrix != nullptr);
}

#ifndef CERES_NO_CUDA
template <Submatrix kSubmatrix, MultiplyDirection kDirection>
static void PMVMultiplyCuda(benchmark::State& state,
                            BALData* data,
                            ContextImpl* context) {
  LinearSolver::Options options;
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  options.num_threads = 1;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);
  auto underlying_matrix = data->BlockSparseJacobianPartitioned(context);
  CudaPartitionedBlockSparseCRSView view(
      *underlying_matrix, jacobian->num_col_blocks_e(), context);

  const int num_cols = (kSubmatrix == Submatrix::kF) ? jacobian->num_cols_f()
                                                     : jacobian->num_cols_e();
  const int num_rows = jacobian->num_rows();
  Vector x = Vector::Random(
      (kDirection == MultiplyDirection::kRight) ? num_cols : num_rows);
  CudaVector cuda_x(context, x.size());
  CudaVector cuda_y(
      context, (kDirection == MultiplyDirection::kRight) ? num_rows : num_cols);

  cuda_x.CopyFromCpu(x);
  cuda_y.SetZero();

  const auto* matrix =
      (kSubmatrix == Submatrix::kF) ? view.matrix_f() : view.matrix_e();
  for (auto _ : state) {
    if constexpr (kDirection == MultiplyDirection::kRight) {
      matrix->RightMultiplyAndAccumulate(cuda_x, &cuda_y);
    } else {
      matrix->LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
    }
  }
  CHECK_GT(cuda_y.Norm(), 0.);
}

// We want CudaBlockSparseCRSView to be not slower than explicit conversion to
// CRS on CPU.
static void JacobianToCRSView(benchmark::State& state,
                              BALData* data,
                              ContextImpl* context) {
  auto jacobian = data->BlockSparseJacobian(context);

  std::unique_ptr<CudaBlockSparseCRSView> matrix;
  for (auto _ : state) {
    matrix = std::make_unique<CudaBlockSparseCRSView>(*jacobian, context);
  }
  CHECK(matrix != nullptr);
}

static void JacobianToCRSMatrix(benchmark::State& state,
                                BALData* data,
                                ContextImpl* context) {
  auto jacobian = data->BlockSparseJacobian(context);

  std::unique_ptr<CudaSparseMatrix> matrix;
  std::unique_ptr<CompressedRowSparseMatrix> matrix_cpu;
  for (auto _ : state) {
    matrix_cpu = jacobian->ToCompressedRowSparseMatrix();
    matrix = std::make_unique<CudaSparseMatrix>(context, *matrix_cpu);
  }
  CHECK(matrix != nullptr);
}

// Updating values in CudaBlockSparseCRSView should be +- as fast as just
// copying values (time spent in value permutation has to be hidden by PCIe
// transfer).
static void JacobianToCRSViewUpdate(benchmark::State& state,
                                    BALData* data,
                                    ContextImpl* context) {
  auto jacobian = data->BlockSparseJacobian(context);

  auto matrix = CudaBlockSparseCRSView(*jacobian, context);
  for (auto _ : state) {
    matrix.UpdateValues(*jacobian);
  }
}

static void JacobianToCRSMatrixUpdate(benchmark::State& state,
                                      BALData* data,
                                      ContextImpl* context) {
  auto jacobian = data->BlockSparseJacobian(context);

  auto matrix_cpu = jacobian->ToCompressedRowSparseMatrix();
  auto matrix = std::make_unique<CudaSparseMatrix>(context, *matrix_cpu);
  for (auto _ : state) {
    CHECK_EQ(cudaSuccess,
             cudaMemcpy(matrix->mutable_values(),
                        matrix_cpu->values(),
                        matrix->num_nonzeros() * sizeof(double),
                        cudaMemcpyHostToDevice));
  }
}
#endif

static void JacobianSquaredColumnNorm(benchmark::State& state,
                                      BALData* data,
                                      ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));
  auto jacobian = data->BlockSparseJacobian(context);
  Vector x = Vector::Zero(jacobian->num_cols());

  for (auto _ : state) {
    jacobian->SquaredColumnNorm(x.data(), context, num_threads);
  }
  CHECK_GT(x.squaredNorm(), 0.);
}

static void JacobianScaleColumns(benchmark::State& state,
                                 BALData* data,
                                 ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));
  auto jacobian_const = data->BlockSparseJacobian(context);
  auto jacobian = const_cast<BlockSparseMatrix*>(jacobian_const);
  Vector x = Vector::Ones(jacobian->num_cols());

  for (auto _ : state) {
    jacobian->ScaleColumns(x.data(), context, num_threads);
  }
}

template <MultiplyDirection kDirection>
static void JacobianMultiply(benchmark::State& state,
                             BALData* data,
                             ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));
  auto jacobian = data->BlockSparseJacobian(context);

  const int num_rows = jacobian->num_rows();
  const int num_cols = jacobian->num_cols();
  Vector y = Vector::Zero(
      (kDirection == MultiplyDirection::kRight) ? num_rows : num_cols);
  Vector x = Vector::Random(
      (kDirection == MultiplyDirection::kRight) ? num_cols : num_rows);

  for (auto _ : state) {
    if constexpr (kDirection == MultiplyDirection::kRight) {
      jacobian->RightMultiplyAndAccumulate(
          x.data(), y.data(), context, num_threads);
    } else {
      jacobian->LeftMultiplyAndAccumulate(
          x.data(), y.data(), context, num_threads);
    }
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

#ifndef CERES_NO_CUDA
template <MultiplyDirection kDirection>
static void JacobianMultiplyCuda(benchmark::State& state,
                                 BALData* data,
                                 ContextImpl* context) {
  auto crs_jacobian = data->CompressedRowSparseJacobian(context);
  CudaSparseMatrix cuda_jacobian(context, *crs_jacobian);
  CudaVector cuda_x(context, 0);
  CudaVector cuda_y(context, 0);

  const int num_rows = crs_jacobian->num_rows();
  const int num_cols = crs_jacobian->num_cols();
  Vector x((kDirection == MultiplyDirection::kRight) ? num_cols : num_rows);
  Vector y((kDirection == MultiplyDirection::kRight) ? num_rows : num_cols);
  x.setRandom();
  y.setRandom();

  cuda_x.CopyFromCpu(x);
  cuda_y.CopyFromCpu(y);
  double sum = 0;
  for (auto _ : state) {
    if constexpr (kDirection == MultiplyDirection::kRight) {
      cuda_jacobian.RightMultiplyAndAccumulate(cuda_x, &cuda_y);
    } else {
      cuda_jacobian.LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
    }
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

  if (argc == 1) {
    LOG(FATAL) << "No input datasets specified. Usage: " << argv[0]
               << " [benchmark flags] path_to_BAL_data_1.txt ... "
                  "path_to_BAL_data_N.txt";
    return -1;
  }

  std::vector<std::string> paths;
  for (int i = 1; i < argc; ++i) {
    paths.emplace_back(argv[i]);
  }

  ceres::internal::ContextImpl context;
  context.EnsureMinimumThreads(16);
#ifndef CERES_NO_CUDA
  std::string message;
  context.InitCuda(&message);
#endif

  using ceres::internal::ColmapOpenCVReprojectionError;
  using ceres::internal::EvaluationOutputs;
  using ceres::internal::JacobianFormat;
  using ceres::internal::LibmvBrownReprojectionError;
  using ceres::internal::MultiplyDirection;
  using ceres::internal::SnavelyReprojectionError;
  using ceres::internal::Submatrix;

  std::vector<std::unique_ptr<ceres::internal::BALData>> benchmark_data;
  for (const std::string& path : paths) {
    benchmark_data.emplace_back(
        std::make_unique<ceres::internal::BALData>(path));
    auto* data = benchmark_data.back().get();

    auto register_threaded = [&](const std::string& prefix, auto fn) {
      const std::string name = prefix + "<" + path + ">";
      ::benchmark::RegisterBenchmark(name.c_str(), fn, data, &context)
          ->Arg(1)
          ->Arg(2)
          ->Arg(4)
          ->Arg(8)
          ->Arg(16);
    };
    auto register_unthreaded = [&](const std::string& prefix, auto fn) {
      const std::string name = prefix + "<" + path + ">";
      return ::benchmark::RegisterBenchmark(name.c_str(), fn, data, &context);
    };
    auto register_evaluator_suite = [&](const std::string& prefix,
                                        auto functor) {
      using Functor = decltype(functor);
      register_threaded(
          prefix + "Residuals",
          ceres::internal::EvaluateProgram<Functor,
                                           EvaluationOutputs::kResidualsOnly,
                                           JacobianFormat::kNone>);
      register_threaded(
          prefix + "ResidualsAndJacobian",
          ceres::internal::EvaluateProgram<
              Functor,
              EvaluationOutputs::kResidualsAndJacobian,
              JacobianFormat::kBlockSparse>);
      register_threaded(
          prefix + "ResidualsAndJacobianCRS",
          ceres::internal::EvaluateProgram<
              Functor,
              EvaluationOutputs::kResidualsAndJacobian,
              JacobianFormat::kCompressedRow>);
      register_threaded(
          prefix + "ResidualsGradientAndJacobian",
          ceres::internal::EvaluateProgram<
              Functor,
              EvaluationOutputs::kResidualsGradientAndJacobian,
              JacobianFormat::kBlockSparse>);
    };

    // 1. Whole-problem ProgramEvaluator::Evaluate benchmarks across problem
    //    formulations (2-block Snavely Bundler, 3-block COLMAP OpenCV with
    //    EigenQuaternionManifold and HuberLoss, and 3-block libmv Brown-Conrady
    //    with SubsetManifold), outputs, and sparse Jacobian formats.
    register_evaluator_suite("", SnavelyReprojectionError{});
    register_evaluator_suite("Colmap", ColmapOpenCVReprojectionError{});
    register_evaluator_suite("Libmv", LibmvBrownReprojectionError{});
    register_threaded("Plus", ceres::internal::Plus);

    // 2. Sparse matrix-vector product, PartitionedMatrixView, and Schur
    //    complement preconditioner benchmarks on the BAL Jacobian.
    register_threaded(
        "JacobianRightMultiplyAndAccumulate",
        ceres::internal::JacobianMultiply<MultiplyDirection::kRight>);
    register_threaded("PMVRightMultiplyAndAccumulateF",
                      ceres::internal::PMVMultiply<Submatrix::kF,
                                                   MultiplyDirection::kRight>);
#ifndef CERES_NO_CUDA
    register_unthreaded(
        "PMVRightMultiplyAndAccumulateFCuda",
        ceres::internal::PMVMultiplyCuda<Submatrix::kF,
                                         MultiplyDirection::kRight>);
#endif
    register_threaded("PMVRightMultiplyAndAccumulateE",
                      ceres::internal::PMVMultiply<Submatrix::kE,
                                                   MultiplyDirection::kRight>);
#ifndef CERES_NO_CUDA
    register_unthreaded(
        "PMVRightMultiplyAndAccumulateECuda",
        ceres::internal::PMVMultiplyCuda<Submatrix::kE,
                                         MultiplyDirection::kRight>);
#endif
    register_threaded("PMVUpdateBlockDiagonalFtF",
                      ceres::internal::PMVUpdateBlockDiagonal<Submatrix::kF>);
    register_threaded("PSEPreconditionerRightMultiplyAndAccumulate",
                      ceres::internal::PSEPreconditioner);
    register_threaded("ISCRightMultiplyAndAccumulate",
                      ceres::internal::ISCRightMultiply<false>);
    register_threaded("PMVUpdateBlockDiagonalEtE",
                      ceres::internal::PMVUpdateBlockDiagonal<Submatrix::kE>);
    register_threaded("ISCRightMultiplyAndAccumulateDiag",
                      ceres::internal::ISCRightMultiply<true>);
#ifndef CERES_NO_CUDA
    register_unthreaded(
        "JacobianRightMultiplyAndAccumulateCuda",
        ceres::internal::JacobianMultiplyCuda<MultiplyDirection::kRight>)
        ->Arg(1);
#endif
    register_threaded(
        "JacobianLeftMultiplyAndAccumulate",
        ceres::internal::JacobianMultiply<MultiplyDirection::kLeft>);
    register_threaded(
        "PMVLeftMultiplyAndAccumulateF",
        ceres::internal::PMVMultiply<Submatrix::kF, MultiplyDirection::kLeft>);
#ifndef CERES_NO_CUDA
    register_unthreaded(
        "PMVLeftMultiplyAndAccumulateFCuda",
        ceres::internal::PMVMultiplyCuda<Submatrix::kF,
                                         MultiplyDirection::kLeft>);
#endif
    register_threaded(
        "PMVLeftMultiplyAndAccumulateE",
        ceres::internal::PMVMultiply<Submatrix::kE, MultiplyDirection::kLeft>);
#ifndef CERES_NO_CUDA
    register_unthreaded(
        "PMVLeftMultiplyAndAccumulateECuda",
        ceres::internal::PMVMultiplyCuda<Submatrix::kE,
                                         MultiplyDirection::kLeft>);
    register_unthreaded(
        "JacobianLeftMultiplyAndAccumulateCuda",
        ceres::internal::JacobianMultiplyCuda<MultiplyDirection::kLeft>)
        ->Arg(1);
#endif
    register_threaded("JacobianSquaredColumnNorm",
                      ceres::internal::JacobianSquaredColumnNorm);
    register_threaded("JacobianScaleColumns",
                      ceres::internal::JacobianScaleColumns);
    register_unthreaded("JacobianToCRS", ceres::internal::JacobianToCRS);
#ifndef CERES_NO_CUDA
    register_unthreaded("JacobianToCRSView",
                        ceres::internal::JacobianToCRSView);
    register_unthreaded("JacobianToCRSMatrix",
                        ceres::internal::JacobianToCRSMatrix);
    register_unthreaded("JacobianToCRSViewUpdate",
                        ceres::internal::JacobianToCRSViewUpdate);
    register_unthreaded("JacobianToCRSMatrixUpdate",
                        ceres::internal::JacobianToCRSMatrixUpdate);
#endif
  }
  ::benchmark::RunSpecifiedBenchmarks();

  using namespace ::benchmark;
  using namespace benchmark_shutdown_fallback;
  Shutdown();
  return 0;
}
