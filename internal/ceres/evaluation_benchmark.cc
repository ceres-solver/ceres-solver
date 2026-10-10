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

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "benchmark/benchmark.h"
#include "ceres/benchmark_cost_functions.h"
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
#include "ceres/product_manifold.h"
#include "ceres/program.h"
#include "ceres/program_evaluator.h"
#include "ceres/scratch_evaluate_preparer.h"
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

  std::unique_ptr<BlockSparseMatrix> CreateBlockSparseJacobian(
      ContextImpl* context, bool sequential) {
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

  std::unique_ptr<CompressedRowSparseMatrix> CreateCompressedRowSparseJacobian(
      ContextImpl* context) {
    auto block_sparse = BlockSparseJacobian(context);
    return block_sparse->ToCompressedRowSparseMatrix();
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
      crs_jacobian = CreateCompressedRowSparseJacobian(context);
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
    auto block_sparse = BlockSparseJacobianPartitioned(options.context);
    implicit_schur_complement =
        std::make_unique<ImplicitSchurComplement>(options);
    implicit_schur_complement->Init(*block_sparse, nullptr, b.data());
    return implicit_schur_complement.get();
  }

  const ImplicitSchurComplement* ImplicitSchurComplementWithDiagonal(
      const LinearSolver::Options& options) {
    auto block_sparse = BlockSparseJacobianPartitioned(options.context);
    implicit_schur_complement_diag =
        std::make_unique<ImplicitSchurComplement>(options);
    implicit_schur_complement_diag->Init(*block_sparse, D.data(), b.data());
    return implicit_schur_complement_diag.get();
  }

  Vector parameters;
  Vector D;
  Vector b;
  std::unique_ptr<BundleAdjustmentProblem> bal_problem;
  std::unique_ptr<PreprocessedProblem> preprocessed_problem;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian_partitioned;
  std::unique_ptr<BlockSparseMatrix> block_sparse_jacobian;
  std::unique_ptr<CompressedRowSparseMatrix> crs_jacobian;
  std::unique_ptr<BlockSparseMatrix> block_diagonal_ete;
  // Holds a self-contained bundle adjustment Problem, its preprocessed
  // reduced_program, and its initial state vector for ProgramEvaluator
  // benchmarks.
  struct EvaluatorTestProblem {
    EvaluatorTestProblem() : problem(MakeProblemOptions()) {}

    static Problem::Options MakeProblemOptions() {
      Problem::Options options;
      options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
      return options;
    }

    Problem problem;
    HuberLoss huber_loss{1.0};
    std::unique_ptr<PreprocessedProblem> preprocessed_problem;
    Vector parameters;
    std::vector<double> parameter_storage;
  };

  // Builds a 3-block bundle adjustment problem from the BAL dataset using
  // COLMAP's OpenCVCameraModel reprojection functor
  // (`ColmapOpenCVReprojectionError`, <2, 3, 7, 8>):
  //   - Block 0: 3D world point [X, Y, Z] (size 3, Euclidean)
  //   - Block 1: Camera-from-world pose [qx, qy, qz, qw, tx, ty, tz] (size 7,
  //     tangent size 6 via ProductManifold<EigenQuaternionManifold,
  //     EuclideanManifold<3>>)
  //   - Block 2: Camera intrinsics [fx, fy, cx, cy, k1, k2, p1, p2] (size 8,
  //     Euclidean)
  // Each residual block also uses a HuberLoss(1.0) to exercise LossFunction
  // residual/Jacobian rescaling and ambient-to-tangent Manifold::PlusJacobian
  // multiplication inside ResidualBlock::Evaluate and ProgramEvaluator.
  EvaluatorTestProblem* ColmapOpenCVProblem() {
    if (!colmap_problem) {
      colmap_problem = std::make_unique<EvaluatorTestProblem>();
      const int num_cams = bal_problem->num_cameras();
      const int num_pts = bal_problem->num_points();
      const int num_obs = bal_problem->num_observations();
      const double* orig_cams = bal_problem->mutable_cameras();
      const double* orig_pts = bal_problem->mutable_points();
      const int* cam_idx = bal_problem->camera_index();
      const int* pt_idx = bal_problem->point_index();
      const double* obs = bal_problem->observations();

      colmap_problem->parameter_storage.resize(3 * num_pts + 15 * num_cams,
                                               0.0);
      double* pts = colmap_problem->parameter_storage.data();
      double* poses = pts + 3 * num_pts;
      double* intrinsics = poses + 7 * num_cams;

      std::copy_n(orig_pts, 3 * num_pts, pts);
      for (int i = 0; i < num_cams; ++i) {
        const double* c = orig_cams + 9 * i;
        double q_wxyz[4];
        AngleAxisToQuaternion(c, q_wxyz);
        // Convert Ceres [w, x, y, z] quaternion to Eigen [qx, qy, qz, qw].
        poses[7 * i + 0] = q_wxyz[1];
        poses[7 * i + 1] = q_wxyz[2];
        poses[7 * i + 2] = q_wxyz[3];
        poses[7 * i + 3] = q_wxyz[0];
        poses[7 * i + 4] = c[3];
        poses[7 * i + 5] = c[4];
        poses[7 * i + 6] = c[5];

        // BAL cameras look down -Z with focal length c[6] and radial
        // distortion (c[7], c[8]); negate focal length so projection matches
        // COLMAP's +Z camera convention.
        intrinsics[8 * i + 0] = -c[6];
        intrinsics[8 * i + 1] = -c[6];
        intrinsics[8 * i + 2] = 0.0;
        intrinsics[8 * i + 3] = 0.0;
        intrinsics[8 * i + 4] = c[7];
        intrinsics[8 * i + 5] = c[8];
        intrinsics[8 * i + 6] = 1e-4;
        intrinsics[8 * i + 7] = 1e-4;
      }

      for (int i = 0; i < num_obs; ++i) {
        CostFunction* cost_function =
            new AutoDiffCostFunction<ColmapOpenCVReprojectionError, 2, 3, 7, 8>(
                new ColmapOpenCVReprojectionError(obs[2 * i + 0],
                                                  obs[2 * i + 1]));
        colmap_problem->problem.AddResidualBlock(cost_function,
                                                 &colmap_problem->huber_loss,
                                                 pts + 3 * pt_idx[i],
                                                 poses + 7 * cam_idx[i],
                                                 intrinsics + 8 * cam_idx[i]);
      }

      for (int i = 0; i < num_cams; ++i) {
        colmap_problem->problem.SetManifold(
            poses + 7 * i,
            new ProductManifold<EigenQuaternionManifold,
                                EuclideanManifold<3>>());
      }

      Solver::Options options = bal_problem->options();
      options.linear_solver_type = ITERATIVE_SCHUR;
      options.linear_solver_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_pts; ++i) {
        options.linear_solver_ordering->AddElementToGroup(pts + 3 * i, 0);
      }
      for (int i = 0; i < num_cams; ++i) {
        options.linear_solver_ordering->AddElementToGroup(poses + 7 * i, 1);
        options.linear_solver_ordering->AddElementToGroup(intrinsics + 8 * i,
                                                          1);
      }

      auto preprocessor = Preprocessor::Create(MinimizerType::TRUST_REGION);
      colmap_problem->preprocessed_problem =
          std::make_unique<PreprocessedProblem>();
      CHECK(
          preprocessor->Preprocess(options,
                                   colmap_problem->problem.mutable_impl(),
                                   colmap_problem->preprocessed_problem.get()));
      auto* program =
          colmap_problem->preprocessed_problem->reduced_program.get();
      colmap_problem->parameters.resize(program->NumParameters());
      program->ParameterBlocksToStateVector(colmap_problem->parameters.data());
    }
    return colmap_problem.get();
  }

  // Builds a 3-block bundle adjustment problem from the BAL dataset using
  // Blender libmv's Brown-Conrady camera model (`LibmvBrownReprojectionError`,
  // <2, 9, 6, 3>):
  //   - Block 0: Camera intrinsics [f, px, py, k1, k2, k3, k4, p1, p2] (size 9,
  //     tangent size 5 via SubsetManifold(9, {1, 2, 5, 6}) holding principal
  //     point (px, py) and higher-order radial terms (k3, k4) constant)
  //   - Block 1: Camera extrinsics [rx, ry, rz, tx, ty, tz] (size 6, Euclidean)
  //   - Block 2: 3D world point [X, Y, Z] (size 3, Euclidean)
  // This exercises SubsetManifold Jacobian column projection and 3-block
  // angle-axis rotation evaluation across the whole problem.
  EvaluatorTestProblem* LibmvBrownProblem() {
    if (!libmv_problem) {
      libmv_problem = std::make_unique<EvaluatorTestProblem>();
      const int num_cams = bal_problem->num_cameras();
      const int num_pts = bal_problem->num_points();
      const int num_obs = bal_problem->num_observations();
      const double* orig_cams = bal_problem->mutable_cameras();
      const double* orig_pts = bal_problem->mutable_points();
      const int* cam_idx = bal_problem->camera_index();
      const int* pt_idx = bal_problem->point_index();
      const double* obs = bal_problem->observations();

      libmv_problem->parameter_storage.resize(3 * num_pts + 15 * num_cams, 0.0);
      double* pts = libmv_problem->parameter_storage.data();
      double* poses = pts + 3 * num_pts;
      double* intrinsics = poses + 6 * num_cams;

      std::copy_n(orig_pts, 3 * num_pts, pts);
      for (int i = 0; i < num_cams; ++i) {
        const double* c = orig_cams + 9 * i;
        std::copy_n(c, 6, poses + 6 * i);
        intrinsics[9 * i + 0] = -c[6];
        intrinsics[9 * i + 1] = 0.0;
        intrinsics[9 * i + 2] = 0.0;
        intrinsics[9 * i + 3] = c[7];
        intrinsics[9 * i + 4] = c[8];
        intrinsics[9 * i + 5] = 0.0;
        intrinsics[9 * i + 6] = 0.0;
        intrinsics[9 * i + 7] = 1e-4;
        intrinsics[9 * i + 8] = 1e-4;
      }

      for (int i = 0; i < num_obs; ++i) {
        CostFunction* cost_function =
            new AutoDiffCostFunction<LibmvBrownReprojectionError, 2, 9, 6, 3>(
                new LibmvBrownReprojectionError(obs[2 * i + 0],
                                                obs[2 * i + 1]));
        libmv_problem->problem.AddResidualBlock(cost_function,
                                                nullptr,
                                                intrinsics + 9 * cam_idx[i],
                                                poses + 6 * cam_idx[i],
                                                pts + 3 * pt_idx[i]);
      }

      for (int i = 0; i < num_cams; ++i) {
        libmv_problem->problem.SetManifold(intrinsics + 9 * i,
                                           new SubsetManifold(9, {1, 2, 5, 6}));
      }

      Solver::Options options = bal_problem->options();
      options.linear_solver_type = ITERATIVE_SCHUR;
      options.linear_solver_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_pts; ++i) {
        options.linear_solver_ordering->AddElementToGroup(pts + 3 * i, 0);
      }
      for (int i = 0; i < num_cams; ++i) {
        options.linear_solver_ordering->AddElementToGroup(intrinsics + 9 * i,
                                                          1);
        options.linear_solver_ordering->AddElementToGroup(poses + 6 * i, 1);
      }

      auto preprocessor = Preprocessor::Create(MinimizerType::TRUST_REGION);
      libmv_problem->preprocessed_problem =
          std::make_unique<PreprocessedProblem>();
      CHECK(
          preprocessor->Preprocess(options,
                                   libmv_problem->problem.mutable_impl(),
                                   libmv_problem->preprocessed_problem.get()));
      auto* program =
          libmv_problem->preprocessed_problem->reduced_program.get();
      libmv_problem->parameters.resize(program->NumParameters());
      program->ParameterBlocksToStateVector(libmv_problem->parameters.data());
    }
    return libmv_problem.get();
  }

  std::unique_ptr<BlockSparseMatrix> block_diagonal_ftf;
  std::unique_ptr<ImplicitSchurComplement> implicit_schur_complement;
  std::unique_ptr<ImplicitSchurComplement> implicit_schur_complement_diag;
  std::unique_ptr<EvaluatorTestProblem> colmap_problem;
  std::unique_ptr<EvaluatorTestProblem> libmv_problem;
};

// Problem formulation used by `EvaluateProgram`:
//   - kBundler:      2-block Snavely BundlerResidual (<2, 9, 3>), Euclidean
//                    parameters, no LossFunction.
//   - kColmapOpenCV: 3-block ColmapOpenCVReprojectionError (<2, 3, 7, 8>),
//                    EigenQuaternionManifold x EuclideanManifold<3> on pose,
//                    HuberLoss(1.0) on every residual block.
//   - kLibmvBrown:   3-block LibmvBrownReprojectionError (<2, 9, 6, 3>),
//                    SubsetManifold(9, {1, 2, 5, 6}) on intrinsics.
enum class ProblemVariant {
  kBundler,
  kColmapOpenCV,
  kLibmvBrown,
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

// Benchmarks whole-problem `Evaluator::Evaluate` across problem formulations,
// output combinations, sparse Jacobian storage formats, and thread counts.
template <ProblemVariant kVariant,
          EvaluationOutputs kOutputs,
          JacobianFormat kJacobianFormat>
static void EvaluateProgram(benchmark::State& state,
                            BALData* data,
                            ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));

  Program* program = nullptr;
  const double* parameters = nullptr;
  if constexpr (kVariant == ProblemVariant::kBundler) {
    CHECK(data->preprocessed_problem != nullptr);
    program = data->preprocessed_problem->reduced_program.get();
    parameters = data->parameters.data();
  } else if constexpr (kVariant == ProblemVariant::kColmapOpenCV) {
    auto* sub = data->ColmapOpenCVProblem();
    program = sub->preprocessed_problem->reduced_program.get();
    parameters = sub->parameters.data();
  } else if constexpr (kVariant == ProblemVariant::kLibmvBrown) {
    auto* sub = data->LibmvBrownProblem();
    program = sub->preprocessed_problem->reduced_program.get();
    parameters = sub->parameters.data();
  }
  CHECK(program != nullptr);

  Evaluator::Options options;
  options.num_threads = num_threads;
  options.context = context;
  if constexpr (kJacobianFormat == JacobianFormat::kBlockSparse &&
                (kOutputs == EvaluationOutputs::kResidualsGradientAndJacobian ||
                 kVariant != ProblemVariant::kBundler)) {
    options.linear_solver_type = ITERATIVE_SCHUR;
    options.num_eliminate_blocks = data->bal_problem->num_points();
  } else {
    options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
    options.num_eliminate_blocks = 0;
  }

  std::unique_ptr<Evaluator> evaluator;
  if constexpr (kJacobianFormat == JacobianFormat::kCompressedRow) {
    evaluator = std::make_unique<
        ProgramEvaluator<ScratchEvaluatePreparer, CompressedRowJacobianWriter>>(
        options, program);
  } else {
    std::string error;
    evaluator = Evaluator::Create(options, program, &error);
    CHECK(evaluator != nullptr) << error;
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
  for (auto _ : state) {
    CHECK(evaluator->Evaluate(eval_options,
                              parameters,
                              &cost,
                              residuals.data(),
                              gradient_ptr,
                              jacobian_ptr));
  }
}

static void Plus(benchmark::State& state, BALData* data, ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));

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

  Vector state_plus_delta = Vector::Zero(program->NumParameters());
  Vector delta = Vector::Random(program->NumEffectiveParameters());

  for (auto _ : state) {
    CHECK(evaluator->Plus(
        data->parameters.data(), delta.data(), state_plus_delta.data()));
  }
  CHECK_GT(state_plus_delta.squaredNorm(), 0.);
}

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

static void PMVRightMultiplyAndAccumulateF(benchmark::State& state,
                                           BALData* data,
                                           ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
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
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);

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
  options.num_threads = static_cast<int>(state.range(0));
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
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);

  Vector y = Vector::Zero(jacobian->num_cols_e());
  Vector x = Vector::Random(jacobian->num_rows());

  for (auto _ : state) {
    jacobian->LeftMultiplyAndAccumulateE(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void PMVUpdateBlockDiagonalEtE(benchmark::State& state,
                                      BALData* data,
                                      ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);
  auto block_diagonal_ete = data->BlockDiagonalEtE(options);

  for (auto _ : state) {
    jacobian->UpdateBlockDiagonalEtE(block_diagonal_ete);
  }
}

static void PMVUpdateBlockDiagonalFtF(benchmark::State& state,
                                      BALData* data,
                                      ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->PartitionedMatrixViewJacobian(options);
  auto block_diagonal_ftf = data->BlockDiagonalFtF(options);

  for (auto _ : state) {
    jacobian->UpdateBlockDiagonalFtF(block_diagonal_ftf);
  }
}

static void ISCRightMultiplyNoDiag(benchmark::State& state,
                                   BALData* data,
                                   ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;
  auto jacobian = data->ImplicitSchurComplementWithoutDiagonal(options);

  Vector y = Vector::Zero(jacobian->num_rows());
  Vector x = Vector::Random(jacobian->num_cols());
  for (auto _ : state) {
    jacobian->RightMultiplyAndAccumulate(x.data(), y.data());
  }
  CHECK_GT(y.squaredNorm(), 0.);
}

static void ISCRightMultiplyDiag(benchmark::State& state,
                                 BALData* data,
                                 ContextImpl* context) {
  LinearSolver::Options options;
  options.num_threads = static_cast<int>(state.range(0));
  options.elimination_groups.push_back(data->bal_problem->num_points());
  options.context = context;

  auto jacobian = data->ImplicitSchurComplementWithDiagonal(options);

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
static void PMVRightMultiplyAndAccumulateFCuda(benchmark::State& state,
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

  Vector x = Vector::Random(jacobian->num_cols_f());
  CudaVector cuda_x(context, x.size());
  CudaVector cuda_y(context, jacobian->num_rows());

  cuda_x.CopyFromCpu(x);
  cuda_y.SetZero();

  auto matrix = view.matrix_f();
  for (auto _ : state) {
    matrix->RightMultiplyAndAccumulate(cuda_x, &cuda_y);
  }
  CHECK_GT(cuda_y.Norm(), 0.);
}

static void PMVLeftMultiplyAndAccumulateFCuda(benchmark::State& state,
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

  Vector x = Vector::Random(jacobian->num_rows());
  CudaVector cuda_x(context, x.size());
  CudaVector cuda_y(context, jacobian->num_cols_f());

  cuda_x.CopyFromCpu(x);
  cuda_y.SetZero();

  auto matrix = view.matrix_f();
  for (auto _ : state) {
    matrix->LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
  }
  CHECK_GT(cuda_y.Norm(), 0.);
}

static void PMVRightMultiplyAndAccumulateECuda(benchmark::State& state,
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

  Vector x = Vector::Random(jacobian->num_cols_e());
  CudaVector cuda_x(context, x.size());
  CudaVector cuda_y(context, jacobian->num_rows());

  cuda_x.CopyFromCpu(x);
  cuda_y.SetZero();

  auto matrix = view.matrix_e();
  for (auto _ : state) {
    matrix->RightMultiplyAndAccumulate(cuda_x, &cuda_y);
  }
  CHECK_GT(cuda_y.Norm(), 0.);
}

static void PMVLeftMultiplyAndAccumulateECuda(benchmark::State& state,
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

  Vector x = Vector::Random(jacobian->num_rows());
  CudaVector cuda_x(context, x.size());
  CudaVector cuda_y(context, jacobian->num_cols_e());

  cuda_x.CopyFromCpu(x);
  cuda_y.SetZero();

  auto matrix = view.matrix_e();
  for (auto _ : state) {
    matrix->LeftMultiplyAndAccumulate(cuda_x, &cuda_y);
  }
  CHECK_GT(cuda_y.Norm(), 0.);
}

// We want CudaBlockSparseCRSView to be not slower than explicit conversion to
// CRS on CPU
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
// transfer)
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

static void JacobianRightMultiplyAndAccumulate(benchmark::State& state,
                                               BALData* data,
                                               ContextImpl* context) {
  const int num_threads = static_cast<int>(state.range(0));

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
  const int num_threads = static_cast<int>(state.range(0));

  auto jacobian = data->BlockSparseJacobian(context);

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

  using ceres::internal::EvaluationOutputs;
  using ceres::internal::JacobianFormat;
  using ceres::internal::ProblemVariant;

  for (int i = 1; i < argc; ++i) {
    const std::string path(argv[i]);
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

    // 1. Whole-problem ProgramEvaluator::Evaluate benchmarks across problem
    //    formulations (Bundler, COLMAP OpenCV with EigenQuaternionManifold and
    //    HuberLoss, and libmv Brown-Conrady with SubsetManifold), outputs, and
    //    sparse Jacobian formats (BlockSparseMatrix vs
    //    CompressedRowSparseMatrix).
    register_threaded(
        "Residuals",
        ceres::internal::EvaluateProgram<ProblemVariant::kBundler,
                                         EvaluationOutputs::kResidualsOnly,
                                         JacobianFormat::kNone>);
    register_threaded("ResidualsAndJacobian",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kBundler,
                          EvaluationOutputs::kResidualsAndJacobian,
                          JacobianFormat::kBlockSparse>);
    register_threaded("ResidualsAndJacobianCRS",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kBundler,
                          EvaluationOutputs::kResidualsAndJacobian,
                          JacobianFormat::kCompressedRow>);
    register_threaded("ResidualsGradientAndJacobian",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kBundler,
                          EvaluationOutputs::kResidualsGradientAndJacobian,
                          JacobianFormat::kBlockSparse>);
    register_threaded(
        "ColmapResiduals",
        ceres::internal::EvaluateProgram<ProblemVariant::kColmapOpenCV,
                                         EvaluationOutputs::kResidualsOnly,
                                         JacobianFormat::kNone>);
    register_threaded("ColmapResidualsAndJacobianCRS",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kColmapOpenCV,
                          EvaluationOutputs::kResidualsAndJacobian,
                          JacobianFormat::kCompressedRow>);
    register_threaded("ColmapResidualsGradientAndJacobianBlockSparse",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kColmapOpenCV,
                          EvaluationOutputs::kResidualsGradientAndJacobian,
                          JacobianFormat::kBlockSparse>);
    register_threaded("LibmvResidualsAndJacobianBlockSparse",
                      ceres::internal::EvaluateProgram<
                          ProblemVariant::kLibmvBrown,
                          EvaluationOutputs::kResidualsAndJacobian,
                          JacobianFormat::kBlockSparse>);
    register_threaded("Plus", ceres::internal::Plus);

    // 2. Sparse matrix-vector product, PartitionedMatrixView, and Schur
    //    complement preconditioner benchmarks on the BAL Jacobian.
    register_threaded("JacobianRightMultiplyAndAccumulate",
                      ceres::internal::JacobianRightMultiplyAndAccumulate);
    register_threaded("PMVRightMultiplyAndAccumulateF",
                      ceres::internal::PMVRightMultiplyAndAccumulateF);
#ifndef CERES_NO_CUDA
    register_unthreaded("PMVRightMultiplyAndAccumulateFCuda",
                        ceres::internal::PMVRightMultiplyAndAccumulateFCuda);
#endif
    register_threaded("PMVRightMultiplyAndAccumulateE",
                      ceres::internal::PMVRightMultiplyAndAccumulateE);
#ifndef CERES_NO_CUDA
    register_unthreaded("PMVRightMultiplyAndAccumulateECuda",
                        ceres::internal::PMVRightMultiplyAndAccumulateECuda);
#endif
    register_threaded("PMVUpdateBlockDiagonalFtF",
                      ceres::internal::PMVUpdateBlockDiagonalFtF);
    register_threaded("PSEPreconditionerRightMultiplyAndAccumulate",
                      ceres::internal::PSEPreconditioner);
    register_threaded("ISCRightMultiplyAndAccumulate",
                      ceres::internal::ISCRightMultiplyNoDiag);
    register_threaded("PMVUpdateBlockDiagonalEtE",
                      ceres::internal::PMVUpdateBlockDiagonalEtE);
    register_threaded("ISCRightMultiplyAndAccumulateDiag",
                      ceres::internal::ISCRightMultiplyDiag);
#ifndef CERES_NO_CUDA
    register_unthreaded("JacobianRightMultiplyAndAccumulateCuda",
                        ceres::internal::JacobianRightMultiplyAndAccumulateCuda)
        ->Arg(1);
#endif
    register_threaded("JacobianLeftMultiplyAndAccumulate",
                      ceres::internal::JacobianLeftMultiplyAndAccumulate);
    register_threaded("PMVLeftMultiplyAndAccumulateF",
                      ceres::internal::PMVLeftMultiplyAndAccumulateF);
#ifndef CERES_NO_CUDA
    register_unthreaded("PMVLeftMultiplyAndAccumulateFCuda",
                        ceres::internal::PMVLeftMultiplyAndAccumulateFCuda);
#endif
    register_threaded("PMVLeftMultiplyAndAccumulateE",
                      ceres::internal::PMVLeftMultiplyAndAccumulateE);
#ifndef CERES_NO_CUDA
    register_unthreaded("PMVLeftMultiplyAndAccumulateECuda",
                        ceres::internal::PMVLeftMultiplyAndAccumulateECuda);
    register_unthreaded("JacobianLeftMultiplyAndAccumulateCuda",
                        ceres::internal::JacobianLeftMultiplyAndAccumulateCuda)
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
