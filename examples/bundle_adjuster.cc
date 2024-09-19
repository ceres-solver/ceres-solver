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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// An example of solving a dynamically sized problem with various
// solvers and loss functions.
//
// For a simpler bare bones example of doing bundle adjustment with
// Ceres, please see simple_bundle_adjuster.cc.
//
// NOTE: This example will not compile without gflags and SuiteSparse.
//
// The problem being solved here is known as a Bundle Adjustment
// problem in computer vision. Given a set of 3d points X_1, ..., X_n,
// a set of cameras P_1, ..., P_m. If the point X_i is visible in
// image j, then there is a 2D observation u_ij that is the expected
// projection of X_i using P_j. The aim of this optimization is to
// find values of X_i and P_j such that the reprojection error
//
//    E(X,P) =  sum_ij  |u_ij - P_j X_i|^2
//
// is minimized.
//
// The problem used here comes from a collection of bundle adjustment
// problems published at University of Washington.
// http://grail.cs.washington.edu/projects/bal

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "bal_problem.h"
#include "ceres/ceres.h"
#include "snavely_reprojection_error.h"

// clang-format makes the gflags definitions too verbose
// clang-format off

ABSL_FLAG(std::string, input, "", "Input File name");
ABSL_FLAG(std::string, trust_region_strategy, "levenberg_marquardt",
              "Options are: levenberg_marquardt, dogleg.");
ABSL_FLAG(std::string, dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
              "subspace_dogleg.");

ABSL_FLAG(bool, inner_iterations, false, "Use inner iterations to non-linearly "
            "refine each successful trust region step.");

ABSL_FLAG(std::string, blocks_for_inner_iterations, "automatic", "Options are: "
              "automatic, cameras, points, cameras,points, points,cameras");

ABSL_FLAG(std::string, linear_solver, "sparse_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, "
              "sparse_normal_cholesky, dense_qr, dense_normal_cholesky, "
              "and cgnr.");
ABSL_FLAG(bool, explicit_schur_complement, false, "If using ITERATIVE_SCHUR "
            "then explicitly compute the Schur complement.");
ABSL_FLAG(std::string, preconditioner, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, schur_power_series_expansion, cluster_jacobi, "
              "cluster_tridiagonal.");
ABSL_FLAG(std::string, visibility_clustering, "canonical_views",
              "single_linkage, canonical_views");
ABSL_FLAG(bool, use_spse_initialization, false,
            "Use power series expansion to initialize the solution in ITERATIVE_SCHUR linear solver.");

ABSL_FLAG(std::string, sparse_linear_algebra_library, "suite_sparse",
              "Options are: suite_sparse, accelerate_sparse, eigen_sparse and cuda_sparse");
ABSL_FLAG(std::string, dense_linear_algebra_library, "eigen",
              "Options are: eigen, lapack, and cuda");
ABSL_FLAG(std::string, ordering_type, "amd", "Options are: amd, nesdis");
ABSL_FLAG(std::string, linear_solver_ordering, "user",
              "Options are: automatic and user");

ABSL_FLAG(bool, use_quaternions, false, "If true, uses quaternions to represent "
            "rotations. If false, angle axis is used.");
ABSL_FLAG(bool, use_manifolds, false, "For quaternions, use a manifold.");
ABSL_FLAG(bool, robustify, false, "Use a robust loss function.");

ABSL_FLAG(double, eta, 1e-2, "Default value for eta. Eta determines the "
              "accuracy of each linear solve of the truncated newton step. "
              "Changing this parameter can affect solve performance.");

ABSL_FLAG(int32_t, num_threads, -1, "Number of threads. -1 = std::thread::hardware_concurrency.");
ABSL_FLAG(int32_t, num_iterations, 5, "Number of iterations.");
ABSL_FLAG(int32_t, max_linear_solver_iterations, 500, "Maximum number of iterations"
            " for solution of linear system.");
ABSL_FLAG(double, spse_tolerance, 0.1,
             "Tolerance to reach during the iterations of power series expansion initialization or preconditioning.");
ABSL_FLAG(int32_t, max_num_spse_iterations, 5,
             "Maximum number of iterations for power series expansion initialization or preconditioning.");
ABSL_FLAG(double, max_solver_time, 1e32, "Maximum solve time in seconds.");
ABSL_FLAG(bool, nonmonotonic_steps, false, "Trust region algorithm can use"
            " nonmonotic steps.");

ABSL_FLAG(double, rotation_sigma, 0.0, "Standard deviation of camera rotation "
              "perturbation.");
ABSL_FLAG(double, translation_sigma, 0.0, "Standard deviation of the camera "
              "translation perturbation.");
ABSL_FLAG(double, point_sigma, 0.0, "Standard deviation of the point "
              "perturbation.");
ABSL_FLAG(int32_t, random_seed, 38401, "Random seed used to set the state "
             "of the pseudo random number generator used to generate "
             "the perturbations.");
ABSL_FLAG(bool, line_search, false, "Use a line search instead of trust region "
            "algorithm.");
ABSL_FLAG(bool, mixed_precision_solves, false, "Use mixed precision solves.");
ABSL_FLAG(int32_t, max_num_refinement_iterations, 0, "Iterative refinement iterations");
ABSL_FLAG(std::string, initial_ply, "", "Export the BAL file data as a PLY file.");
ABSL_FLAG(std::string, final_ply, "", "Export the refined BAL file data as a PLY "
              "file.");
// clang-format on

namespace ceres::examples {
namespace {

void SetLinearSolver(Solver::Options* options) {
  CHECK(StringToLinearSolverType(absl::GetFlag(FLAGS_linear_solver),
                                 &options->linear_solver_type));
  CHECK(StringToPreconditionerType(absl::GetFlag(FLAGS_preconditioner),
                                   &options->preconditioner_type));
  CHECK(StringToVisibilityClusteringType(
      absl::GetFlag(FLAGS_visibility_clustering),
      &options->visibility_clustering_type));
  CHECK(StringToSparseLinearAlgebraLibraryType(
      absl::GetFlag(FLAGS_sparse_linear_algebra_library),
      &options->sparse_linear_algebra_library_type));
  CHECK(StringToDenseLinearAlgebraLibraryType(
      absl::GetFlag(FLAGS_dense_linear_algebra_library),
      &options->dense_linear_algebra_library_type));
  CHECK(
      StringToLinearSolverOrderingType(absl::GetFlag(FLAGS_ordering_type),
                                       &options->linear_solver_ordering_type));
  options->use_explicit_schur_complement =
      absl::GetFlag(FLAGS_explicit_schur_complement);
  options->use_mixed_precision_solves =
      absl::GetFlag(FLAGS_mixed_precision_solves);
  options->max_num_refinement_iterations =
      absl::GetFlag(FLAGS_max_num_refinement_iterations);
  options->max_linear_solver_iterations =
      absl::GetFlag(FLAGS_max_linear_solver_iterations);
  options->use_spse_initialization =
      absl::GetFlag(FLAGS_use_spse_initialization);
  options->spse_tolerance = absl::GetFlag(FLAGS_spse_tolerance);
  options->max_num_spse_iterations =
      absl::GetFlag(FLAGS_max_num_spse_iterations);
}

void SetOrdering(BALProblem* bal_problem, Solver::Options* options) {
  const int num_points = bal_problem->num_points();
  const int point_block_size = bal_problem->point_block_size();
  double* points = bal_problem->mutable_points();

  const int num_cameras = bal_problem->num_cameras();
  const int camera_block_size = bal_problem->camera_block_size();
  double* cameras = bal_problem->mutable_cameras();

  if (options->use_inner_iterations) {
    if (absl::GetFlag(FLAGS_blocks_for_inner_iterations) == "cameras") {
      LOG(INFO) << "Camera blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
    } else if (absl::GetFlag(FLAGS_blocks_for_inner_iterations) == "points") {
      LOG(INFO) << "Point blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (absl::GetFlag(FLAGS_blocks_for_inner_iterations) ==
               "cameras,points") {
      LOG(INFO) << "Camera followed by point blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 0);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 1);
      }
    } else if (absl::GetFlag(FLAGS_blocks_for_inner_iterations) ==
               "points,cameras") {
      LOG(INFO) << "Point followed by camera blocks for inner iterations";
      options->inner_iteration_ordering =
          std::make_shared<ParameterBlockOrdering>();
      for (int i = 0; i < num_cameras; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            cameras + camera_block_size * i, 1);
      }
      for (int i = 0; i < num_points; ++i) {
        options->inner_iteration_ordering->AddElementToGroup(
            points + point_block_size * i, 0);
      }
    } else if (absl::GetFlag(FLAGS_blocks_for_inner_iterations) ==
               "automatic") {
      LOG(INFO) << "Choosing automatic blocks for inner iterations";
    } else {
      LOG(FATAL) << "Unknown block type for inner iterations: "
                 << absl::GetFlag(FLAGS_blocks_for_inner_iterations);
    }
  }

  // Bundle adjustment problems have a sparsity structure that makes
  // them amenable to more specialized and much more efficient
  // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
  // ITERATIVE_SCHUR solvers make use of this specialized
  // structure.
  //
  // This can either be done by specifying a
  // Options::linear_solver_ordering or having Ceres figure it out
  // automatically using a greedy maximum independent set algorithm.
  if (absl::GetFlag(FLAGS_linear_solver_ordering) == "user") {
    auto* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras.
    for (int i = 0; i < num_points; ++i) {
      ordering->AddElementToGroup(points + point_block_size * i, 0);
    }

    for (int i = 0; i < num_cameras; ++i) {
      // When using axis-angle, there is a single parameter block for
      // the entire camera.
      ordering->AddElementToGroup(cameras + camera_block_size * i, 1);
    }

    options->linear_solver_ordering.reset(ordering);
  }
}

void SetMinimizerOptions(Solver::Options* options) {
  options->max_num_iterations = absl::GetFlag(FLAGS_num_iterations);
  options->minimizer_progress_to_stdout = true;
  if (absl::GetFlag(FLAGS_num_threads) == -1) {
    const int num_available_threads =
        static_cast<int>(std::thread::hardware_concurrency());
    if (num_available_threads > 0) {
      options->num_threads = num_available_threads;
    }
  } else {
    options->num_threads = absl::GetFlag(FLAGS_num_threads);
  }
  CHECK_GE(options->num_threads, 1);

  options->eta = absl::GetFlag(FLAGS_eta);
  options->max_solver_time_in_seconds = absl::GetFlag(FLAGS_max_solver_time);
  options->use_nonmonotonic_steps = absl::GetFlag(FLAGS_nonmonotonic_steps);
  if (absl::GetFlag(FLAGS_line_search)) {
    options->minimizer_type = ceres::LINE_SEARCH;
  }

  CHECK(StringToTrustRegionStrategyType(
      absl::GetFlag(FLAGS_trust_region_strategy),
      &options->trust_region_strategy_type));
  CHECK(StringToDoglegType(absl::GetFlag(FLAGS_dogleg), &options->dogleg_type));
  options->use_inner_iterations = absl::GetFlag(FLAGS_inner_iterations);
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               Solver::Options* options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  SetOrdering(bal_problem, options);
}

void BuildProblem(BALProblem* bal_problem, Problem* problem) {
  const absl::Time start_time = absl::Now();
  const int point_block_size = bal_problem->point_block_size();
  const int camera_block_size = bal_problem->camera_block_size();
  double* points = bal_problem->mutable_points();
  double* cameras = bal_problem->mutable_cameras();

  // Observations is 2*num_observations long array observations =
  // [u_1, u_2, ... , u_n], where each u_i is two dimensional, the x
  // and y positions of the observation.
  const double* observations = bal_problem->observations();
  for (int i = 0; i < bal_problem->num_observations(); ++i) {
    CostFunction* cost_function;
    // Each Residual block takes a point and a camera as input and
    // outputs a 2 dimensional residual.
    cost_function = (absl::GetFlag(FLAGS_use_quaternions))
                        ? SnavelyReprojectionErrorWithQuaternions::Create(
                              observations[2 * i + 0], observations[2 * i + 1])
                        : SnavelyReprojectionError::Create(
                              observations[2 * i + 0], observations[2 * i + 1]);

    // If enabled use Huber's loss function.
    LossFunction* loss_function =
        absl::GetFlag(FLAGS_robustify) ? new HuberLoss(1.0) : nullptr;

    // Each observation corresponds to a pair of a camera and a point
    // which are identified by camera_index()[i] and point_index()[i]
    // respectively.
    double* camera =
        cameras + camera_block_size * bal_problem->camera_index()[i];
    double* point = points + point_block_size * bal_problem->point_index()[i];
    problem->AddResidualBlock(cost_function, loss_function, camera, point);
  }

  if (absl::GetFlag(FLAGS_use_quaternions) &&
      absl::GetFlag(FLAGS_use_manifolds)) {
    Manifold* camera_manifold =
        new ProductManifold<QuaternionManifold, EuclideanManifold<6>>{};
    for (int i = 0; i < bal_problem->num_cameras(); ++i) {
      problem->SetManifold(cameras + camera_block_size * i, camera_manifold);
    }
  }
  LOG(INFO) << "Time to build problem: " << absl::Now() - start_time;
}

void SolveProblem(const char* filename) {
  BALProblem bal_problem(filename, absl::GetFlag(FLAGS_use_quaternions));
  if (!absl::GetFlag(FLAGS_initial_ply).empty()) {
    bal_problem.WriteToPLYFile(absl::GetFlag(FLAGS_initial_ply));
  }

  Problem problem;

  srand(absl::GetFlag(FLAGS_random_seed));
  bal_problem.Normalize();
  bal_problem.Perturb(absl::GetFlag(FLAGS_rotation_sigma),
                      absl::GetFlag(FLAGS_translation_sigma),
                      absl::GetFlag(FLAGS_point_sigma));

  BuildProblem(&bal_problem, &problem);
  Solver::Options options;
  SetSolverOptionsFromFlags(&bal_problem, &options);
  options.gradient_tolerance = 1e-16;
  options.function_tolerance = 1e-16;
  options.parameter_tolerance = 1e-16;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  if (!absl::GetFlag(FLAGS_final_ply).empty()) {
    bal_problem.WriteToPLYFile(absl::GetFlag(FLAGS_final_ply));
  }
}

}  // namespace
}  // namespace ceres::examples

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  if (absl::GetFlag(FLAGS_input).empty()) {
    LOG(ERROR) << "Usage: bundle_adjuster --input=bal_problem";
    return 1;
  }

  CHECK(absl::GetFlag(FLAGS_use_quaternions) ||
        !absl::GetFlag(FLAGS_use_manifolds))
      << "--use_manifolds can only be used with --use_quaternions.";
  ceres::examples::SolveProblem(absl::GetFlag(FLAGS_input).c_str());
  return 0;
}
