// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
#include <string>
#include <vector>

#include "bal_problem.h"
#include "ceres/ceres.h"
#include "ceres/random.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "snavely_reprojection_error.h"

DEFINE_string(input, "", "Input File name");
DEFINE_bool(use_quaternions, false, "If true, uses quaternions to represent "
            "rotations. If false, angle axis is used");
DEFINE_bool(use_local_parameterization, false, "For quaternions, use a local "
            "parameterization.");
DEFINE_bool(robustify, false, "Use a robust loss function");

DEFINE_string(trust_region_strategy, "lm", "Options are: lm, dogleg");
DEFINE_double(eta, 1e-2, "Default value for eta. Eta determines the "
             "accuracy of each linear solve of the truncated newton step. "
             "Changing this parameter can affect solve performance ");
DEFINE_string(solver_type, "sparse_schur", "Options are: "
              "sparse_schur, dense_schur, iterative_schur, cholesky, "
              "dense_qr, and conjugate_gradients");
DEFINE_string(preconditioner_type, "jacobi", "Options are: "
              "identity, jacobi, schur_jacobi, cluster_jacobi, "
              "cluster_tridiagonal");
DEFINE_string(sparse_linear_algebra_library, "suitesparse",
              "Options are: suitesparse and cxsparse");

DEFINE_string(ordering_type, "schur", "Options are: schur, user, natural");
DEFINE_bool(use_block_amd, true, "Use a block oriented fill reducing "
            "ordering.");

DEFINE_int32(num_threads, 1, "Number of threads");
DEFINE_int32(num_iterations, 5, "Number of iterations");
DEFINE_double(max_solver_time, 1e32, "Maximum solve time in seconds.");
DEFINE_bool(nonmonotonic_steps, false, "Trust region algorithm can use"
            " nonmonotic steps");

DEFINE_double(rotation_sigma, 0.0, "Standard deviation of camera rotation "
              "perturbation.");
DEFINE_double(translation_sigma, 0.0, "Standard deviation of the camera "
              "translation perturbation.");
DEFINE_double(point_sigma, 0.0, "Standard deviation of the point "
              "perturbation");
DEFINE_int32(random_seed, 38401, "Random seed used to set the state "
             "of the pseudo random number generator used to generate "
             "the pertubations.");
DEFINE_string(solver_log, "", "File to record the solver execution to.");

namespace ceres {
namespace examples {

void SetLinearSolver(Solver::Options* options) {
  if (FLAGS_solver_type == "sparse_schur") {
    options->linear_solver_type = ceres::SPARSE_SCHUR;
  } else if (FLAGS_solver_type == "dense_schur") {
    options->linear_solver_type = ceres::DENSE_SCHUR;
  } else if (FLAGS_solver_type == "iterative_schur") {
    options->linear_solver_type = ceres::ITERATIVE_SCHUR;
  } else if (FLAGS_solver_type == "cholesky") {
    options->linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else if (FLAGS_solver_type == "cgnr") {
    options->linear_solver_type = ceres::CGNR;
  } else if (FLAGS_solver_type == "dense_qr") {
    // DENSE_QR is included here for completeness, but actually using
    // this option is a bad idea due to the amount of memory needed
    // to store even the smallest of the bundle adjustment jacobian
    // arrays
    options->linear_solver_type = ceres::DENSE_QR;
  } else {
    LOG(FATAL) << "Unknown ceres solver type: "
               << FLAGS_solver_type;
  }

  if (options->linear_solver_type == ceres::CGNR) {
    options->linear_solver_min_num_iterations = 5;
    if (FLAGS_preconditioner_type == "identity") {
      options->preconditioner_type = ceres::IDENTITY;
    } else if (FLAGS_preconditioner_type == "jacobi") {
      options->preconditioner_type = ceres::JACOBI;
    } else {
      LOG(FATAL) << "For CGNR, only identity and jacobian "
                 << "preconditioners are supported. Got: "
                 << FLAGS_preconditioner_type;
    }
  }

  if (options->linear_solver_type == ceres::ITERATIVE_SCHUR) {
    options->linear_solver_min_num_iterations = 5;
    if (FLAGS_preconditioner_type == "identity") {
      options->preconditioner_type = ceres::IDENTITY;
    } else if (FLAGS_preconditioner_type == "jacobi") {
      options->preconditioner_type = ceres::JACOBI;
    } else if (FLAGS_preconditioner_type == "schur_jacobi") {
      options->preconditioner_type = ceres::SCHUR_JACOBI;
    } else if (FLAGS_preconditioner_type == "cluster_jacobi") {
      options->preconditioner_type = ceres::CLUSTER_JACOBI;
    } else if (FLAGS_preconditioner_type == "cluster_tridiagonal") {
      options->preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    } else {
      LOG(FATAL) << "Unknown ceres preconditioner type: "
                 << FLAGS_preconditioner_type;
    }
  }

  if (FLAGS_sparse_linear_algebra_library == "suitesparse") {
    options->sparse_linear_algebra_library = SUITE_SPARSE;
  } else if (FLAGS_sparse_linear_algebra_library == "cxsparse") {
    options->sparse_linear_algebra_library = CX_SPARSE;
  } else {
    LOG(FATAL) << "Unknown sparse linear algebra library type.";
  }

  options->num_linear_solver_threads = FLAGS_num_threads;
}

void SetOrdering(BALProblem* bal_problem, Solver::Options* options) {
  options->use_block_amd = FLAGS_use_block_amd;

  // Only non-Schur solvers support the natural ordering for this
  // problem.
  if (FLAGS_ordering_type == "natural") {
    if (options->linear_solver_type == SPARSE_SCHUR ||
        options->linear_solver_type == DENSE_SCHUR ||
        options->linear_solver_type == ITERATIVE_SCHUR) {
      LOG(FATAL) << "Natural ordering with Schur type solver does not work.";
    }
    return;
  }

  // Bundle adjustment problems have a sparsity structure that makes
  // them amenable to more specialized and much more efficient
  // solution strategies. The SPARSE_SCHUR, DENSE_SCHUR and
  // ITERATIVE_SCHUR solvers make use of this specialized
  // structure. Using them however requires that the ParameterBlocks
  // are in a particular order (points before cameras) and
  // Solver::Options::num_eliminate_blocks is set to the number of
  // points.
  //
  // This can either be done by specifying Options::ordering_type =
  // ceres::SCHUR, in which case Ceres will automatically determine
  // the right ParameterBlock ordering, or by manually specifying a
  // suitable ordering vector and defining
  // Options::num_eliminate_blocks.
  if (FLAGS_ordering_type == "schur") {
    options->ordering_type = ceres::SCHUR;
    return;
  }

  options->ordering_type = ceres::USER;
  const int num_points = bal_problem->num_points();
  const int point_block_size = bal_problem->point_block_size();
  double* points = bal_problem->mutable_points();
  const int num_cameras = bal_problem->num_cameras();
  const int camera_block_size = bal_problem->camera_block_size();
  double* cameras = bal_problem->mutable_cameras();

  // The points come before the cameras.
  for (int i = 0; i < num_points; ++i) {
    options->ordering.push_back(points + point_block_size * i);
  }

  for (int i = 0; i < num_cameras; ++i) {
    // When using axis-angle, there is a single parameter block for
    // the entire camera.
    options->ordering.push_back(cameras + camera_block_size * i);

    // If quaternions are used, there are two blocks, so add the
    // second block to the ordering.
    if (FLAGS_use_quaternions) {
      options->ordering.push_back(cameras + camera_block_size * i + 4);
    }
  }

  options->num_eliminate_blocks = num_points;
}

void SetMinimizerOptions(Solver::Options* options) {
  options->max_num_iterations = FLAGS_num_iterations;
  options->minimizer_progress_to_stdout = true;
  options->num_threads = FLAGS_num_threads;
  options->eta = FLAGS_eta;
  options->max_solver_time_in_seconds = FLAGS_max_solver_time;
  options->use_nonmonotonic_steps = FLAGS_nonmonotonic_steps;
  if (FLAGS_trust_region_strategy == "lm") {
    options->trust_region_strategy_type = LEVENBERG_MARQUARDT;
  } else if (FLAGS_trust_region_strategy == "dogleg") {
    options->trust_region_strategy_type = DOGLEG;
  } else {
    LOG(FATAL) << "Unknown trust region strategy: "
               << FLAGS_trust_region_strategy;
  }
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               Solver::Options* options) {
  SetMinimizerOptions(options);
  SetLinearSolver(options);
  SetOrdering(bal_problem, options);
}

// Uniform random numbers between 0 and 1.
double UniformRandom() {
  return static_cast<double>(random()) / static_cast<double>(RAND_MAX);
}

// Normal random numbers using the Box-Mueller algorithm. Its a bit
// wasteful, as it generates two but only returns one.
double RandNormal() {
  double x1, x2, w, y1, y2;
  do {
    x1 = 2.0 * UniformRandom() - 1.0;
    x2 = 2.0 * UniformRandom() - 1.0;
    w = x1 * x1 + x2 * x2;
  } while ( w >= 1.0 );

  w = sqrt((-2.0 * log(w)) / w);
  y1 = x1 * w;
  y2 = x2 * w;
  return y1;
}

void BuildProblem(BALProblem* bal_problem, Problem* problem) {
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
    if (FLAGS_use_quaternions) {
      cost_function = new AutoDiffCostFunction<
          SnavelyReprojectionErrorWithQuaternions, 2, 4, 6, 3>(
              new SnavelyReprojectionErrorWithQuaternions(
                  observations[2 * i + 0],
                  observations[2 * i + 1]));
    } else {
      cost_function =
          new AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
              new SnavelyReprojectionError(observations[2 * i + 0],
                                           observations[2 * i + 1]));
    }

    // If enabled use Huber's loss function.
    LossFunction* loss_function = FLAGS_robustify ? new HuberLoss(1.0) : NULL;

    // Each observation correponds to a pair of a camera and a point
    // which are identified by camera_index()[i] and point_index()[i]
    // respectively.
    double* camera =
        cameras + camera_block_size * bal_problem->camera_index()[i];
    double* point = points + point_block_size * bal_problem->point_index()[i];

    if (FLAGS_use_quaternions) {
      // When using quaternions, we split the camera into two
      // parameter blocks. One of size 4 for the quaternion and the
      // other of size 6 containing the translation, focal length and
      // the radial distortion parameters.
      problem->AddResidualBlock(cost_function,
                                loss_function,
                                camera,
                                camera + 4,
                                point);
    } else {
      problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }
  }

  if (FLAGS_use_quaternions && FLAGS_use_local_parameterization) {
    LocalParameterization* quaternion_parameterization =
         new QuaternionParameterization;
    for (int i = 0; i < bal_problem->num_cameras(); ++i) {
      problem->SetParameterization(cameras + camera_block_size * i,
                                   quaternion_parameterization);
    }
  }
}

void SolveProblem(const char* filename) {
  BALProblem bal_problem(filename, FLAGS_use_quaternions);
  Problem problem;

  SetRandomState(FLAGS_random_seed);
  bal_problem.Normalize();
  bal_problem.Perturb(FLAGS_rotation_sigma,
                      FLAGS_translation_sigma,
                      FLAGS_point_sigma);

  BuildProblem(&bal_problem, &problem);
  Solver::Options options;
  SetSolverOptionsFromFlags(&bal_problem, &options);
  options.solver_log = FLAGS_solver_log;
  options.gradient_tolerance *= 1e-3;

  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
}

}  // namespace examples
}  // namespace ceres

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (FLAGS_input.empty()) {
    LOG(ERROR) << "Usage: bundle_adjustment_example --input=bal_problem";
    return 1;
  }

  CHECK(FLAGS_use_quaternions || !FLAGS_use_local_parameterization)
      << "--use_local_parameterization can only be used with "
      << "--use_quaternions.";
  ceres::examples::SolveProblem(FLAGS_input.c_str());
  return 0;
}
