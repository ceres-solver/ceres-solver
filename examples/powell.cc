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
// An example program that minimizes Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.
//
// From: Testing Unconstrained Optimization Software by Jorge J. More, Burton S.
// Garbow and Kenneth E. Hillstrom in ACM Transactions on Mathematical Software,
// Vol 7(1), March 1981.

#include <vector>
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct F1 {
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x2,
                                        T* residual) const {
    // f1 = x1 + 10 * x2;
    residual[0] = x1[0] + T(10.0) * x2[0];
    return true;
  }
};

struct F2 {
  template <typename T> bool operator()(const T* const x3,
                                        const T* const x4,
                                        T* residual) const {
    // f2 = sqrt(5) (x3 - x4)
    residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3 {
  template <typename T> bool operator()(const T* const x2,
                                        const T* const x4,
                                        T* residual) const {
    // f3 = (x2 - 2 x3)^2
    residual[0] = (x2[0] - T(2.0) * x4[0]) * (x2[0] - T(2.0) * x4[0]);
    return true;
  }
};

struct F4 {
  template <typename T> bool operator()(const T* const x1,
                                        const T* const x4,
                                        T* residual) const {
    // f4 = sqrt(10) (x1 - x4)^2
    residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

DEFINE_string(minimizer_type, "trust_region",
              "Minimizer type to use, choices are: line_search & trust_region");
DEFINE_string(line_search_direction_type, "lbfgs",
              "Line search direction algorithm to use, choices: lbfgs, bfgs");
DEFINE_string(line_search_type, "armijo",
              "Line search algorithm to use, choices are: armijo and wolfe.");
DEFINE_string(line_search_interpolation_type, "cubic",
              "Degree of polynomial aproximation in line search, "
              "choices are: bisection, quadratic & cubic.");
DEFINE_double(sufficient_decrease, 1.0e-4,
              "Line search Armijo sufficient (function) decrease factor.");
DEFINE_double(sufficient_curvature_decrease, 0.9,
              "Line search Wolfe sufficient curvature decrease factor.");
DEFINE_int32(lbfgs_rank, 20,
             "Rank of L-BFGS inverse Hessian approximation in line search.");
DEFINE_bool(approximate_eigenvalue_bfgs_scaling, true,
            "Use approximate eigenvalue scaling in (L)BFGS line search.");
DEFINE_double(line_search_min_step_size, 1e-3,
              "Minimum step size bound in any parameter for line search.");
DEFINE_int32(max_iterations, 100,
             "Maximum number of solver iterations allowed.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  double x1 =  3.0;
  double x2 = -1.0;
  double x3 =  0.0;
  double x4 =  1.0;

  Problem problem;
  // Add residual terms to the problem using the using the autodiff
  // wrapper to get the derivatives automatically. The parameters, x1 through
  // x4, are modified in place.
  problem.AddResidualBlock(new AutoDiffCostFunction<F1, 1, 1, 1>(new F1),
                           NULL,
                           &x1, &x2);
  problem.AddResidualBlock(new AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                           NULL,
                           &x3, &x4);
  problem.AddResidualBlock(new AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                           NULL,
                           &x2, &x3);
  problem.AddResidualBlock(new AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                           NULL,
                           &x1, &x4);

  Solver::Options options;
  LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer_type,
                                              &options.minimizer_type))
      << "Invalid minimizer_type: " << FLAGS_minimizer_type
      << ", valid options are: TRUST_REGION and LINE_SEARCH.";

  // Line search specific options (ignored if minimizer_type is TRUST_REGION).
  LOG_IF(FATAL, !ceres::StringToLineSearchDirectionType(
      FLAGS_line_search_direction_type, &options.line_search_direction_type))
      << "Invalid line_search_direction_type: "
      << FLAGS_line_search_direction_type << ", valid options are: "
      << "lbfgs & bfgs.";
  LOG_IF(FATAL, !ceres::StringToLineSearchType(FLAGS_line_search_type,
                                               &options.line_search_type))
      << "Invalid line_search_type: " << FLAGS_line_search_type
      << ", valid options are: armijo and wolfe.";
  LOG_IF(FATAL, !ceres::StringToLineSearchInterpolationType(
      FLAGS_line_search_interpolation_type,
      &options.line_search_interpolation_type))
      << "Invalid line_search_interpolation_type: "
      << FLAGS_line_search_interpolation_type
      << ", valid options are: bisection, quadratic & cubic.";
  options.max_lbfgs_rank = FLAGS_lbfgs_rank;
  options.line_search_sufficient_function_decrease = FLAGS_sufficient_decrease;
  options.line_search_sufficient_curvature_decrease =
      FLAGS_sufficient_curvature_decrease;
  options.min_line_search_step_size = FLAGS_line_search_min_step_size;
  options.max_lbfgs_rank = FLAGS_lbfgs_rank;
  options.use_approximate_eigenvalue_bfgs_scaling =
      FLAGS_approximate_eigenvalue_bfgs_scaling;

  options.max_num_iterations = FLAGS_max_iterations;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  std::cout << "Initial x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";

  // Run the solver!
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "Final x1 = " << x1
            << ", x2 = " << x2
            << ", x3 = " << x3
            << ", x4 = " << x4
            << "\n";
  return 0;
}
