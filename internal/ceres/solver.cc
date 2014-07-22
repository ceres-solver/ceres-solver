// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/internal/port.h"
#include "ceres/solver.h"

#include <sstream>   // NOLINT
#include <vector>

#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/solver_impl.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "ceres/version.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace {

#define OPTION_OP(x, y, OP)                                             \
  if (!(options.x OP y)) {                                              \
    std::stringstream ss;                                               \
    ss << "Invalid configuration. ";                                    \
    ss << string("Solver::Options::" #x " = ") << options.x << ". ";    \
    ss << "Violated constraint: ";                                      \
    ss << string("Solver::Options::" #x " " #OP " "#y);                 \
    *error = ss.str();                                                  \
    return false;                                                       \
  }

#define OPTION_OP_OPTION(x, y, OP)                                      \
  if (!(options.x OP options.y)) {                                      \
    std::stringstream ss;                                               \
    ss << "Invalid configuration. ";                                    \
    ss << string("Solver::Options::" #x " = ") << options.x << ". ";    \
    ss << string("Solver::Options::" #y " = ") << options.y << ". ";    \
    ss << "Violated constraint: ";                                      \
    ss << string("Solver::Options::" #x );                              \
    ss << string(#OP " Solver::Options::" #y ".");                      \
    *error = ss.str();                                                  \
    return false;                                                       \
  }

#define OPTION_GE(x, y) OPTION_OP(x, y, >=);
#define OPTION_GT(x, y) OPTION_OP(x, y, >);
#define OPTION_LE(x, y) OPTION_OP(x, y, <=);
#define OPTION_LT(x, y) OPTION_OP(x, y, <);
#define OPTION_LE_OPTION(x, y) OPTION_OP_OPTION(x ,y, <=)
#define OPTION_LT_OPTION(x, y) OPTION_OP_OPTION(x ,y, <)

bool CommonOptionsAreValid(const Solver::Options& options, string* error) {
  OPTION_GE(max_num_iterations, 0);
  OPTION_GE(max_solver_time_in_seconds, 0.0);
  OPTION_GE(function_tolerance, 0.0);
  OPTION_GE(gradient_tolerance, 0.0);
  OPTION_GE(parameter_tolerance, 0.0);
  OPTION_GT(num_threads, 0);
  OPTION_GT(num_linear_solver_threads, 0);
  if (options.check_gradients) {
    OPTION_GT(gradient_check_relative_precision, 0.0);
    OPTION_GT(numeric_derivative_relative_step_size, 0.0);
  }
  return true;
}

bool TrustRegionOptionsAreValid(const Solver::Options& options, string* error) {
  OPTION_GT(initial_trust_region_radius, 0.0);
  OPTION_GT(min_trust_region_radius, 0.0);
  OPTION_GT(max_trust_region_radius, 0.0);
  OPTION_LE_OPTION(min_trust_region_radius, max_trust_region_radius);
  OPTION_LE_OPTION(min_trust_region_radius, initial_trust_region_radius);
  OPTION_LE_OPTION(initial_trust_region_radius, max_trust_region_radius);
  OPTION_GE(min_relative_decrease, 0.0);
  OPTION_GE(min_lm_diagonal, 0.0);
  OPTION_GE(max_lm_diagonal, 0.0);
  OPTION_LE_OPTION(min_lm_diagonal, max_lm_diagonal);
  OPTION_GE(max_num_consecutive_invalid_steps, 0);
  OPTION_GT(eta, 0.0);
  OPTION_GE(min_linear_solver_iterations, 1);
  OPTION_GE(max_linear_solver_iterations, 1);
  OPTION_LE_OPTION(min_linear_solver_iterations, max_linear_solver_iterations);

  if (options.use_inner_iterations) {
    OPTION_GE(inner_iteration_tolerance, 0.0);
  }

  if (options.use_nonmonotonic_steps) {
    OPTION_GT(max_consecutive_nonmonotonic_steps, 0);
  }

  if (options.preconditioner_type == CLUSTER_JACOBI &&
      options.sparse_linear_algebra_library_type != SUITE_SPARSE) {
    *error =  "CLUSTER_JACOBI requires "
        "Solver::Options::sparse_linear_algebra_library_type to be "
        "SUITE_SPARSE";
    return false;
  }

  if (options.preconditioner_type == CLUSTER_TRIDIAGONAL &&
      options.sparse_linear_algebra_library_type != SUITE_SPARSE) {
    *error =  "CLUSTER_TRIDIAGONAL requires "
        "Solver::Options::sparse_linear_algebra_library_type to be "
        "SUITE_SPARSE";
    return false;
  }

#ifdef CERES_NO_LAPACK
  if (options.dense_linear_algebra_library_type == LAPACK) {
    if (options.linear_solver_type == DENSE_NORMAL_CHOLESKY) {
      *error = "Can't use DENSE_NORMAL_CHOLESKY with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }

    if (options.linear_solver_type == DENSE_QR) {
      *error = "Can't use DENSE_QR with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }

    if (options.linear_solver_type == DENSE_SCHUR) {
      *error = "Can't use DENSE_SCHUR with LAPACK because "
          "LAPACK was not enabled when Ceres was built.";
      return false;
    }
  }
#endif

#ifdef CERES_NO_SUITESPARSE
  if (options.sparse_linear_algebra_library_type == SUITE_SPARSE) {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY) {
      *error = "Can't use SPARSE_NORMAL_CHOLESKY with SUITESPARSE because "
             "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }

    if (options.linear_solver_type == SPARSE_SCHUR) {
      *error = "Can't use SPARSE_SCHUR with SUITESPARSE because "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }

    if (options.preconditioner_type == CLUSTER_JACOBI) {
      *error =  "CLUSTER_JACOBI preconditioner not supported. "
          "SuiteSparse was not enabled when Ceres was built.";
      return false;
    }

    if (options.preconditioner_type == CLUSTER_TRIDIAGONAL) {
      *error =  "CLUSTER_TRIDIAGONAL preconditioner not supported. "
          "SuiteSparse was not enabled when Ceres was built.";
    return false;
    }
  }
#endif

#ifdef CERES_NO_CXSPARSE
  if (options.sparse_linear_algebra_library_type == CX_SPARSE) {
    if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY) {
      *error = "Can't use SPARSE_NORMAL_CHOLESKY with CX_SPARSE because "
             "CXSparse was not enabled when Ceres was built.";
      return false;
    }

    if (options.linear_solver_type == SPARSE_SCHUR) {
      *error = "Can't use SPARSE_SCHUR with CX_SPARSE because "
          "CXSparse was not enabled when Ceres was built.";
      return false;
    }
  }
#endif

  if (options.trust_region_strategy_type == DOGLEG) {
    if (options.linear_solver_type == ITERATIVE_SCHUR ||
        options.linear_solver_type == CGNR) {
      *error = "DOGLEG only supports exact factorization based linear "
          "solvers. If you want to use an iterative solver please "
          "use LEVENBERG_MARQUARDT as the trust_region_strategy_type";
      return false;
    }
  }

  if (options.trust_region_minimizer_iterations_to_dump.size() > 0 &&
      options.trust_region_problem_dump_format_type != CONSOLE &&
      options.trust_region_problem_dump_directory.empty()) {
    *error = "Solver::Options::trust_region_problem_dump_directory is empty.";
    return false;
  }

  if (options.dynamic_sparsity &&
      options.linear_solver_type != SPARSE_NORMAL_CHOLESKY) {
    *error = "Dynamic sparsity is only supported with SPARSE_NORMAL_CHOLESKY.";
    return false;
  }

  return true;
}

bool LineSearchOptionsAreValid(const Solver::Options& options, string* error) {
  OPTION_GT(max_lbfgs_rank, 0);
  OPTION_GT(min_line_search_step_size, 0.0);
  OPTION_GT(max_line_search_step_contraction, 0.0);
  OPTION_LT(max_line_search_step_contraction, 1.0);
  OPTION_LT_OPTION(max_line_search_step_contraction,
                   min_line_search_step_contraction);
  OPTION_LE(min_line_search_step_contraction, 1.0);
  OPTION_GT(max_num_line_search_step_size_iterations, 0);
  OPTION_GT(line_search_sufficient_function_decrease, 0.0);
  OPTION_LT_OPTION(line_search_sufficient_function_decrease,
                   line_search_sufficient_curvature_decrease);
  OPTION_LT(line_search_sufficient_curvature_decrease, 1.0);
  OPTION_GT(max_line_search_step_expansion, 1.0);

  if ((options.line_search_direction_type == ceres::BFGS ||
       options.line_search_direction_type == ceres::LBFGS) &&
      options.line_search_type != ceres::WOLFE) {

    *error =
        string("Invalid configuration: Solver::Options::line_search_type = ")
        + string(LineSearchTypeToString(options.line_search_type))
        + string(". When using (L)BFGS, "
                 "Solver::Options::line_search_type must be set to WOLFE.");
    return false;
  }

  // Warn user if they have requested BISECTION interpolation, but constraints
  // on max/min step size change during line search prevent bisection scaling
  // from occurring. Warn only, as this is likely a user mistake, but one which
  // does not prevent us from continuing.
  LOG_IF(WARNING,
         (options.line_search_interpolation_type == ceres::BISECTION &&
          (options.max_line_search_step_contraction > 0.5 ||
           options.min_line_search_step_contraction < 0.5)))
      << "Line search interpolation type is BISECTION, but specified "
      << "max_line_search_step_contraction: "
      << options.max_line_search_step_contraction << ", and "
      << "min_line_search_step_contraction: "
      << options.min_line_search_step_contraction
      << ", prevent bisection (0.5) scaling, continuing with solve regardless.";

  return true;
}

#undef OPTION_OP
#undef OPTION_OP_OPTION
#undef OPTION_GT
#undef OPTION_GE
#undef OPTION_LE
#undef OPTION_LT
#undef OPTION_LE_OPTION
#undef OPTION_LT_OPTION

void StringifyOrdering(const vector<int>& ordering, string* report) {
  if (ordering.size() == 0) {
    internal::StringAppendF(report, "AUTOMATIC");
    return;
  }

  for (int i = 0; i < ordering.size() - 1; ++i) {
    internal::StringAppendF(report, "%d, ", ordering[i]);
  }
  internal::StringAppendF(report, "%d", ordering.back());
}

} // namespace

bool Solver::Options::IsValid(string* error) const {
  if (!CommonOptionsAreValid(*this, error)) {
    return false;
  }

  if (minimizer_type == TRUST_REGION) {
    return TrustRegionOptionsAreValid(*this, error);
  }

  CHECK_EQ(minimizer_type, LINE_SEARCH);
  return LineSearchOptionsAreValid(*this, error);
}

Solver::~Solver() {}

void Solver::Solve(const Solver::Options& options,
                   Problem* problem,
                   Solver::Summary* summary) {
  double start_time_seconds = internal::WallTimeInSeconds();
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(summary);

  *summary = Summary();
  if (!options.IsValid(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  internal::ProblemImpl* problem_impl = problem->problem_impl_.get();
  internal::SolverImpl::Solve(options, problem_impl, summary);
  summary->total_time_in_seconds =
      internal::WallTimeInSeconds() - start_time_seconds;
}

void Solve(const Solver::Options& options,
           Problem* problem,
           Solver::Summary* summary) {
  Solver solver;
  solver.Solve(options, problem, summary);
}

Solver::Summary::Summary()
    // Invalid values for most fields, to ensure that we are not
    // accidentally reporting default values.
    : minimizer_type(TRUST_REGION),
      termination_type(FAILURE),
      message("ceres::Solve was not called."),
      initial_cost(-1.0),
      final_cost(-1.0),
      fixed_cost(-1.0),
      num_successful_steps(-1),
      num_unsuccessful_steps(-1),
      num_inner_iteration_steps(-1),
      preprocessor_time_in_seconds(-1.0),
      minimizer_time_in_seconds(-1.0),
      postprocessor_time_in_seconds(-1.0),
      total_time_in_seconds(-1.0),
      linear_solver_time_in_seconds(-1.0),
      residual_evaluation_time_in_seconds(-1.0),
      jacobian_evaluation_time_in_seconds(-1.0),
      inner_iteration_time_in_seconds(-1.0),
      num_parameter_blocks(-1),
      num_parameters(-1),
      num_effective_parameters(-1),
      num_residual_blocks(-1),
      num_residuals(-1),
      num_parameter_blocks_reduced(-1),
      num_parameters_reduced(-1),
      num_effective_parameters_reduced(-1),
      num_residual_blocks_reduced(-1),
      num_residuals_reduced(-1),
      num_threads_given(-1),
      num_threads_used(-1),
      num_linear_solver_threads_given(-1),
      num_linear_solver_threads_used(-1),
      linear_solver_type_given(SPARSE_NORMAL_CHOLESKY),
      linear_solver_type_used(SPARSE_NORMAL_CHOLESKY),
      inner_iterations_given(false),
      inner_iterations_used(false),
      preconditioner_type(IDENTITY),
      visibility_clustering_type(CANONICAL_VIEWS),
      trust_region_strategy_type(LEVENBERG_MARQUARDT),
      dense_linear_algebra_library_type(EIGEN),
      sparse_linear_algebra_library_type(SUITE_SPARSE),
      line_search_direction_type(LBFGS),
      line_search_type(ARMIJO),
      line_search_interpolation_type(BISECTION),
      nonlinear_conjugate_gradient_type(FLETCHER_REEVES),
      max_lbfgs_rank(-1) {
}

using internal::StringAppendF;
using internal::StringPrintf;

string Solver::Summary::BriefReport() const {
  return StringPrintf("Ceres Solver Report: "
                      "Iterations: %d, "
                      "Initial cost: %e, "
                      "Final cost: %e, "
                      "Termination: %s",
                      num_successful_steps + num_unsuccessful_steps,
                      initial_cost,
                      final_cost,
                      TerminationTypeToString(termination_type));
};

string Solver::Summary::FullReport() const {
  string report =
      "\n"
      "Ceres Solver v" CERES_VERSION_STRING " Solve Report\n"
      "----------------------------------\n";

  StringAppendF(&report, "%45s    %21s\n", "Original", "Reduced");
  StringAppendF(&report, "Parameter blocks    % 25d% 25d\n",
                num_parameter_blocks, num_parameter_blocks_reduced);
  StringAppendF(&report, "Parameters          % 25d% 25d\n",
                num_parameters, num_parameters_reduced);
  if (num_effective_parameters_reduced != num_parameters_reduced) {
    StringAppendF(&report, "Effective parameters% 25d% 25d\n",
                  num_effective_parameters, num_effective_parameters_reduced);
  }
  StringAppendF(&report, "Residual blocks     % 25d% 25d\n",
                num_residual_blocks, num_residual_blocks_reduced);
  StringAppendF(&report, "Residual            % 25d% 25d\n",
                num_residuals, num_residuals_reduced);

  if (minimizer_type == TRUST_REGION) {
    // TRUST_SEARCH HEADER
    StringAppendF(&report, "\nMinimizer                 %19s\n",
                  "TRUST_REGION");

    if (linear_solver_type_used == DENSE_NORMAL_CHOLESKY ||
        linear_solver_type_used == DENSE_SCHUR ||
        linear_solver_type_used == DENSE_QR) {
      StringAppendF(&report, "\nDense linear algebra library  %15s\n",
                    DenseLinearAlgebraLibraryTypeToString(
                        dense_linear_algebra_library_type));
    }

    if (linear_solver_type_used == SPARSE_NORMAL_CHOLESKY ||
        linear_solver_type_used == SPARSE_SCHUR ||
        (linear_solver_type_used == ITERATIVE_SCHUR &&
         (preconditioner_type == CLUSTER_JACOBI ||
          preconditioner_type == CLUSTER_TRIDIAGONAL))) {
      StringAppendF(&report, "\nSparse linear algebra library %15s\n",
                    SparseLinearAlgebraLibraryTypeToString(
                        sparse_linear_algebra_library_type));
    }

    StringAppendF(&report, "Trust region strategy     %19s",
                  TrustRegionStrategyTypeToString(
                      trust_region_strategy_type));
    if (trust_region_strategy_type == DOGLEG) {
      if (dogleg_type == TRADITIONAL_DOGLEG) {
        StringAppendF(&report, " (TRADITIONAL)");
      } else {
        StringAppendF(&report, " (SUBSPACE)");
      }
    }
    StringAppendF(&report, "\n");
    StringAppendF(&report, "\n");

    StringAppendF(&report, "%45s    %21s\n", "Given",  "Used");
    StringAppendF(&report, "Linear solver       %25s%25s\n",
                  LinearSolverTypeToString(linear_solver_type_given),
                  LinearSolverTypeToString(linear_solver_type_used));

    if (linear_solver_type_given == CGNR ||
        linear_solver_type_given == ITERATIVE_SCHUR) {
      StringAppendF(&report, "Preconditioner      %25s%25s\n",
                    PreconditionerTypeToString(preconditioner_type),
                    PreconditionerTypeToString(preconditioner_type));
    }

    if (preconditioner_type == CLUSTER_JACOBI ||
        preconditioner_type == CLUSTER_TRIDIAGONAL) {
      StringAppendF(&report, "Visibility clustering%24s%25s\n",
                    VisibilityClusteringTypeToString(
                        visibility_clustering_type),
                    VisibilityClusteringTypeToString(
                        visibility_clustering_type));
    }
    StringAppendF(&report, "Threads             % 25d% 25d\n",
                  num_threads_given, num_threads_used);
    StringAppendF(&report, "Linear solver threads % 23d% 25d\n",
                  num_linear_solver_threads_given,
                  num_linear_solver_threads_used);

    if (IsSchurType(linear_solver_type_used)) {
      string given;
      StringifyOrdering(linear_solver_ordering_given, &given);
      string used;
      StringifyOrdering(linear_solver_ordering_used, &used);
      StringAppendF(&report,
                    "Linear solver ordering %22s %24s\n",
                    given.c_str(),
                    used.c_str());
    }

    if (inner_iterations_given) {
      StringAppendF(&report,
                    "Use inner iterations     %20s     %20s\n",
                    inner_iterations_given ? "True" : "False",
                    inner_iterations_used ? "True" : "False");
    }

    if (inner_iterations_used) {
      string given;
      StringifyOrdering(inner_iteration_ordering_given, &given);
      string used;
      StringifyOrdering(inner_iteration_ordering_used, &used);
    StringAppendF(&report,
                  "Inner iteration ordering %20s %24s\n",
                  given.c_str(),
                  used.c_str());
    }
  } else {
    // LINE_SEARCH HEADER
    StringAppendF(&report, "\nMinimizer                 %19s\n", "LINE_SEARCH");


    string line_search_direction_string;
    if (line_search_direction_type == LBFGS) {
      line_search_direction_string = StringPrintf("LBFGS (%d)", max_lbfgs_rank);
    } else if (line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT) {
      line_search_direction_string =
          NonlinearConjugateGradientTypeToString(
              nonlinear_conjugate_gradient_type);
    } else {
      line_search_direction_string =
          LineSearchDirectionTypeToString(line_search_direction_type);
    }

    StringAppendF(&report, "Line search direction     %19s\n",
                  line_search_direction_string.c_str());

    const string line_search_type_string =
        StringPrintf("%s %s",
                     LineSearchInterpolationTypeToString(
                         line_search_interpolation_type),
                     LineSearchTypeToString(line_search_type));
    StringAppendF(&report, "Line search type          %19s\n",
                  line_search_type_string.c_str());
    StringAppendF(&report, "\n");

    StringAppendF(&report, "%45s    %21s\n", "Given",  "Used");
    StringAppendF(&report, "Threads             % 25d% 25d\n",
                  num_threads_given, num_threads_used);
  }

  StringAppendF(&report, "\nCost:\n");
  StringAppendF(&report, "Initial        % 30e\n", initial_cost);
  if (termination_type != FAILURE &&
      termination_type != USER_FAILURE) {
    StringAppendF(&report, "Final          % 30e\n", final_cost);
    StringAppendF(&report, "Change         % 30e\n",
                  initial_cost - final_cost);
  }

  StringAppendF(&report, "\nMinimizer iterations         % 16d\n",
                num_successful_steps + num_unsuccessful_steps);

  // Successful/Unsuccessful steps only matter in the case of the
  // trust region solver. Line search terminates when it encounters
  // the first unsuccessful step.
  if (minimizer_type == TRUST_REGION) {
    StringAppendF(&report, "Successful steps               % 14d\n",
                  num_successful_steps);
    StringAppendF(&report, "Unsuccessful steps             % 14d\n",
                  num_unsuccessful_steps);
  }
  if (inner_iterations_used) {
    StringAppendF(&report, "Steps with inner iterations    % 14d\n",
                  num_inner_iteration_steps);
  }

  StringAppendF(&report, "\nTime (in seconds):\n");
  StringAppendF(&report, "Preprocessor        %25.3f\n",
                preprocessor_time_in_seconds);

  StringAppendF(&report, "\n  Residual evaluation %23.3f\n",
                residual_evaluation_time_in_seconds);
  StringAppendF(&report, "  Jacobian evaluation %23.3f\n",
                jacobian_evaluation_time_in_seconds);

  if (minimizer_type == TRUST_REGION) {
    StringAppendF(&report, "  Linear solver       %23.3f\n",
                  linear_solver_time_in_seconds);
  }

  if (inner_iterations_used) {
    StringAppendF(&report, "  Inner iterations    %23.3f\n",
                  inner_iteration_time_in_seconds);
  }

  StringAppendF(&report, "Minimizer           %25.3f\n\n",
                minimizer_time_in_seconds);

  StringAppendF(&report, "Postprocessor        %24.3f\n",
                postprocessor_time_in_seconds);

  StringAppendF(&report, "Total               %25.3f\n\n",
                total_time_in_seconds);

  StringAppendF(&report, "Termination:        %25s (%s)\n",
                TerminationTypeToString(termination_type), message.c_str());
  return report;
};

bool Solver::Summary::IsSolutionUsable() const {
  return (termination_type == CONVERGENCE ||
          termination_type == NO_CONVERGENCE ||
          termination_type == USER_SUCCESS);
}

}  // namespace ceres
