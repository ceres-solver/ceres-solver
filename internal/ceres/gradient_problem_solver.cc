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

#include "ceres/gradient_problem_solver.h"

#include <map>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ceres/callbacks.h"
#include "ceres/gradient_problem.h"
#include "ceres/gradient_problem_evaluator.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "ceres/map_util.h"
#include "ceres/minimizer.h"
#include "ceres/solver.h"
#include "ceres/solver_utils.h"
#include "ceres/types.h"

namespace ceres {
namespace {

Solver::Options GradientProblemSolverOptionsToSolverOptions(
    const GradientProblemSolver::Options& options) {
#define COPY_OPTION(x) solver_options.x = options.x

  Solver::Options solver_options;
  solver_options.minimizer_type = LINE_SEARCH;
  COPY_OPTION(line_search_direction_type);
  COPY_OPTION(line_search_type);
  COPY_OPTION(nonlinear_conjugate_gradient_type);
  COPY_OPTION(max_lbfgs_rank);
  COPY_OPTION(use_approximate_eigenvalue_bfgs_scaling);
  COPY_OPTION(line_search_interpolation_type);
  COPY_OPTION(min_line_search_step_size);
  COPY_OPTION(line_search_sufficient_function_decrease);
  COPY_OPTION(max_line_search_step_contraction);
  COPY_OPTION(min_line_search_step_contraction);
  COPY_OPTION(max_num_line_search_step_size_iterations);
  COPY_OPTION(max_num_line_search_direction_restarts);
  COPY_OPTION(line_search_sufficient_curvature_decrease);
  COPY_OPTION(max_line_search_step_expansion);
  COPY_OPTION(max_num_iterations);
  COPY_OPTION(max_solver_time_in_seconds);
  COPY_OPTION(parameter_tolerance);
  COPY_OPTION(function_tolerance);
  COPY_OPTION(gradient_tolerance);
  COPY_OPTION(logging_type);
  COPY_OPTION(minimizer_progress_to_stdout);
  COPY_OPTION(callbacks);
  return solver_options;
#undef COPY_OPTION
}

}  // namespace

bool GradientProblemSolver::Options::IsValid(std::string* error) const {
  const Solver::Options solver_options =
      GradientProblemSolverOptionsToSolverOptions(*this);
  return solver_options.IsValid(error);
}

GradientProblemSolver::~GradientProblemSolver() = default;

void GradientProblemSolver::Solve(const GradientProblemSolver::Options& options,
                                  const GradientProblem& problem,
                                  double* parameters_ptr,
                                  GradientProblemSolver::Summary* summary) {
  using internal::CallStatistics;
  using internal::GradientProblemEvaluator;
  using internal::GradientProblemSolverStateUpdatingCallback;
  using internal::LoggingCallback;
  using internal::Minimizer;
  using internal::SetSummaryFinalCost;

  const absl::Time start_time = absl::Now();

  CHECK(summary != nullptr);
  *summary = Summary();
  // clang-format off
  summary->num_parameters                    = problem.NumParameters();
  summary->num_tangent_parameters            = problem.NumTangentParameters();
  summary->line_search_direction_type        = options.line_search_direction_type;         //  NOLINT
  summary->line_search_interpolation_type    = options.line_search_interpolation_type;     //  NOLINT
  summary->line_search_type                  = options.line_search_type;
  summary->max_lbfgs_rank                    = options.max_lbfgs_rank;
  summary->nonlinear_conjugate_gradient_type = options.nonlinear_conjugate_gradient_type;  //  NOLINT
  // clang-format on

  // Check validity
  if (!options.IsValid(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  VectorRef parameters(parameters_ptr, problem.NumParameters());
  Vector solution(problem.NumParameters());
  solution = parameters;

  // TODO(sameeragarwal): This is a bit convoluted, we should be able
  // to convert to minimizer options directly, but this will do for
  // now.
  Minimizer::Options minimizer_options =
      Minimizer::Options(GradientProblemSolverOptionsToSolverOptions(options));
  minimizer_options.evaluator =
      std::make_unique<GradientProblemEvaluator>(problem);

  std::unique_ptr<IterationCallback> logging_callback;
  if (options.logging_type != SILENT) {
    logging_callback = std::make_unique<LoggingCallback>(
        LINE_SEARCH, options.minimizer_progress_to_stdout);
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       logging_callback.get());
  }

  std::unique_ptr<IterationCallback> state_updating_callback;
  if (options.update_state_every_iteration) {
    state_updating_callback =
        std::make_unique<GradientProblemSolverStateUpdatingCallback>(
            problem.NumParameters(), solution.data(), parameters_ptr);
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       state_updating_callback.get());
  }

  std::unique_ptr<Minimizer> minimizer(Minimizer::Create(LINE_SEARCH));

  Solver::Summary solver_summary;
  solver_summary.fixed_cost = 0.0;
  solver_summary.preprocessor_time_in_seconds = 0.0;
  solver_summary.postprocessor_time_in_seconds = 0.0;
  solver_summary.line_search_polynomial_minimization_time_in_seconds = 0.0;

  minimizer->Minimize(minimizer_options, solution.data(), &solver_summary);

  // clang-format off
  summary->termination_type = solver_summary.termination_type;
  summary->message          = solver_summary.message;
  summary->initial_cost     = solver_summary.initial_cost;
  summary->final_cost       = solver_summary.final_cost;
  summary->iterations       = solver_summary.iterations;
  // clang-format on
  summary->line_search_polynomial_minimization_time_in_seconds =
      solver_summary.line_search_polynomial_minimization_time_in_seconds;

  if (summary->IsSolutionUsable()) {
    parameters = solution;
    SetSummaryFinalCost(summary);
  }

  const std::map<std::string, CallStatistics>& evaluator_statistics =
      minimizer_options.evaluator->Statistics();
  {
    const CallStatistics& call_stats = FindWithDefault(
        evaluator_statistics, "Evaluator::Residual", CallStatistics());
    summary->cost_evaluation_time_in_seconds =
        absl::ToDoubleSeconds(call_stats.time);
    summary->num_cost_evaluations = call_stats.calls;
  }

  {
    const CallStatistics& call_stats = FindWithDefault(
        evaluator_statistics, "Evaluator::Jacobian", CallStatistics());
    summary->gradient_evaluation_time_in_seconds =
        absl::ToDoubleSeconds(call_stats.time);
    summary->num_gradient_evaluations = call_stats.calls;
  }

  summary->total_time_in_seconds =
      absl::ToDoubleSeconds(absl::Now() - start_time);
}

bool GradientProblemSolver::Summary::IsSolutionUsable() const {
  return internal::IsSolutionUsable(*this);
}

std::string GradientProblemSolver::Summary::BriefReport() const {
  return absl::StrFormat(
      "Ceres GradientProblemSolver Report: "
      "Iterations: %d, "
      "Initial cost: %e, "
      "Final cost: %e, "
      "Termination: %s",
      static_cast<int>(iterations.size()),
      initial_cost,
      final_cost,
      TerminationTypeToString(termination_type));
}

std::string GradientProblemSolver::Summary::FullReport() const {
  std::string report =
      absl::StrCat("\nSolver Summary (v ", internal::VersionString(), ")\n\n");

  absl::StrAppendFormat(&report, "Parameters          % 25d\n", num_parameters);
  if (num_tangent_parameters != num_parameters) {
    absl::StrAppendFormat(
        &report, "Tangent parameters   % 25d\n", num_tangent_parameters);
  }

  std::string line_search_direction_string;
  if (line_search_direction_type == LBFGS) {
    line_search_direction_string =
        absl::StrFormat("LBFGS (%d)", max_lbfgs_rank);
  } else if (line_search_direction_type == NONLINEAR_CONJUGATE_GRADIENT) {
    line_search_direction_string = NonlinearConjugateGradientTypeToString(
        nonlinear_conjugate_gradient_type);
  } else {
    line_search_direction_string =
        LineSearchDirectionTypeToString(line_search_direction_type);
  }

  absl::StrAppendFormat(&report,
                        "Line search direction     %19s\n",
                        line_search_direction_string);

  const std::string line_search_type_string = absl::StrFormat(
      "%s %s",
      LineSearchInterpolationTypeToString(line_search_interpolation_type),
      LineSearchTypeToString(line_search_type));
  absl::StrAppendFormat(
      &report, "Line search type          %19s\n", line_search_type_string);
  absl::StrAppendFormat(&report, "\n");

  absl::StrAppendFormat(&report, "\nCost:\n");
  absl::StrAppendFormat(&report, "Initial        % 30e\n", initial_cost);
  if (termination_type != FAILURE && termination_type != USER_FAILURE) {
    absl::StrAppendFormat(&report, "Final          % 30e\n", final_cost);
    absl::StrAppendFormat(
        &report, "Change         % 30e\n", initial_cost - final_cost);
  }

  absl::StrAppendFormat(&report,
                        "\nMinimizer iterations         % 16d\n",
                        static_cast<int>(iterations.size()));

  absl::StrAppendFormat(&report, "\nTime (in seconds):\n");
  absl::StrAppendFormat(&report,
                        "\n  Cost evaluation     %23.6f (%d)\n",
                        cost_evaluation_time_in_seconds,
                        num_cost_evaluations);
  absl::StrAppendFormat(&report,
                        "  Gradient & cost evaluation %16.6f (%d)\n",
                        gradient_evaluation_time_in_seconds,
                        num_gradient_evaluations);
  absl::StrAppendFormat(&report,
                        "  Polynomial minimization   %17.6f\n",
                        line_search_polynomial_minimization_time_in_seconds);
  absl::StrAppendFormat(
      &report, "Total               %25.6f\n\n", total_time_in_seconds);

  absl::StrAppendFormat(&report,
                        "Termination:        %25s (%s)\n",
                        TerminationTypeToString(termination_type),
                        message);
  return report;
}

void Solve(const GradientProblemSolver::Options& options,
           const GradientProblem& problem,
           double* parameters,
           GradientProblemSolver::Summary* summary) {
  GradientProblemSolver solver;
  solver.Solve(options, problem, parameters, summary);
}

}  // namespace ceres
