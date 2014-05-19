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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/line_search_solver.h"

#include <numeric>
#include <string>
#include "ceres/callbacks.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/line_search_minimizer.h"
#include "ceres/minimizer.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/solver_utils.h"
#include "ceres/summary_utils.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {
namespace {

void Minimize(const Solver::Options& options,
              Program* program,
              Evaluator* evaluator,
              Solver::Summary* summary) {
  Minimizer::Options minimizer_options(options);

  // The optimizer works on contiguous parameter vectors; allocate
  // some and collect the discontiguous parameters into the continuous
  // parameter vector.
  Vector parameters(program->NumParameters());
  program->ParameterBlocksToStateVector(parameters.data());

  LoggingCallback logging_callback(LINE_SEARCH,
                                   options.minimizer_progress_to_stdout);
  if (options.logging_type != SILENT) {
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       &logging_callback);
  }

  StateUpdatingCallback updating_callback(program, parameters.data());
  if (options.update_state_every_iteration) {
    // This must get pushed to the front of the callbacks so that it is run
    // before any of the user callbacks.
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       &updating_callback);
  }

  minimizer_options.evaluator = evaluator;
  LineSearchMinimizer minimizer;
  double minimizer_start_time = WallTimeInSeconds();
  minimizer.Minimize(minimizer_options, parameters.data(), summary);

  // If the user aborted mid-optimization or the optimization
  // terminated because of a numerical failure, then do not update
  // user state.
  if (summary->IsSolutionUsable()) {
    program->StateVectorToParameterBlocks(parameters.data());
    program->CopyParameterBlockStateToUserState();
  }

  summary->minimizer_time_in_seconds =
      WallTimeInSeconds() - minimizer_start_time;
}

}  // namespace

void LineSearchSolver::Solve(const Solver::Options& original_options,
                             ProblemImpl* problem_impl,
                             Solver::Summary* summary) {
  EventLogger event_logger("LineSearchSolver::Solve");
  const double solver_start_time = WallTimeInSeconds();
  Program* original_program = problem_impl->mutable_program();

  VLOG(2) << "Initial problem: "
          << original_program->NumParameterBlocks()
          << " parameter blocks, "
          << original_program->NumParameters()
          << " parameters,  "
          << original_program->NumResidualBlocks()
          << " residual blocks, "
          << original_program->NumResiduals()
          << " residuals.";

  SummarizeGivenProgram(*original_program, summary);
  summary->minimizer_type = LINE_SEARCH;
  summary->line_search_direction_type =
      original_options.line_search_direction_type;
  summary->max_lbfgs_rank = original_options.max_lbfgs_rank;
  summary->line_search_type = original_options.line_search_type;
  summary->line_search_interpolation_type =
      original_options.line_search_interpolation_type;
  summary->nonlinear_conjugate_gradient_type =
      original_options.nonlinear_conjugate_gradient_type;

  if (original_program->IsBoundsConstrained()) {
    summary->message =  "LINE_SEARCH Minimizer does not support bounds.";
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  if (!original_program->ParameterBlocksAreFinite(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  original_program->SetParameterBlockStatePtrsToUserStatePtrs();
  event_logger.AddEvent("Init");

  Solver::Options options(original_options);

#ifndef CERES_USE_OPENMP
  if (options.num_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_threads=1 is supported. Switching "
        << "to single threaded mode.";
    options.num_threads = 1;
  }
#endif  // CERES_USE_OPENMP

  summary->num_threads_given = original_options.num_threads;
  summary->num_threads_used = options.num_threads;
  event_logger.AddEvent("Init");

  // If the user requests gradient checking, construct a new
  // ProblemImpl by wrapping the CostFunctions of problem_impl inside
  // GradientCheckingCostFunction and replacing problem_impl with
  // gradient_checking_problem_impl.
  scoped_ptr<ProblemImpl> gradient_checking_problem_impl;
  if (options.check_gradients) {
    VLOG(1) << "Checking Gradients";
    gradient_checking_problem_impl.reset(
        CreateGradientCheckingProblemImpl(
            problem_impl,
            options.numeric_derivative_relative_step_size,
            options.gradient_check_relative_precision));

    // From here on, problem_impl will point to the gradient checking
    // version.
    problem_impl = gradient_checking_problem_impl.get();
    event_logger.AddEvent("ConstructGradientCheckingProblem");
  }

  // Removed fixed blocks from the program.
  vector<double*> removed_parameter_blocks;
  scoped_ptr<Program> reduced_program(
      CreateReducedProgram(problem_impl->program(),
                           &removed_parameter_blocks,
                           &summary->fixed_cost,
                           &summary->message));

  event_logger.AddEvent("CreateReducedProgram");
  if (reduced_program == NULL) {
    return;
  }

  VLOG(2) << "Reduced problem: "
          << reduced_program->NumParameterBlocks()
          << " parameter blocks, "
          << reduced_program->NumParameters()
          << " parameters,  "
          << reduced_program->NumResidualBlocks()
          << " residual blocks, "
          << reduced_program->NumResiduals()
          << " residuals.";

  SummarizeReducedProgram(*reduced_program, summary);

  if (summary->num_parameter_blocks_reduced == 0) {
    summary->preprocessor_time_in_seconds =
        WallTimeInSeconds() - solver_start_time;

    summary->message =
        "Function tolerance reached. "
        "No non-constant parameter blocks found.";
    VLOG_IF(1, options.logging_type != SILENT)
        << "Terminating: " << summary->message;

    summary->termination_type = CONVERGENCE;
    summary->initial_cost = summary->fixed_cost;
    Finish(map<string, double>(),
           map<string, double>(),
           original_program,
           summary);
    return;
  }

  Evaluator::Options evaluator_options;
  // This ensures that we get a Block Jacobian Evaluator without any
  // requirement on orderings.
  evaluator_options.linear_solver_type = CGNR;
  evaluator_options.num_eliminate_blocks = 0;
  evaluator_options.num_threads = options.num_threads;
  scoped_ptr<Evaluator> evaluator(
      Evaluator::Create(evaluator_options,
                        reduced_program.get(),
                        &summary->message));
  event_logger.AddEvent("CreateEvaluator");
  if (evaluator == NULL) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  summary->preprocessor_time_in_seconds =
      WallTimeInSeconds() - solver_start_time;

  Minimize(options, reduced_program.get(), evaluator.get(), summary);
  Finish(evaluator->TimeStatistics(),
         map<string, double>(),
         original_program,
         summary);
}

}  // namespace internal
}  // namespace ceres
