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

#include "ceres/solver_impl.h"

#include <cstdio>
#include <iostream>  // NOLINT
#include <numeric>
#include <string>
#include "ceres/array_utils.h"
#include "ceres/callbacks.h"
#include "ceres/coordinate_descent_minimizer.h"
#include "ceres/cxsparse.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/iteration_callback.h"
#include "ceres/levenberg_marquardt_strategy.h"
#include "ceres/line_search_minimizer.h"
#include "ceres/linear_solver.h"
#include "ceres/map_util.h"
#include "ceres/minimizer.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/parameter_block_ordering.h"
#include "ceres/preconditioner.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/reorder_program.h"
#include "ceres/residual_block.h"
#include "ceres/stringprintf.h"
#include "ceres/suitesparse.h"
#include "ceres/summary_utils.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

void SolverImpl::TrustRegionMinimize(
    const Solver::Options& options,
    Program* program,
    CoordinateDescentMinimizer* inner_iteration_minimizer,
    Evaluator* evaluator,
    LinearSolver* linear_solver,
    Solver::Summary* summary) {
  Minimizer::Options minimizer_options(options);
  minimizer_options.is_constrained = program->IsBoundsConstrained();

  // The optimizer works on contiguous parameter vectors; allocate
  // some.
  Vector parameters(program->NumParameters());

  // Collect the discontiguous parameters into a contiguous state
  // vector.
  program->ParameterBlocksToStateVector(parameters.data());

  LoggingCallback logging_callback(TRUST_REGION,
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
  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());

  minimizer_options.jacobian = jacobian.get();
  minimizer_options.inner_iteration_minimizer = inner_iteration_minimizer;

  TrustRegionStrategy::Options trust_region_strategy_options;
  trust_region_strategy_options.linear_solver = linear_solver;
  trust_region_strategy_options.initial_radius =
      options.initial_trust_region_radius;
  trust_region_strategy_options.max_radius = options.max_trust_region_radius;
  trust_region_strategy_options.min_lm_diagonal = options.min_lm_diagonal;
  trust_region_strategy_options.max_lm_diagonal = options.max_lm_diagonal;
  trust_region_strategy_options.trust_region_strategy_type =
      options.trust_region_strategy_type;
  trust_region_strategy_options.dogleg_type = options.dogleg_type;
  scoped_ptr<TrustRegionStrategy> strategy(
      TrustRegionStrategy::Create(trust_region_strategy_options));
  minimizer_options.trust_region_strategy = strategy.get();

  TrustRegionMinimizer minimizer;
  double minimizer_start_time = WallTimeInSeconds();
  minimizer.Minimize(minimizer_options, parameters.data(), summary);

  // If the user aborted mid-optimization or the optimization
  // terminated because of a numerical failure, then do not update
  // user state.
  if (summary->termination_type != USER_FAILURE &&
      summary->termination_type != FAILURE) {
    program->StateVectorToParameterBlocks(parameters.data());
    program->CopyParameterBlockStateToUserState();
  }

  summary->minimizer_time_in_seconds =
      WallTimeInSeconds() - minimizer_start_time;
}

void SolverImpl::LineSearchMinimize(
    const Solver::Options& options,
    Program* program,
    Evaluator* evaluator,
    Solver::Summary* summary) {
  Minimizer::Options minimizer_options(options);

  // The optimizer works on contiguous parameter vectors; allocate some.
  Vector parameters(program->NumParameters());

  // Collect the discontiguous parameters into a contiguous state vector.
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
  if (summary->termination_type != USER_FAILURE &&
      summary->termination_type != FAILURE) {
    program->StateVectorToParameterBlocks(parameters.data());
    program->CopyParameterBlockStateToUserState();
  }

  summary->minimizer_time_in_seconds =
      WallTimeInSeconds() - minimizer_start_time;
}

void SolverImpl::Solve(const Solver::Options& options,
                       ProblemImpl* problem_impl,
                       Solver::Summary* summary) {
  VLOG(2) << "Initial problem: "
          << problem_impl->NumParameterBlocks()
          << " parameter blocks, "
          << problem_impl->NumParameters()
          << " parameters,  "
          << problem_impl->NumResidualBlocks()
          << " residual blocks, "
          << problem_impl->NumResiduals()
          << " residuals.";
  if (options.minimizer_type == TRUST_REGION) {
    TrustRegionSolve(options, problem_impl, summary);
  } else {
    LineSearchSolve(options, problem_impl, summary);
  }
}

void SolverImpl::TrustRegionSolve(const Solver::Options& original_options,
                                  ProblemImpl* original_problem_impl,
                                  Solver::Summary* summary) {
  EventLogger event_logger("TrustRegionSolve");
  double solver_start_time = WallTimeInSeconds();

  Program* original_program = original_problem_impl->mutable_program();
  ProblemImpl* problem_impl = original_problem_impl;

  summary->minimizer_type = TRUST_REGION;

  SummarizeGivenProgram(*original_program, summary);
  OrderingToGroupSizes(original_options.linear_solver_ordering.get(),
                       &(summary->linear_solver_ordering_given));
  OrderingToGroupSizes(original_options.inner_iteration_ordering.get(),
                       &(summary->inner_iteration_ordering_given));

  Solver::Options options(original_options);

#ifndef CERES_USE_OPENMP
  if (options.num_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_threads=1 is supported. Switching "
        << "to single threaded mode.";
    options.num_threads = 1;
  }
  if (options.num_linear_solver_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_linear_solver_threads=1 is supported. Switching "
        << "to single threaded mode.";
    options.num_linear_solver_threads = 1;
  }
#endif

  summary->num_threads_given = original_options.num_threads;
  summary->num_threads_used = options.num_threads;

  if (options.trust_region_minimizer_iterations_to_dump.size() > 0 &&
      options.trust_region_problem_dump_format_type != CONSOLE &&
      options.trust_region_problem_dump_directory.empty()) {
    summary->message =
        "Solver::Options::trust_region_problem_dump_directory is empty.";
    LOG(ERROR) << summary->message;
    return;
  }

  if (!original_program->ParameterBlocksAreFinite(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  if (!original_program->IsFeasible(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  event_logger.AddEvent("Init");

  original_program->SetParameterBlockStatePtrsToUserStatePtrs();
  event_logger.AddEvent("SetParameterBlockPtrs");

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
  }

  if (options.linear_solver_ordering.get() != NULL) {
    if (!IsOrderingValid(options, problem_impl, &summary->message)) {
      LOG(ERROR) << summary->message;
      return;
    }
    event_logger.AddEvent("CheckOrdering");
  } else {
    options.linear_solver_ordering.reset(new ParameterBlockOrdering);
    const ProblemImpl::ParameterMap& parameter_map =
        problem_impl->parameter_map();
    for (ProblemImpl::ParameterMap::const_iterator it = parameter_map.begin();
         it != parameter_map.end();
         ++it) {
      options.linear_solver_ordering->AddElementToGroup(it->first, 0);
    }
    event_logger.AddEvent("ConstructOrdering");
  }

  // Create the three objects needed to minimize: the transformed program, the
  // evaluator, and the linear solver.
  scoped_ptr<Program> reduced_program(CreateReducedProgram(&options,
                                                           problem_impl,
                                                           &summary->fixed_cost,
                                                           &summary->message));

  event_logger.AddEvent("CreateReducedProgram");
  if (reduced_program == NULL) {
    return;
  }

  OrderingToGroupSizes(options.linear_solver_ordering.get(),
                       &(summary->linear_solver_ordering_used));
  SummarizeReducedProgram(*reduced_program, summary);

  if (summary->num_parameter_blocks_reduced == 0) {
    summary->preprocessor_time_in_seconds =
        WallTimeInSeconds() - solver_start_time;

    double post_process_start_time = WallTimeInSeconds();

     summary->message =
        "Function tolerance reached. "
        "No non-constant parameter blocks found.";
    summary->termination_type = CONVERGENCE;
    VLOG_IF(1, options.logging_type != SILENT) << summary->message;

    summary->initial_cost = summary->fixed_cost;
    summary->final_cost = summary->fixed_cost;

    // Ensure the program state is set to the user parameters on the way out.
    original_program->SetParameterBlockStatePtrsToUserStatePtrs();
    original_program->SetParameterOffsetsAndIndex();

    summary->postprocessor_time_in_seconds =
        WallTimeInSeconds() - post_process_start_time;
    return;
  }

  scoped_ptr<LinearSolver>
      linear_solver(CreateLinearSolver(&options, &summary->message));
  event_logger.AddEvent("CreateLinearSolver");
  if (linear_solver == NULL) {
    return;
  }

  summary->linear_solver_type_given = original_options.linear_solver_type;
  summary->linear_solver_type_used = options.linear_solver_type;

  summary->preconditioner_type = options.preconditioner_type;
  summary->visibility_clustering_type = options.visibility_clustering_type;

  summary->num_linear_solver_threads_given =
      original_options.num_linear_solver_threads;
  summary->num_linear_solver_threads_used = options.num_linear_solver_threads;

  summary->dense_linear_algebra_library_type =
      options.dense_linear_algebra_library_type;
  summary->sparse_linear_algebra_library_type =
      options.sparse_linear_algebra_library_type;

  summary->trust_region_strategy_type = options.trust_region_strategy_type;
  summary->dogleg_type = options.dogleg_type;

  scoped_ptr<Evaluator> evaluator(CreateEvaluator(options,
                                                  problem_impl->parameter_map(),
                                                  reduced_program.get(),
                                                  &summary->message));

  event_logger.AddEvent("CreateEvaluator");

  if (evaluator == NULL) {
    return;
  }

  scoped_ptr<CoordinateDescentMinimizer> inner_iteration_minimizer;
  if (options.use_inner_iterations) {
    if (reduced_program->parameter_blocks().size() < 2) {
      LOG(WARNING) << "Reduced problem only contains one parameter block."
                   << "Disabling inner iterations.";
    } else {
      inner_iteration_minimizer.reset(
          CreateInnerIterationMinimizer(options,
                                        *reduced_program,
                                        problem_impl->parameter_map(),
                                        summary));
      if (inner_iteration_minimizer == NULL) {
        LOG(ERROR) << summary->message;
        return;
      }
    }
  }
  event_logger.AddEvent("CreateInnerIterationMinimizer");

  double minimizer_start_time = WallTimeInSeconds();
  summary->preprocessor_time_in_seconds =
      minimizer_start_time - solver_start_time;

  // Run the optimization.
  TrustRegionMinimize(options,
                      reduced_program.get(),
                      inner_iteration_minimizer.get(),
                      evaluator.get(),
                      linear_solver.get(),
                      summary);
  event_logger.AddEvent("Minimize");

  double post_process_start_time = WallTimeInSeconds();

  SetSummaryFinalCost(summary);

  // Ensure the program state is set to the user parameters on the way
  // out.
  original_program->SetParameterBlockStatePtrsToUserStatePtrs();
  original_program->SetParameterOffsetsAndIndex();

  const map<string, double>& linear_solver_time_statistics =
      linear_solver->TimeStatistics();
  summary->linear_solver_time_in_seconds =
      FindWithDefault(linear_solver_time_statistics,
                      "LinearSolver::Solve",
                      0.0);

  const map<string, double>& evaluator_time_statistics =
      evaluator->TimeStatistics();

  summary->residual_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Residual", 0.0);
  summary->jacobian_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Jacobian", 0.0);

  // Stick a fork in it, we're done.
  summary->postprocessor_time_in_seconds =
      WallTimeInSeconds() - post_process_start_time;
  event_logger.AddEvent("PostProcess");
}

void SolverImpl::LineSearchSolve(const Solver::Options& original_options,
                                 ProblemImpl* original_problem_impl,
                                 Solver::Summary* summary) {
  double solver_start_time = WallTimeInSeconds();

  Program* original_program = original_problem_impl->mutable_program();
  ProblemImpl* problem_impl = original_problem_impl;

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

  Solver::Options options(original_options);

  // This ensures that we get a Block Jacobian Evaluator along with
  // none of the Schur nonsense. This file will have to be extensively
  // refactored to deal with the various bits of cleanups related to
  // line search.
  options.linear_solver_type = CGNR;


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

  if (!original_program->ParameterBlocksAreFinite(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  if (options.linear_solver_ordering.get() != NULL) {
    if (!IsOrderingValid(options, problem_impl, &summary->message)) {
      LOG(ERROR) << summary->message;
      return;
    }
  } else {
    options.linear_solver_ordering.reset(new ParameterBlockOrdering);
    const ProblemImpl::ParameterMap& parameter_map =
        problem_impl->parameter_map();
    for (ProblemImpl::ParameterMap::const_iterator it = parameter_map.begin();
         it != parameter_map.end();
         ++it) {
      options.linear_solver_ordering->AddElementToGroup(it->first, 0);
    }
  }


  original_program->SetParameterBlockStatePtrsToUserStatePtrs();

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
  }

  // Create the three objects needed to minimize: the transformed program, the
  // evaluator, and the linear solver.
  scoped_ptr<Program> reduced_program(CreateReducedProgram(&options,
                                                           problem_impl,
                                                           &summary->fixed_cost,
                                                           &summary->message));
  if (reduced_program == NULL) {
    return;
  }

  SummarizeReducedProgram(*reduced_program, summary);
  if (summary->num_parameter_blocks_reduced == 0) {
    summary->preprocessor_time_in_seconds =
        WallTimeInSeconds() - solver_start_time;

    summary->message =
        "Function tolerance reached. "
        "No non-constant parameter blocks found.";
    summary->termination_type = CONVERGENCE;
    VLOG_IF(1, options.logging_type != SILENT) << summary->message;
    summary->initial_cost = summary->fixed_cost;
    summary->final_cost = summary->fixed_cost;

    const double post_process_start_time = WallTimeInSeconds();
    SetSummaryFinalCost(summary);

    // Ensure the program state is set to the user parameters on the way out.
    original_program->SetParameterBlockStatePtrsToUserStatePtrs();
    original_program->SetParameterOffsetsAndIndex();

    summary->postprocessor_time_in_seconds =
        WallTimeInSeconds() - post_process_start_time;
    return;
  }

  scoped_ptr<Evaluator> evaluator(CreateEvaluator(options,
                                                  problem_impl->parameter_map(),
                                                  reduced_program.get(),
                                                  &summary->message));
  if (evaluator == NULL) {
    return;
  }

  const double minimizer_start_time = WallTimeInSeconds();
  summary->preprocessor_time_in_seconds =
      minimizer_start_time - solver_start_time;

  // Run the optimization.
  LineSearchMinimize(options, reduced_program.get(), evaluator.get(), summary);

  const double post_process_start_time = WallTimeInSeconds();

  SetSummaryFinalCost(summary);

  // Ensure the program state is set to the user parameters on the way out.
  original_program->SetParameterBlockStatePtrsToUserStatePtrs();
  original_program->SetParameterOffsetsAndIndex();

  const map<string, double>& evaluator_time_statistics =
      evaluator->TimeStatistics();

  summary->residual_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Residual", 0.0);
  summary->jacobian_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Jacobian", 0.0);

  // Stick a fork in it, we're done.
  summary->postprocessor_time_in_seconds =
      WallTimeInSeconds() - post_process_start_time;
}

bool SolverImpl::IsOrderingValid(const Solver::Options& options,
                                 const ProblemImpl* problem_impl,
                                 string* error) {
  if (options.linear_solver_ordering->NumElements() !=
      problem_impl->NumParameterBlocks()) {
      *error = "Number of parameter blocks in user supplied ordering "
          "does not match the number of parameter blocks in the problem";
    return false;
  }

  const Program& program = problem_impl->program();
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  for (vector<ParameterBlock*>::const_iterator it = parameter_blocks.begin();
       it != parameter_blocks.end();
       ++it) {
    if (!options.linear_solver_ordering
        ->IsMember(const_cast<double*>((*it)->user_state()))) {
      *error = "Problem contains a parameter block that is not in "
          "the user specified ordering.";
      return false;
    }
  }

  if (IsSchurType(options.linear_solver_type) &&
      options.linear_solver_ordering->NumGroups() > 1) {
    const vector<ResidualBlock*>& residual_blocks = program.residual_blocks();
    const set<double*>& e_blocks  =
        options.linear_solver_ordering->group_to_elements().begin()->second;
    if (!IsParameterBlockSetIndependent(e_blocks, residual_blocks)) {
      *error = "The user requested the use of a Schur type solver. "
          "But the first elimination group in the ordering is not an "
          "independent set.";
      return false;
    }
  }
  return true;
}

bool SolverImpl::IsParameterBlockSetIndependent(
    const set<double*>& parameter_block_ptrs,
    const vector<ResidualBlock*>& residual_blocks) {
  // Loop over each residual block and ensure that no two parameter
  // blocks in the same residual block are part of
  // parameter_block_ptrs as that would violate the assumption that it
  // is an independent set in the Hessian matrix.
  for (vector<ResidualBlock*>::const_iterator it = residual_blocks.begin();
       it != residual_blocks.end();
       ++it) {
    ParameterBlock* const* parameter_blocks = (*it)->parameter_blocks();
    const int num_parameter_blocks = (*it)->NumParameterBlocks();
    int count = 0;
    for (int i = 0; i < num_parameter_blocks; ++i) {
      count += parameter_block_ptrs.count(
          parameter_blocks[i]->mutable_user_state());
    }
    if (count > 1) {
      return false;
    }
  }
  return true;
}

Program* SolverImpl::CreateReducedProgram(Solver::Options* options,
                                          ProblemImpl* problem_impl,
                                          double* fixed_cost,
                                          string* error) {
  CHECK_NOTNULL(options->linear_solver_ordering.get());
  Program* original_program = problem_impl->mutable_program();

  vector<double*> removed_parameter_blocks;
  scoped_ptr<Program> reduced_program(
      original_program->CreateReducedProgram(&removed_parameter_blocks,
                                             fixed_cost,
                                             error));
  if (reduced_program.get() == NULL) {
    return NULL;
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

  if (reduced_program->NumParameterBlocks() == 0) {
    LOG(WARNING) << "No varying parameter blocks to optimize; "
                 << "bailing early.";
    return reduced_program.release();
  }

  ParameterBlockOrdering* linear_solver_ordering =
      options->linear_solver_ordering.get();
  const int min_group_id =
      linear_solver_ordering->MinNonZeroGroup();
  linear_solver_ordering->Remove(removed_parameter_blocks);

  ParameterBlockOrdering* inner_iteration_ordering =
      options->inner_iteration_ordering.get();
  if (inner_iteration_ordering != NULL) {
    inner_iteration_ordering->Remove(removed_parameter_blocks);
  }

  if (IsSchurType(options->linear_solver_type) &&
      linear_solver_ordering->GroupSize(min_group_id) == 0) {
    // If the user requested the use of a Schur type solver, and
    // supplied a non-NULL linear_solver_ordering object with more than
    // one elimination group, then it can happen that after all the
    // parameter blocks which are fixed or unused have been removed from
    // the program and the ordering, there are no more parameter blocks
    // in the first elimination group.
    //
    // In such a case, the use of a Schur type solver is not possible,
    // as they assume there is at least one e_block. Thus, we
    // automatically switch to the closest solver to the one indicated
    // by the user.
    if (options->linear_solver_type == ITERATIVE_SCHUR) {
      options->preconditioner_type =
        Preconditioner::PreconditionerForZeroEBlocks(
            options->preconditioner_type);
    }

    options->linear_solver_type =
        LinearSolver::LinearSolverForZeroEBlocks(
            options->linear_solver_type);
  }

  if (IsSchurType(options->linear_solver_type)) {
    if (!ReorderProgramForSchurTypeLinearSolver(
            options->linear_solver_type,
            options->sparse_linear_algebra_library_type,
            problem_impl->parameter_map(),
            linear_solver_ordering,
            reduced_program.get(),
            error)) {
      return NULL;
    }
    return reduced_program.release();
  }

  if (options->linear_solver_type == SPARSE_NORMAL_CHOLESKY &&
      !options->dynamic_sparsity) {
    if (!ReorderProgramForSparseNormalCholesky(
            options->sparse_linear_algebra_library_type,
            *linear_solver_ordering,
            reduced_program.get(),
            error)) {
      return NULL;
    }

    return reduced_program.release();
  }

  reduced_program->SetParameterOffsetsAndIndex();
  return reduced_program.release();
}

LinearSolver* SolverImpl::CreateLinearSolver(Solver::Options* options,
                                             string* error) {
  CHECK_NOTNULL(options);
  CHECK_NOTNULL(options->linear_solver_ordering.get());
  CHECK_NOTNULL(error);

  if (options->trust_region_strategy_type == DOGLEG) {
    if (options->linear_solver_type == ITERATIVE_SCHUR ||
        options->linear_solver_type == CGNR) {
      *error = "DOGLEG only supports exact factorization based linear "
               "solvers. If you want to use an iterative solver please "
               "use LEVENBERG_MARQUARDT as the trust_region_strategy_type";
      return NULL;
    }
  }

#ifdef CERES_NO_LAPACK
  if (options->linear_solver_type == DENSE_NORMAL_CHOLESKY &&
      options->dense_linear_algebra_library_type == LAPACK) {
    *error = "Can't use DENSE_NORMAL_CHOLESKY with LAPACK because "
        "LAPACK was not enabled when Ceres was built.";
    return NULL;
  }

  if (options->linear_solver_type == DENSE_QR &&
      options->dense_linear_algebra_library_type == LAPACK) {
    *error = "Can't use DENSE_QR with LAPACK because "
        "LAPACK was not enabled when Ceres was built.";
    return NULL;
  }

  if (options->linear_solver_type == DENSE_SCHUR &&
      options->dense_linear_algebra_library_type == LAPACK) {
    *error = "Can't use DENSE_SCHUR with LAPACK because "
        "LAPACK was not enabled when Ceres was built.";
    return NULL;
  }
#endif

#ifdef CERES_NO_SUITESPARSE
  if (options->linear_solver_type == SPARSE_NORMAL_CHOLESKY &&
      options->sparse_linear_algebra_library_type == SUITE_SPARSE) {
    *error = "Can't use SPARSE_NORMAL_CHOLESKY with SUITESPARSE because "
             "SuiteSparse was not enabled when Ceres was built.";
    return NULL;
  }

  if (options->preconditioner_type == CLUSTER_JACOBI) {
    *error =  "CLUSTER_JACOBI preconditioner not suppored. Please build Ceres "
        "with SuiteSparse support.";
    return NULL;
  }

  if (options->preconditioner_type == CLUSTER_TRIDIAGONAL) {
    *error =  "CLUSTER_TRIDIAGONAL preconditioner not suppored. Please build "
        "Ceres with SuiteSparse support.";
    return NULL;
  }
#endif

#ifdef CERES_NO_CXSPARSE
  if (options->linear_solver_type == SPARSE_NORMAL_CHOLESKY &&
      options->sparse_linear_algebra_library_type == CX_SPARSE) {
    *error = "Can't use SPARSE_NORMAL_CHOLESKY with CXSPARSE because "
             "CXSparse was not enabled when Ceres was built.";
    return NULL;
  }
#endif

  if (options->max_linear_solver_iterations <= 0) {
    *error = "Solver::Options::max_linear_solver_iterations is not positive.";
    return NULL;
  }
  if (options->min_linear_solver_iterations <= 0) {
    *error = "Solver::Options::min_linear_solver_iterations is not positive.";
    return NULL;
  }
  if (options->min_linear_solver_iterations >
      options->max_linear_solver_iterations) {
    *error = "Solver::Options::min_linear_solver_iterations > "
        "Solver::Options::max_linear_solver_iterations.";
    return NULL;
  }

  LinearSolver::Options linear_solver_options;
  linear_solver_options.min_num_iterations =
        options->min_linear_solver_iterations;
  linear_solver_options.max_num_iterations =
      options->max_linear_solver_iterations;
  linear_solver_options.type = options->linear_solver_type;
  linear_solver_options.preconditioner_type = options->preconditioner_type;
  linear_solver_options.visibility_clustering_type =
      options->visibility_clustering_type;
  linear_solver_options.sparse_linear_algebra_library_type =
      options->sparse_linear_algebra_library_type;
  linear_solver_options.dense_linear_algebra_library_type =
      options->dense_linear_algebra_library_type;
  linear_solver_options.use_postordering = options->use_postordering;
  linear_solver_options.dynamic_sparsity = options->dynamic_sparsity;

  // Ignore user's postordering preferences and force it to be true if
  // cholmod_camd is not available. This ensures that the linear
  // solver does not assume that a fill-reducing pre-ordering has been
  // done.
#if !defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CAMD)
  if (IsSchurType(linear_solver_options.type) &&
      options->sparse_linear_algebra_library_type == SUITE_SPARSE) {
    linear_solver_options.use_postordering = true;
  }
#endif

  linear_solver_options.num_threads = options->num_linear_solver_threads;
  options->num_linear_solver_threads = linear_solver_options.num_threads;

  OrderingToGroupSizes(options->linear_solver_ordering.get(),
                       &linear_solver_options.elimination_groups);
  // Schur type solvers, expect at least two elimination groups. If
  // there is only one elimination group, then CreateReducedProgram
  // guarantees that this group only contains e_blocks. Thus we add a
  // dummy elimination group with zero blocks in it.
  if (IsSchurType(linear_solver_options.type) &&
      linear_solver_options.elimination_groups.size() == 1) {
    linear_solver_options.elimination_groups.push_back(0);
  }

  return LinearSolver::Create(linear_solver_options);
}

Evaluator* SolverImpl::CreateEvaluator(
    const Solver::Options& options,
    const ProblemImpl::ParameterMap& parameter_map,
    Program* program,
    string* error) {
  Evaluator::Options evaluator_options;
  evaluator_options.linear_solver_type = options.linear_solver_type;
  evaluator_options.num_eliminate_blocks =
      (options.linear_solver_ordering->NumGroups() > 0 &&
       IsSchurType(options.linear_solver_type))
      ? (options.linear_solver_ordering
         ->group_to_elements().begin()
         ->second.size())
      : 0;
  evaluator_options.num_threads = options.num_threads;
  evaluator_options.dynamic_sparsity = options.dynamic_sparsity;
  return Evaluator::Create(evaluator_options, program, error);
}

CoordinateDescentMinimizer* SolverImpl::CreateInnerIterationMinimizer(
    const Solver::Options& options,
    const Program& program,
    const ProblemImpl::ParameterMap& parameter_map,
    Solver::Summary* summary) {
  summary->inner_iterations_given = true;

  scoped_ptr<CoordinateDescentMinimizer> inner_iteration_minimizer(
      new CoordinateDescentMinimizer);
  scoped_ptr<ParameterBlockOrdering> inner_iteration_ordering;
  ParameterBlockOrdering* ordering_ptr  = NULL;

  if (options.inner_iteration_ordering.get() == NULL) {
    inner_iteration_ordering.reset(
        CoordinateDescentMinimizer::CreateOrdering(program));
    ordering_ptr = inner_iteration_ordering.get();
  } else {
    ordering_ptr = options.inner_iteration_ordering.get();
    if (!CoordinateDescentMinimizer::IsOrderingValid(program,
                                                     *ordering_ptr,
                                                     &summary->message)) {
      return NULL;
    }
  }

  if (!inner_iteration_minimizer->Init(program,
                                       parameter_map,
                                       *ordering_ptr,
                                       &summary->message)) {
    return NULL;
  }

  summary->inner_iterations_used = true;
  summary->inner_iteration_time_in_seconds = 0.0;
  OrderingToGroupSizes(ordering_ptr,
                       &(summary->inner_iteration_ordering_used));
  return inner_iteration_minimizer.release();
}

}  // namespace internal
}  // namespace ceres
