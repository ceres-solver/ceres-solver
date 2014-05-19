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
//         mierle@gmail.com (Keir Mierle)

#include "ceres/trust_region_solver.h"

#include <numeric>
#include <string>
#include "ceres/callbacks.h"
#include "ceres/coordinate_descent_minimizer.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/linear_solver.h"
#include "ceres/minimizer.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/reorder_program.h"
#include "ceres/residual_block.h"
#include "ceres/stringprintf.h"
#include "ceres/summary_utils.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/wall_time.h"
#include "ceres/solver_utils.h"
#include "ceres/suitesparse.h"

namespace ceres {
namespace internal {
namespace {

void Minimize(const Solver::Options& options,
              Program* program,
              CoordinateDescentMinimizer* inner_iteration_minimizer,
              Evaluator* evaluator,
              LinearSolver* linear_solver,
              Solver::Summary* summary) {
  Minimizer::Options minimizer_options(options);
  minimizer_options.is_constrained = program->IsBoundsConstrained();

  // The optimizer works on contiguous parameter vectors; allocate
  // some and collect the discontiguous parameters into the continuous
  // parameter vector.
  Vector parameters(program->NumParameters());
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
  if (summary->IsSolutionUsable()) {
    program->StateVectorToParameterBlocks(parameters.data());
    program->CopyParameterBlockStateToUserState();
  }

  summary->minimizer_time_in_seconds =
      WallTimeInSeconds() - minimizer_start_time;
}

bool ReorderProgram(const Solver::Options& options,
                    const ProblemImpl::ParameterMap& parameter_map,
                    Program* program,
                    string* error) {
  if (IsSchurType(options.linear_solver_type)) {
    if (!ReorderProgramForSchurTypeLinearSolver(
            options.linear_solver_type,
            options.sparse_linear_algebra_library_type,
            parameter_map,
            options.linear_solver_ordering.get(),
            program,
            error)) {
      return false;
    }
  }

  if (options.linear_solver_type == SPARSE_NORMAL_CHOLESKY &&
      !options.dynamic_sparsity) {
    if (!ReorderProgramForSparseNormalCholesky(
            options.sparse_linear_algebra_library_type,
            *options.linear_solver_ordering,
            program,
            error)) {
      return false;
    }
  }

  program->SetParameterOffsetsAndIndex();
  return true;
}

Evaluator* CreateEvaluator(const Solver::Options& options,
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

// If the linear solver is of Schur type, then replace it with the
// closest equivalent linear solver. This is done when the user
// requested a Schur type solver but the problem structure makes it
// impossible to use one.
//
// If the linear solver is not of Schur type, the function is a
// no-op.
void AlternateLinearSolverForSchurTypeLinearSolver(Solver::Options* options) {
  const LinearSolverType given_linear_solver_type = options->linear_solver_type;
  options->linear_solver_type = LinearSolver::LinearSolverForZeroEBlocks(
      given_linear_solver_type);

  string message;
  if (given_linear_solver_type == ITERATIVE_SCHUR) {
    const PreconditionerType given_preconditioner_type =
        options->preconditioner_type;
    //options->preconditioner_type = Preconditioner::PreconditionerForZeroEBlocks(
    // given_preconditioner_type);
    message =
        StringPrintf("No E blocks. Switching from %s(%s) to %s(%s).",
                     LinearSolverTypeToString(given_linear_solver_type),
                     PreconditionerTypeToString(given_preconditioner_type),
                     LinearSolverTypeToString(options->linear_solver_type),
                     PreconditionerTypeToString(options->preconditioner_type));
  } else {
    message =
        StringPrintf("No E blocks. Switching from %s to %s.",
                     LinearSolverTypeToString(given_linear_solver_type),
                     LinearSolverTypeToString(options->linear_solver_type));
  }

  VLOG_IF(1, options->logging_type != SILENT) << message;
}

CoordinateDescentMinimizer* CreateInnerIterationMinimizer(
    const Solver::Options& options,
    const Program& program,
    const ProblemImpl::ParameterMap& parameter_map,
    Solver::Summary* summary) {
  summary->inner_iterations_given = true;
  shared_ptr<ParameterBlockOrdering> inner_iteration_ordering;
  if (options.inner_iteration_ordering.get() == NULL) {
    inner_iteration_ordering.reset(
        CoordinateDescentMinimizer::CreateOrdering(program));
  } else {
    inner_iteration_ordering = options.inner_iteration_ordering;
    if (!CoordinateDescentMinimizer::IsOrderingValid(program,
                                                     *inner_iteration_ordering,
                                                     &summary->message)) {
      return NULL;
    }
  }

  scoped_ptr<CoordinateDescentMinimizer> inner_iteration_minimizer(
      new CoordinateDescentMinimizer);
  if (!inner_iteration_minimizer->Init(program,
                                       parameter_map,
                                       *inner_iteration_ordering,
                                       &summary->message)) {
    return NULL;
  }

  summary->inner_iterations_used = true;
  summary->inner_iteration_time_in_seconds = 0.0;
  OrderingToGroupSizes(inner_iteration_ordering.get(),
                       &(summary->inner_iteration_ordering_used));
  return inner_iteration_minimizer.release();
}

LinearSolver* CreateLinearSolver(const Solver::Options& options,
                                 string* error) {
  CHECK_NOTNULL(options.linear_solver_ordering.get());
  CHECK_NOTNULL(error);

  LinearSolver::Options linear_solver_options;
  linear_solver_options.min_num_iterations =
        options.min_linear_solver_iterations;
  linear_solver_options.max_num_iterations =
      options.max_linear_solver_iterations;
  linear_solver_options.type = options.linear_solver_type;
  linear_solver_options.preconditioner_type = options.preconditioner_type;
  linear_solver_options.visibility_clustering_type =
      options.visibility_clustering_type;
  linear_solver_options.sparse_linear_algebra_library_type =
      options.sparse_linear_algebra_library_type;
  linear_solver_options.dense_linear_algebra_library_type =
      options.dense_linear_algebra_library_type;
  linear_solver_options.dynamic_sparsity = options.dynamic_sparsity;
  linear_solver_options.num_threads = options.num_linear_solver_threads;

  linear_solver_options.use_postordering = options.use_postordering;

  // Ignore user's postordering preferences and force it to be true
  // if CAMD is not available. This ensures that the linear solver
  // does not assume that a fill-reducing pre-ordering has been
  // done.
  if (IsSchurType(linear_solver_options.type) &&
      options.sparse_linear_algebra_library_type == SUITE_SPARSE &&
      !SuiteSparse::IsConstrainedApproximateMinimumDegreeOrderingAvailable()) {
    linear_solver_options.use_postordering = true;
  }

  if (IsSchurType(linear_solver_options.type)) {
    OrderingToGroupSizes(options.linear_solver_ordering.get(),
                         &linear_solver_options.elimination_groups);
    // Schur type solvers, expect at least two elimination groups. If
    // there is only one elimination group, then it is guaranteed this
    // group only contains e_blocks. Thus we add a dummy elimination
    // group with zero blocks in it.
    if (linear_solver_options.elimination_groups.size() == 1) {
      linear_solver_options.elimination_groups.push_back(0);
    }
  }

  return LinearSolver::Create(linear_solver_options);
}

}  // namespace

void TrustRegionSolver::Solve(const Solver::Options& original_options,
                              ProblemImpl* problem_impl,
                              Solver::Summary* summary) {
  EventLogger event_logger("TrustRegionSolve");
  double solver_start_time = WallTimeInSeconds();
  Solver::Options options(original_options);
  Program* original_program = problem_impl->mutable_program();
  const ProblemImpl::ParameterMap& parameter_map =
      problem_impl->parameter_map();

#ifndef CERES_USE_OPENMP
  if (options.num_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_threads = 1 is supported. Switching "
        << "to single threaded mode.";
    options.num_threads = 1;
  }
  if (options.num_linear_solver_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_linear_solver_threads = 1 is supported. Switching "
        << "to single threaded mode.";
    options.num_linear_solver_threads = 1;
  }
#endif

  summary->num_threads_given = original_options.num_threads;
  summary->num_threads_used = options.num_threads;
  summary->trust_region_strategy_type = options.trust_region_strategy_type;
  summary->dogleg_type = options.dogleg_type;
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
  summary->minimizer_type = TRUST_REGION;
  SummarizeGivenProgram(*original_program, summary);
  OrderingToGroupSizes(options.linear_solver_ordering.get(),
                       &(summary->linear_solver_ordering_given));
  OrderingToGroupSizes(options.inner_iteration_ordering.get(),
                       &(summary->inner_iteration_ordering_given));

  if (!original_program->ParameterBlocksAreFinite(&summary->message) ||
      !original_program->IsFeasible(&summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  original_program->SetParameterBlockStatePtrsToUserStatePtrs();

  // If the user requests gradient checking, construct a new
  // ProblemImpl by wrapping the CostFunctions of problem_impl inside
  // GradientCheckingCostFunction and replacing problem_impl with
  // gradient_checking_problem_impl.
  scoped_ptr<ProblemImpl> gradient_checking_problem_impl;
  if (options.check_gradients) {
    VLOG_IF(1, options.logging_type != SILENT) << "Checking Gradients";
    gradient_checking_problem_impl.reset(
        CreateGradientCheckingProblemImpl(
            problem_impl,
            options.numeric_derivative_relative_step_size,
            options.gradient_check_relative_precision));

    // From here on, problem_impl will point to the gradient checking
    // version.
    problem_impl = gradient_checking_problem_impl.get();
  }

  vector<double*> removed_parameter_blocks;
  scoped_ptr<Program> reduced_program(
      CreateReducedProgram(*original_program,
                           &removed_parameter_blocks,
                           &summary->fixed_cost,
                           &summary->message));
  event_logger.AddEvent("CreateReducedProgram");
  if (reduced_program == NULL) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  SummarizeReducedProgram(*reduced_program, summary);

  if (summary->num_parameter_blocks_reduced == 0) {
    summary->preprocessor_time_in_seconds =
        WallTimeInSeconds() - solver_start_time;
    summary->termination_type = CONVERGENCE;
    summary->initial_cost = summary->fixed_cost;
    summary->message =
        "Function tolerance reached. "
        "No non-constant parameter blocks found.";
    VLOG_IF(1, options.logging_type != SILENT) << "Terminating: "
                                               << summary->message;
    Finish(map<string, double>(),
           map<string, double>(),
           original_program,
           summary);
    return;
  }

  if (options.linear_solver_ordering.get() != NULL) {
    ParameterBlockOrdering* linear_solver_ordering =
        options.linear_solver_ordering.get();
    const int min_group_id =
        linear_solver_ordering->group_to_elements().begin()->first;
    linear_solver_ordering->Remove(removed_parameter_blocks);

    // If the user requested the use of a Schur type solver, and
    // supplied a linear_solver_ordering with more than one
    // elimination group, then it can happen that after all the
    // parameter blocks which are fixed or unused have been removed
    // from the ordering, there are no more parameter blocks in the
    // first elimination group. In such a case, the use of a Schur
    // type solver is not possible, as they assume there is at least
    // one e_block. Thus, we automatically switch to the closest
    // solver to the one indicated by the user.
    if (IsSchurType(options.linear_solver_type) &&
        linear_solver_ordering->GroupSize(min_group_id) == 0) {
      AlternateLinearSolverForSchurTypeLinearSolver(&options);
      summary->linear_solver_type_used = options.linear_solver_type;
      summary->preconditioner_type = options.preconditioner_type;
    }

    if (!IsOrderingValid(options, *reduced_program, &summary->message)) {
      LOG(ERROR) << "Terminating: " << summary->message;
      return;
    }
  } else {
    options.linear_solver_ordering.reset(new ParameterBlockOrdering);
    const vector<ParameterBlock*>& parameter_blocks =
        reduced_program->parameter_blocks();
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      options.linear_solver_ordering->AddElementToGroup(
          parameter_blocks[i]->mutable_user_state(), 0);
    }
  }

  if (!ReorderProgram(options,
                      parameter_map,
                      reduced_program.get(),
                      &summary->message)) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  OrderingToGroupSizes(options.linear_solver_ordering.get(),
                       &(summary->linear_solver_ordering_used));

  scoped_ptr<LinearSolver> linear_solver(
      CreateLinearSolver(options, &summary->message));
  if (linear_solver == NULL) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  scoped_ptr<Evaluator> evaluator(CreateEvaluator(options,
                                                  parameter_map,
                                                  reduced_program.get(),
                                                  &summary->message));
  event_logger.AddEvent("CreateEvaluator");
  if (evaluator == NULL) {
    LOG(ERROR) << "Terminating: " << summary->message;
    return;
  }

  scoped_ptr<CoordinateDescentMinimizer> inner_iteration_minimizer;
  if (options.use_inner_iterations) {
    if (options.inner_iteration_ordering.get() != NULL) {
      options.inner_iteration_ordering->Remove(removed_parameter_blocks);
    }

    if (reduced_program->parameter_blocks().size() == 1) {
      VLOG_IF(1, options.logging_type != SILENT)
          << "Reduced problem only contains one parameter block."
          << "Disabling inner iterations.";
    } else {
      inner_iteration_minimizer.reset(
          CreateInnerIterationMinimizer(options,
                                        *reduced_program,
                                        parameter_map,
                                        summary));
      event_logger.AddEvent("CreateInnerIterationMinimizer");
      if (inner_iteration_minimizer == NULL) {
        LOG(ERROR) << "Terminating: " << summary->message;
        return;
      }
    }
  }

  summary->preprocessor_time_in_seconds =
      WallTimeInSeconds() - solver_start_time;

  Minimize(options,
           reduced_program.get(),
           inner_iteration_minimizer.get(),
           evaluator.get(),
           linear_solver.get(),
           summary);
  event_logger.AddEvent("Minimize");

  LOG(INFO) << "Call Finish";
  Finish(evaluator->TimeStatistics(),
         linear_solver->TimeStatistics(),
         original_program,
         summary);
  event_logger.AddEvent("Finish");
}

bool TrustRegionSolver::IsOrderingValid(const Solver::Options& options,
                                        const Program& program,
                                        string* error) {
  if (options.linear_solver_ordering->NumElements() !=
      program.NumParameterBlocks()) {
      *error = "Number of parameter blocks in user supplied ordering "
          "does not match the number of parameter blocks in the problem";
    return false;
  }

  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  for (vector<ParameterBlock*>::const_iterator it = parameter_blocks.begin();
       it != parameter_blocks.end();
       ++it) {
    if (!options.linear_solver_ordering->IsMember(
            const_cast<double*>((*it)->user_state()))) {
      *error = "Problem contains a parameter block that is not in "
          "the user specified ordering.";
      return false;
    }
  }

  if (IsSchurType(options.linear_solver_type) &&
      options.linear_solver_ordering->NumGroups() > 1) {
    const set<double*>& e_blocks  =
        options.linear_solver_ordering->group_to_elements().begin()->second;
    if (!program.IsParameterBlockSetIndependent(e_blocks)) {
      *error = "The user requested the use of a Schur type solver. "
          "But the first elimination group in the ordering is not an "
          "independent set.";
      return false;
    }
  }
  return true;
}

}  // namespace internal
}  // namespace ceres
