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

#include "ceres/trust_region_preprocessor.h"

#include <numeric>
#include <string>
#include "ceres/callbacks.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/map_util.h"
#include "ceres/minimizer.h"
#include "ceres/preprocessor.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

TrustRegionPreprocessor::~TrustRegionPreprocessor() {
}

bool TrustRegionPreprocessor::Preprocess(const Solver::Options& options,
                                         ProblemImpl* problem,
                                         PreprocessedProblem* pp) {
  CHECK_NOTNULL(pp);
  pp->options = options;
  ChangeNumThreadsIfNeeded(&pp->options);

  Program* program = problem->mutable_program();
  program->SetParameterBlockStatePtrsToUserStatePtrs();
  if (!IsProgramValid(*program, &pp->error)) {
    return false;
  }

  if (options.check_gradients) {
    pp->gradient_checking_problem.reset(
        CreateGradientCheckingProblem(options, problem));
  }

  vector<double*> removed_parameter_blocks;
  pp->reduced_program.reset(CreateReducedProgram(program,
                                                 &removed_parameter_blocks,
                                                 &pp->fixed_cost,
                                                 &pp->error));
  if (pp->reduced_program.get() == NULL) {
    return false;
  }

  if (pp->reduced_program->NumParameterBlocks() == 0) {
    return true;
  }

  pp->inner_iteration_ordering = options.inner_iteration_ordering;
  pp->linear_solver_ordering = options.linear_solver_ordering;

  if (pp->linear_solver_ordering.get() == NULL) {
    pp->linear_solver_ordering.reset(
        CreateDefaultLinearSolverOrdering(*pp->reduced_program));
  } else {
    const int min_group_id = pp->linear_solver_ordering->MinNonZeroGroup();
    pp->linear_solver_ordering->Remove(removed_parameter_blocks);
    if (IsSchurType(options.linear_solver_type) &&
        min_group_id != pp->linear_solver_ordering->MinNonZeroGroup()) {
      AlternateLinearSolverAndPreconditionerForSchurTypeLinearSolver(&pp->options);
    }
  }

  if (!ReorderProgram()) {
    return false;
  }

  pp->evaluator.reset(
      CreateEvaluator(pp->options.num_threads,
                      pp->reduced_program.get(),
                      &pp->error));

  if (pp->evaluator.get() == NULL) {
    return false;
  }

  pp->jacobian.reset(
      pp->evaluator->CreateJacobian());

  pp->linear_solver.reset(CreateLinearSolver());

  if (options.use_inner_iterations) {
    if (pp->reduced_program()->NumParameterBlocks() == 1) {
      LOG(WARNING) << "Reduced problem only contains one parameter block."
                   << "Disabling inner iterations.";
    } else {
      pp->inner_iteration_minimizer.reset(
          CreateInnerIterationMinimizer());
      if (pp->inner_iteration_minimizer.get() == NULL) {
        return false;
      }
    }
  }

  ConfigureMinimizer(pp);
  return true;
}

ParameterBlockOrdering* TrustRegionPreprocessor::CreateDefaultLinearSolverOrdering(
    const Program& program) {
  linear_solver_ordering = new ParameterBlockOrdering;
  const vector<ParameterBlock*>& parameter_blocks = program.parameter_blocks();
  for (int i = 0; i < parameter_blocks.size(); ++i) {
    linear_solver_ordering->AddElementToGroup(
        const_cast<double*>(parameter_blocks[i]->user_state()), 0);
  }
}

bool TrustRegionPreprocessor::IsProgramValid(const Program& program,
                                             string* error) {
  return (program.ParameterBlocksAreFinite(error) &&
          program.IsFeasible(error));
}

void AlternateLinearSolverAndPreconditionerForSchurTypeLinearSolver() {
  if (!IsSchurType(options_.linear_solver_type)) {
    return;
  }

  options_.linear_solver_type = LinearSolver::LinearSolverForZeroEBlocks(
      options_.linear_solver_type);
  summary_->linear_solver_type_used = options_.linear_solver_type;

  string message;
  if (summary_->linear_solver_type_given == ITERATIVE_SCHUR) {
    options_.preconditioner_type = Preconditioner::PreconditionerForZeroEBlocks(
        options_.preconditoner_type);
    summary_->preconditioner_type_used = options_.preconditioner_type;

    message =
        StringPrintf(
            "No E blocks. Switching from %s(%s) to %s(%s).",
            LinearSolverTypeToString(summary_->linear_solver_type_given),
            PreconditionerTypeToString(summary_->preconditioner_type_given),
            LinearSolverTypeToString(summary_->linear_solver_type_used),
            PreconditionerTypeToString(summary_->preconditioner_type_used));
  } else {
    message =
        StringPrintf(
            "No E blocks. Switching from %s to %s.",
            LinearSolverTypeToString(summary_->linear_solver_type_given),
            LinearSolverTypeToString(summary_->linear_solver_type_used));
  }

  VLOG_IF(1, options->logging_type != SILENT) << message;
}

void CreateLinearSolver() {
  LinearSolver::Options linear_solver_options;
  linear_solver_options.min_num_iterations =
      options_.min_linear_solver_iterations;
  linear_solver_options.max_num_iterations =
      options_.max_linear_solver_iterations;
  linear_solver_options.type = options_.linear_solver_type;
  linear_solver_options.preconditioner_type = options_.preconditioner_type;
  linear_solver_options.visibility_clustering_type =
      options_.visibility_clustering_type;
  linear_solver_options.sparse_linear_algebra_library_type =
      options_.sparse_linear_algebra_library_type;
  linear_solver_options.dense_linear_algebra_library_type =
      options_.dense_linear_algebra_library_type;
  linear_solver_options.dynamic_sparsity = options_.dynamic_sparsity;

  // Ignore user's postordering preferences and force it to be true if
  // cholmod_camd is not available. This ensures that the linear
  // solver does not assume that a fill-reducing pre-ordering has been
  // done.
  linear_solver_options.use_postordering = options_.use_postordering;
  if (options_.linear_solver_type == SPARSE_SCHUR &&
      options_.sparse_linear_algebra_library_type == SUITE_SPARSE &&
      !SuiteSparse::IsConstrainedApproximateMinimumDegreeOrderingAvailable())
  {
    linear_solver_options.use_postordering = true;
  }

    linear_solver_options.num_threads = options_.num_linear_solver_threads;

    OrderingToGroupSizes(options_.linear_solver_ordering.get(),
                         &linear_solver_options.elimination_groups);
    // Schur type solvers, expect at least two elimination groups. If
    // there is only one elimination group, then CreateReducedProgram
    // guarantees that this group only contains e_blocks. Thus we add a
    // dummy elimination group with zero blocks in it.
    if (IsSchurType(linear_solver_options.type) &&
        linear_solver_options.elimination_groups.size() == 1) {
      linear_solver_options.elimination_groups.push_back(0);
    }

    linear_solver_.reset(LinearSolver::Create(linear_solver_options));
}

bool CreateEvaluator() {
  Evaluator::Options evaluator_options;
  evaluator_options.linear_solver_type = options.linear_solver_type;
  evaluator_options.num_eliminate_blocks = 0;
  if (IsSchurType(options_.linear_solver_type)) {
    evaluator_options.num_eliminate_blocks =
        linear_solver_ordering_->group_to_elements().begin()->second.size());
  evaluator_options.num_threads = options.num_threads;
  evaluator_options.dynamic_sparsity = options.dynamic_sparsity;
  evaluator_.reset(Evaluator::Create(evaluator_options,
                                     program_,
                                     &summary_->message));

  return evaluator_ != NULL;

}

bool ReorderProgram() {
  if (IsSchurType(options_.linear_solver_type) &&
      !ReorderProgramForSchurTypeLinearSolver(
          options_.linear_solver_type,
          options_.sparse_linear_algebra_library_type,
          problem_->parameter_map(),
          *linear_solver_ordering_,
          program_,
          &summary->message)) {
    return false;
  }

  if (options_.linear_solver_type == SPARSE_NORMAL_CHOLESKY &&
      !options_.dynamic_sparsity &&
      !ReorderProgramForSparseNormalCholesky(
          options_.sparse_linear_algebra_library_type,
          *linear_solver_ordering_,
          program_,
          &summary->message)) {
    return false;
  }

  return true;
}

CoordinateDescentMinimizer* TrustRegionMinimizer::CreateInnerIterationMinimizer() {
  if (options_.inner_iteration_ordering.get() == NULL) {
    inner_iteration_ordering_=
        CoordinateDescentMinimizer::CreateOrdering(program);
  } else {
    inner_iteration_ordering_ = options.inner_iteration_ordering;
    if (!CoordinateDescentMinimizer::IsOrderingValid(program,
                                                     *inner_iteration_ordering_,
                                                     &summary->message)) {

      return false;
    }
  }

  inner_iteration_minimizer_.reset(new CoordinateDescentMinimizer);
  if (!inner_iteration_minimizer->Init(program_,
                                       problem_->parameter_map(),
                                       inner_iteration_ordering_,
                                       &summary_->message)) {
    inner_iteration_minimizer_.reset(NULL);
    return false;
  }

  summary_->inner_iterations_used = true;
  summary_->inner_iteration_time_in_seconds = 0.0;
  return true;
}

void TrustRegionPreprocessor::SetupMinimizerOptions(PreprocessedProblem* pp) {
  SetupCommonMinimizerOptions(pp);

  minimizer_options.jacobian = jacobian.get();
  minimizer_options.inner_iteration_minimizer = inner_iteration_minimizer_.get();

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
  */
  return true;
}

}  // namespace internal
}  // namespace ceres
