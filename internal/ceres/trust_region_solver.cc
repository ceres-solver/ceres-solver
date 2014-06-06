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

#include "ceres/trust_region_solver.h"

#include <numeric>
#include <string>
#include "ceres/callbacks.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/map_util.h"
#include "ceres/minimizer.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/summary_utils.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {
namespace {

// A class to preprocess a given optimization problem and solve it
// using the trust region minimizer.
//
// This is structured using a class with a single public method as
// there is substantial amount of state that needs to be managed and
// mutated before the minimizer can be called. Having this state be
// class members makes the structure of the code much simpler.
class LineSearchSolver {
 public:
  void Solve(const Solver::Options& options,
             ProblemImpl* problem,
             Solver::Summary* summary) {
    CHECK_NOTNULL(problem);
    CHECK_NOTNULL(summary);
    const double solver_start_time = WallTimeInSeconds();

    Init(options, problem, summary);

    const bool preprocessor_status = Preprocess();
    summary_->preprocessor_time_in_seconds =
        WallTimeInSeconds() - solver_start_time;

    if (preprocessor_status) {
      Minimize();
    } else {
      LOG(ERROR) << "Terminating: " << summary_->message;
    }

    Finish();
  };

 private:
  void Init(const Solver::Options& options,
            ProblemImpl* problem,
            Solver::Summary* summary) {
    options_ = options;
    given_problem_ = problem;
    problem_ = problem;
    summary_ = summary;
    program_ = problem_->mutable_program();
    program_->SetParameterBlockStatePtrsToUserStatePtrs();

    inner_iteration_ordering_ = options.inner_iteration_ordering;
    linear_solver_ordering_ = options.linear_solver_ordering;
    reduced_program_.reset(NULL);
    gradient_checking_problem_.reset(NULL);
    evaluator_.reset(NULL);
  }

  void SummarizeInputs() {
    VLOG(2) << "Initial problem: "
            << program_->NumParameterBlocks()
            << " parameter blocks, "
            << program_->NumParameters()
            << " parameters,  "
            << program_->NumResidualBlocks()
            << " residual blocks, "
            << program_->NumResiduals()
            << " residuals.";

    SummarizeGivenProgram(*program_, summary_);
    summary_->minimizer_type = TRUST_REGION;
    summary_->num_threads_given = options_.num_threads;
    summary_->num_linear_solver_threads_given =
        options_.num_linear_solver_threads;
    summary_->preconditioner_type_given = options_.preconditioner_type;
    summary_->linear_solver_type_given = options_.linear_solver_type;
    OrderingToGroupSizes(options.linear_solver_ordering.get(),
                         &(summary_->linear_solver_ordering_given));
    OrderingToGroupSizes(options.inner_iteration_ordering.get(),
                         &(summary_->inner_iteration_ordering_given));
  }

  void CreateDefaultLinearSolverOrdering() {
    linear_solver_ordering_.reset(new ParameterBlockOrdering);
    const vector<ParameterBlock*>& parameter_blocks =
        program_->parameter_blocks();
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      linear_solver_ordering_->AddElementToGroup(
          parameter_blocks[i]->mutable_user_state(), 0);
    }
  }

  bool Preprocess() {
    SummarizeInputs()
    ChangeNumThreadsIfNeeded();

    if (!IsProblemValid()) {
      return false;
    }

    if (options_.check_gradients) {
      CreateGradientCheckingProblem();
    }

    if (!CreateReducedProgram()) {
      return false;
    }

    if (linear_solver_ordering_ == NULL) {
      // If a non-null ordering is provided by the user, then use it,
      // otherwise create a default ordering in which all parameter
      // blocks are in the same elimination group so that the linear
      // solver has complete freedom to do the best job that it can.
      CreateDefaultLinearSolverOrdering();
    } else {
      const int min_group_id = linear_solver_ordering_->MinNonZeroGroup();
      linear_solver_ordering_->Remove(removed_parameter_blocks_);
      if (IsSchurType(options_.linear_solver_type) &&
        min_group_id != linear_solver_ordering_->MinNonZeroGroup()) {
        AlternateLinearSolverAndPreconditionerForSchurTypeLinearSolver();
      }
    }

    if (!ReorderProgram()) {
      return false;
    }

    if (!CreateEvaluator()) {
      return false;
    }

    CreateLinearSolver();

    if (options_.use_inner_iterations) {
      if (program_->NumParameterBlocks() > 1) {
        if (!CreateInnerIterationMinimizer()) {
          return false;
        }
      } else {
        LOG(WARNING) << "Reduced problem only contains one parameter block."
                     << "Disabling inner iterations.";
      }
    }

    return true;
  }

  bool IsProblemValid() {
    if (!program_->ParameterBlocksAreFinite(&summary->message)) {
      LOG(ERROR) << "Terminating: " << summary->message;
      return false;
    }

    if (!program_->IsFeasible(&summary->message)) {
      LOG(ERROR) << "Terminating: " << summary->message;
      return false;
    }

    return true;
  }

  void CreateGradientCheckingProblem() {
    VLOG(2) << "Checking gradients.";
    gradient_checking_problem_.reset(
        CreateGradientCheckingProblemImpl(
            problem_,
            options_.numeric_derivative_relative_step_size,
            options_.gradient_check_relative_precision));
    problem_ = gradient_checking_problem_.get();
    program_ = problem_->mutable_program();
    program_->SetParameterBlockStatePtrsToUserStatePtrs();
  }

  void ChangeNumThreadsIfNeeded() {
#ifndef CERES_USE_OPENMP
    if (options_.num_threads > 1) {
      LOG(WARNING)
          << "OpenMP support is not compiled into this binary; "
          << "only options.num_threads = 1 is supported. Switching "
          << "to single threaded mode.";
      options_.num_threads = 1;
    }

    if (options_.num_linear_solver_threads > 1) {
      LOG(WARNING)
          << "OpenMP support is not compiled into this binary; "
          << "only options.num_linear_solver_threads=1 is supported. Switching "
          << "to single threaded mode.";
      options_.num_linear_solver_threads = 1;
    }
#endif  // CERES_USE_OPENMP
  }

  void TrustRegionPreprocessor::AlternateLinearSolverAndPreconditionerForSchurTypeLinearSolver(
      Solver::Options* options) const {
    if (!IsSchurType(options->linear_solver_type)) {
      return;
    }

    const LinearSolverType linear_solver_type_given = options->linear_solver_type;
    const PreconditionerType preconditoner_type_given = options->preconditoner_type;
    options->linear_solver_type = LinearSolver::LinearSolverForZeroEBlocks(
        linear_solver_type_given)
    string message;
    if (linear_solver_type_given == ITERATIVE_SCHUR) {
      options->preconditioner_type = Preconditioner::PreconditionerForZeroEBlocks(
          options->preconditoner_type);
      message =
          StringPrintf(
              "No E blocks. Switching from %s(%s) to %s(%s).",
              LinearSolverTypeToString(linear_solver_type_given),
              PreconditionerTypeToString(preconditioner_type_given),
              LinearSolverTypeToString(options->linear_solver_type),
              PreconditionerTypeToString(options->preconditioner_type));
    } else {
      message =
          StringPrintf(
              "No E blocks. Switching from %s to %s.",
              LinearSolverTypeToString(linear_solver_type_given),
              LinearSolverTypeToString(options->linear_solver_type));
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

  bool CreateReducedProgram() {
    removed_parameter_blocks_.clear();
    reduced_program_.reset(
        program_->CreateReducedProgram(&removed_parameter_blocks_,
                                       &summary_->fixed_cost,
                                       &summary_->message));

    if (reduced_program_.get() == NULL) {
      LOG(ERROR) << "Terminating: " << summary_->message;
      return false;
    }

    VLOG(2) << "Reduced problem: "
            << reduced_program_->NumParameterBlocks()
            << " parameter blocks, "
            << reduced_program_->NumParameters()
            << " parameters,  "
            << reduced_program_->NumResidualBlocks()
            << " residual blocks, "
            << reduced_program_->NumResiduals()
            << " residuals.";

    program_ = reduced_program_.get();
    if (program_->NumParameterBlocks() == 0) {
      summary_->initial_cost = summary_->fixed_cost;
      summary_->message =
          "Function tolerance reached. "
          "No non-constant parameter blocks found.";
      summary_->termination_type = CONVERGENCE;
      // TODO(sameeragarwal): Should we be logging convergence here.?
      return false;
    }

    program_->SetParameterOffsetsAndIndex();
    return true;
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

  bool CreateInnerIterationMinimizer() {
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
    OrderingToGroupSizes(*inner_iteration_ordering_,
                         &(summary->inner_iteration_ordering_used));
    return true;
  }

  void Minimize() {
    Minimizer::Options minimizer_options(options);
    minimizer_options.is_constrained = program_->IsBoundsConstrained();

    // The optimizer works on contiguous parameter vectors; allocate
    // some.
    Vector parameters(program_->NumParameters());

    // Collect the discontiguous parameters into a contiguous state
    // vector.
    program_->ParameterBlocksToStateVector(parameters.data());

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

    TrustRegionMinimizer minimizer;
    double minimizer_start_time = WallTimeInSeconds();
    minimizer.Minimize(minimizer_options, parameters.data(), summary);

    // If the user aborted mid-optimization or the optimization
    // terminated because of a numerical failure, then do not update
    // user state.
    if (summary_->termination_type != USER_FAILURE &&
        summary_->termination_type != FAILURE) {
      program_->StateVectorToParameterBlocks(parameters.data());
      program_->CopyParameterBlockStateToUserState();
    }

    summary_->minimizer_time_in_seconds =
        WallTimeInSeconds() - minimizer_start_time;
  }

  void Finish() {
    double post_process_start_time = WallTimeInSeconds();
    summary_->num_threads_used = options_.num_threads;
    summary_->num_linear_solver_threads_used = options_.num_linear_solver_threads;

    SetSummaryFinalCost(summary);

    if (linear_solver_ != NULL) {
      const map<string, double>& linear_solver_time_statistics =
          linear_solver_->TimeStatistics();
      summary_->linear_solver_time_in_seconds =
          FindWithDefault(linear_solver_time_statistics,
                          "LinearSolver::Solve",
                        0.0);
    }

    if (evaluator_ != NULL) {
      const map<string, double>& evaluator_time_statistics =
        evaluator_->TimeStatistics();
      summary_->residual_evaluation_time_in_seconds =
          FindWithDefault(evaluator_time_statistics, "Evaluator::Residual", 0.0);
      summary_->jacobian_evaluation_time_in_seconds =
          FindWithDefault(evaluator_time_statistics, "Evaluator::Jacobian", 0.0);
    }

    Program* program = given_problem_->mutable_program();
    // Ensure the program state is set to the user parameters on the way
    // out.
    program->SetParameterBlockStatePtrsToUserStatePtrs();
    program->SetParameterOffsetsAndIndex();

    // Stick a fork in it, we're done.
    summary_->postprocessor_time_in_seconds =
        WallTimeInSeconds() - post_process_start_time;
  }

  Solver::Options options_;
  ProblemImpl* given_problem_;
  Solver::Summary* summary_;
  ProblemImpl* problem_;
  Program* program_;
  vector<double*> removed_parameter_blocks_;
  shared_ptr<ParameterBlockOrdering> linear_solver_ordering_;
  shared_ptr<ParameterBlockOrdering> inner_iteration_ordering_;
  scoped_ptr<Program> reduced_program_;
  scoped_ptr<ProblemImpl> gradient_checking_problem_;
  scoped_ptr<Evaluator> evaluator_;
  scoped_ptr<LinearSolver> linear_solver_;
  scoped_ptr<CoordinateDescentMinimizer> inner_iteration_minimizer_;
};

}  // namespace

void SolveUsingTrustRegionMinimizer(const Solver::Options& options,
                                    ProblemImpl* problem_impl,
                                    Solver::Summary* summary) {
  TrustRegionSolver solver;
  solver.Solve(options, problem_impl, summary);
}

}  // namespace internal
}  // namespace ceres
