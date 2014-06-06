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
    SummarizeInputs();
  }

  bool Preprocess() {
    ChangeOptionsIfNeeded();

    if (!IsProblemValid()) {
      return false;
    }

    if (options_.check_gradients) {
      CreateGradientCheckingProblem();
    }

    if (!CreateReducedProgram()) {
      return false;
    }

    if (!CreateEvaluator()) {
      return false;
    }

    return true;
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
    summary_->minimizer_type = LINE_SEARCH;
    summary_->line_search_direction_type = options_.line_search_direction_type;
    summary_->max_lbfgs_rank = options_.max_lbfgs_rank;
    summary_->line_search_type = options_.line_search_type;
    summary_->line_search_interpolation_type =
        options_.line_search_interpolation_type;
    summary_->nonlinear_conjugate_gradient_type =
        options_.nonlinear_conjugate_gradient_type;
    summary_->num_threads_given = options_.num_threads;
  }

  bool IsProblemValid() {
    if (program_->IsBoundsConstrained()) {
      summary_->message =  "LINE_SEARCH Minimizer does not support bounds.";
      return false;
    }
    return program_->ParameterBlocksAreFinite(&summary_->message);
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

  void ChangeOptionsIfNeeded() {
#ifndef CERES_USE_OPENMP
    if (options_.num_threads > 1) {
      LOG(WARNING)
          << "OpenMP support is not compiled into this binary; "
          << "only options.num_threads = 1 is supported. Switching "
          << "to single threaded mode.";
      options_.num_threads = 1;
    }
#endif  // CERES_USE_OPENMP
  }

  bool CreateReducedProgram() {
    vector<double*> removed_parameter_blocks;
    reduced_program_.reset(
        program_->CreateReducedProgram(&removed_parameter_blocks,
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

    if (reduced_program_->NumParameterBlocks() == 0) {
      summary_->message =
          "Function tolerance reached. "
          "No non-constant parameter blocks found.";
      summary_->termination_type = CONVERGENCE;
      summary_->initial_cost = summary_->fixed_cost;
      return false;
    }

    program_ = reduced_program_.get();
    program_->SetParameterOffsetsAndIndex();
    return true;
  }

  bool CreateEvaluator() {
    Evaluator::Options evaluator_options;
    // This ensures that we get a Block Jacobian Evaluator without any
    // requirement on orderings.
    evaluator_options.linear_solver_type = CGNR;
    evaluator_options.num_eliminate_blocks = 0;
    evaluator_options.num_threads = options_.num_threads;
    evaluator_.reset(Evaluator::Create(evaluator_options,
                                       program_,
                                       &summary_->message));
    if (evaluator_ == NULL) {
      LOG(ERROR) << "Terminating: " << summary_->message;
      return false;
    }

    return true;
  }

  void Minimize() {
    Minimizer::Options minimizer_options(options_);

    // The optimizer works on contiguous parameter vectors; allocate
    // some and collect the discontiguous parameters into the
    // continuous parameter vector.
    Vector parameters(reduced_program_->NumParameters());
    reduced_program_->ParameterBlocksToStateVector(parameters.data());

    LoggingCallback logging_callback(LINE_SEARCH,
                                     options_.minimizer_progress_to_stdout);
    if (options_.logging_type != SILENT) {
      minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                         &logging_callback);
    }

    StateUpdatingCallback updating_callback(program_, parameters.data());
    if (options_.update_state_every_iteration) {
      // This must get pushed to the front of the callbacks so that it
      // is run before any of the user callbacks.
      minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                         &updating_callback);
    }

    minimizer_options.evaluator = evaluator_.get();
    LineSearchMinimizer minimizer;
    double minimizer_start_time = WallTimeInSeconds();
    minimizer.Minimize(minimizer_options, parameters.data(), summary_);

    // If the user aborted mid-optimization or the optimization
    // terminated because of a numerical failure, then do not update
    // user state.
    if (summary_->IsSolutionUsable()) {
      reduced_program_->StateVectorToParameterBlocks(parameters.data());
      reduced_program_->CopyParameterBlockStateToUserState();
    }

    summary_->minimizer_time_in_seconds =
        WallTimeInSeconds() - minimizer_start_time;
  }

  void Finish() {
    const double post_process_start_time = WallTimeInSeconds();
    summary_->num_threads_used = options_.num_threads;
    if (reduced_program_ != NULL) {
      SummarizeReducedProgram(*reduced_program_, summary_);
    }

    if (evaluator_ != NULL) {
      summary_->residual_evaluation_time_in_seconds =
          FindWithDefault(evaluator_->TimeStatistics(),
                          "Evaluator::Residual",
                          0.0);
      summary_->jacobian_evaluation_time_in_seconds =
          FindWithDefault(evaluator_->TimeStatistics(),
                          "Evaluator::Jacobian",
                          0.0);
      summary_->postprocessor_time_in_seconds =
          WallTimeInSeconds() - post_process_start_time;
    }

    SetSummaryFinalCost(summary_);
    // Ensure the program state is set to the user parameters on the way
    // out.
    Program* program = given_problem_->mutable_program();
    program->SetParameterBlockStatePtrsToUserStatePtrs();
    program->SetParameterOffsetsAndIndex();
  }

  Solver::Options options_;
  ProblemImpl* given_problem_;
  Solver::Summary* summary_;
  ProblemImpl* problem_;
  Program* program_;
  scoped_ptr<Program> reduced_program_;
  scoped_ptr<ProblemImpl> gradient_checking_problem_;
  scoped_ptr<Evaluator> evaluator_;
};

}  // namespace

void SolveUsingLineSearchMinimizer(const Solver::Options& options,
                                   ProblemImpl* problem_impl,
                                   Solver::Summary* summary) {
  LineSearchSolver solver;
  solver.Solve(options, problem_impl, summary);
}

}  // namespace internal
}  // namespace ceres
