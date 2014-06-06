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

#include "ceres/line_search_preprocessor.h"

#include <numeric>
#include <string>
#include "ceres/callbacks.h"
#include "ceres/evaluator.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/map_util.h"
#include "ceres/minimizer.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

LineSearchPreprocessor::~LineSearchPreprocessor() {
}

bool LineSearchPreprocessor::Preprocess(const Solver::Options& options,
                                        ProblemImpl* problem,
                                        PreprocessedProblem* preprocessed_problem) {
  CHECK_NOTNULL(preprocessed_problem);
  preprocessed_problem->options = options;
  Program* program = problem->mutable_program();
  program->SetParameterBlockStatePtrsToUserStatePtrs();

  ChangeNumThreadsIfNeeded(&preprocessed_problem->options);

  if (!IsProgramValid(program, &preprocessed_problem->error)) {
    return false;
  }

  if (options.check_gradients) {
    preprocessed_problem->gradient_checking_problem.reset(
        CreateGradientCheckingProblem(options, problem));
  }

  preprocessed_problem->reduced_program.reset(
      CreateReducedProgram(program,
                           &preprocessed_problem->fixed_cost,
                           &preprocessed_problem->error));
  if (preprocessed_problem->reduced_program.get() == NULL) {
    return false;
  }

  preprocessed_problem->evaluator.reset(
      CreateEvaluator(preprocessed_problem->options.num_threads,
                      preprocessed_problem->reduced_program.get(),
                      &preprocessed_problem->error));
  if (preprocessed_problem->evaluator.get() == NULL) {
    return false;
  }

  ConfigureMinimizer(preprocessed_problem);
  return true;
}

bool LineSearchPreprocessor::IsProgramValid(const Program* program,
                                            string* error) const {
  if (program->IsBoundsConstrained()) {
    *error =  "LINE_SEARCH Minimizer does not support bounds.";
    return false;
  }
  return program->ParameterBlocksAreFinite(error);
}

ProblemImpl* LineSearchPreprocessor::CreateGradientCheckingProblem(
    const Solver::Options& options,
    ProblemImpl* problem) const {
  ProblemImpl* gradient_checking_problem =
      CHECK_NOTNULL(CreateGradientCheckingProblemImpl(
                        problem,
                        options.numeric_derivative_relative_step_size,
                        options.gradient_check_relative_precision));
  gradient_checking_problem
      ->mutable_program()
      ->SetParameterBlockStatePtrsToUserStatePtrs();
  return gradient_checking_problem;
}

void LineSearchPreprocessor::ChangeNumThreadsIfNeeded(
    Solver::Options* options) const {
#ifndef CERES_USE_OPENMP
    if (options->num_threads > 1) {
      LOG(WARNING)
          << "OpenMP support is not compiled into this binary; "
          << "only options.num_threads = 1 is supported. Switching "
          << "to single threaded mode.";
      options->num_threads = 1;
    }
#endif  // CERES_USE_OPENMP
}

Program* LineSearchPreprocessor::CreateReducedProgram(Program* program,
                                                      double* fixed_cost,
                                                      string* error) const {
  vector<double*> removed_parameter_blocks;
  Program* reduced_program =
      program->CreateReducedProgram(&removed_parameter_blocks,
                                    fixed_cost,
                                    error);
  if (reduced_program != NULL) {
    reduced_program->SetParameterOffsetsAndIndex();
  }
  return reduced_program;
}

Evaluator* LineSearchPreprocessor::CreateEvaluator(const int num_threads,
                                                   Program* program,
                                                   string* error) const {
  Evaluator::Options evaluator_options;
  // This ensures that we get a Block Jacobian Evaluator without any
  // requirement on orderings.
  evaluator_options.linear_solver_type = CGNR;
  evaluator_options.num_eliminate_blocks = 0;
  evaluator_options.num_threads = num_threads;
  return Evaluator::Create(evaluator_options, program, error);
}

void LineSearchPreprocessor::ConfigureMinimizer(
    PreprocessedProblem* preprocessed_problem) const {
  const Solver::Options& options = preprocessed_problem->options;
  Program* program = preprocessed_problem->reduced_program.get();

  preprocessed_problem->reduced_parameters.resize(program->NumParameters());
  double* reduced_parameters = preprocessed_problem->reduced_parameters.data();
  program->ParameterBlocksToStateVector(reduced_parameters);
  Minimizer::Options& minimizer_options =
      preprocessed_problem->minimizer_options;

  minimizer_options = Minimizer::Options(options);
  minimizer_options.evaluator = preprocessed_problem->evaluator.get();
  preprocessed_problem->logging_callback.reset(
      new LoggingCallback(LINE_SEARCH, options.minimizer_progress_to_stdout));

  if (options.logging_type != SILENT) {
    minimizer_options.callbacks.insert(
        minimizer_options.callbacks.begin(),
        preprocessed_problem->logging_callback.get());
  }

  preprocessed_problem->state_updating_callback.reset(
      new StateUpdatingCallback(program, reduced_parameters));

  if (options.update_state_every_iteration) {
    // This must get pushed to the front of the callbacks so that it
    // is run before any of the user callbacks.
    minimizer_options.callbacks.insert(
        minimizer_options.callbacks.begin(),
        preprocessed_problem->state_updating_callback.get());
  }
}

}  // namespace internal
}  // namespace ceres
