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
// Author: sameragarwal@google.com (Sameer Agarwal)

#include "ceres/preprocessor.h"
#include "ceres/gradient_checking_cost_function.h"
#include "ceres/problem_impl.h"
#include "ceres/solver.h"
#include "ceres/callbacks.h"

namespace ceres {
namespace internal {

Preprocessor::~Preprocessor() {
}

ProblemImpl* CreateGradientCheckingProblem(const Solver::Options& options,
                                           ProblemImpl* problem) {
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

void ChangeNumThreadsIfNeeded(Solver::Options* options) {
#ifndef CERES_USE_OPENMP
  if (options->num_threads > 1) {
    LOG(WARNING)
        << "OpenMP support is not compiled into this binary; "
        << "only options.num_threads = 1 is supported. Switching "
        << "to single threaded mode.";
    options->num_threads = 1;
  }

  if (options->minimizer_type == TRUST_REGION) {
    if (options->num_linear_solver_threads > 1) {
      LOG(WARNING)
          << "OpenMP support is not compiled into this binary; "
          << "only options.num_linear_solver_threads=1 is supported. Switching "
          << "to single threaded mode.";
      options->num_linear_solver_threads = 1;
    }
  }
#endif  // CERES_USE_OPENMP
}

Program* CreateReducedProgram(Program* program,
                              vector<double*>* removed_parameter_blocks,
                              double* fixed_cost,
                              string* error) {
  Program* reduced_program =
      program->CreateReducedProgram(removed_parameter_blocks,
                                    fixed_cost,
                                    error);
  if (reduced_program != NULL) {
    reduced_program->SetParameterOffsetsAndIndex();
  }
  return reduced_program;
}

void SetupCommonMinimizerOptions(PreprocessedProblem* pp) {
  const Solver::Options& options = pp->options;
  Program* program = pp->reduced_program.get();

  pp->reduced_parameters.resize(program->NumParameters());
  double* reduced_parameters = pp->reduced_parameters.data();
  program->ParameterBlocksToStateVector(reduced_parameters);
  Minimizer::Options& minimizer_options = pp->minimizer_options;
  minimizer_options = Minimizer::Options(options);
  minimizer_options.evaluator = pp->evaluator.get();
  pp->logging_callback.reset(
      new LoggingCallback(options.minimizer_type,
                          options.minimizer_progress_to_stdout));

  if (options.logging_type != SILENT) {
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       pp->logging_callback.get());
  }

  pp->state_updating_callback.reset(
      new StateUpdatingCallback(program, reduced_parameters));

  if (options.update_state_every_iteration) {
    // This must get pushed to the front of the callbacks so that it
    // is run before any of the user callbacks.
    minimizer_options.callbacks.insert(minimizer_options.callbacks.begin(),
                                       pp->state_updating_callback.get());
  }
}

}  // namespace internal
}  // namespace ceres
