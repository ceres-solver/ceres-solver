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
#include "ceres/evaluator.h"
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

  vector<double*> unused;
  pp->reduced_program.reset(CreateReducedProgram(program,
                                                 &unused,
                                                 &pp->fixed_cost,
                                                 &pp->error));
  if (pp->reduced_program.get() == NULL) {
    return false;
  }
  if (pp->reduced_program->NumParameterBlocks() == 0) {
    return true;
  }

  pp->evaluator.reset(CreateEvaluator(pp->options.num_threads,
                                      pp->reduced_program.get(),
                                      &pp->error));
  if (pp->evaluator.get() == NULL) {
    return false;
  }

  SetupCommonMinimizerOptions(pp);
  return true;
}

bool LineSearchPreprocessor::IsProgramValid(const Program& program,
                                            string* error) const {
  if (program.IsBoundsConstrained()) {
    *error =  "LINE_SEARCH Minimizer does not support bounds.";
    return false;
  }
  return program.ParameterBlocksAreFinite(error);
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

}  // namespace internal
}  // namespace ceres
