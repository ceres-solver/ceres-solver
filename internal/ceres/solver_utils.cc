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

#include "ceres/solver_utils.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/map_util.h"
#include "ceres/program.h"
#include "ceres/summary_utils.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

Program* CreateReducedProgram(const Program& program,
                              vector<double*>* removed_parameter_blocks,
                              double* fixed_cost,
                              string* error) {
  scoped_ptr<Program> reduced_program(new Program(program));
  if (!reduced_program->RemoveFixedBlocks(removed_parameter_blocks,
                                          fixed_cost,
                                          error)) {
    return NULL;
  }

  reduced_program->SetParameterOffsetsAndIndex();
  return reduced_program.release();
}


void Finish(const map<string, double>& evaluator_time_statistics,
            const map<string, double>& linear_solver_time_statistics,
            Program* program,
            Solver::Summary* summary) {
  double post_process_start_time = WallTimeInSeconds();
  SetSummaryFinalCost(summary);

  // Ensure the program state is set to the user parameters on the way
  // out.
  program->SetParameterBlockStatePtrsToUserStatePtrs();
  program->SetParameterOffsetsAndIndex();

  summary->linear_solver_time_in_seconds =
      FindWithDefault(linear_solver_time_statistics,
                      "LinearSolver::Solve",
                      0.0);

  summary->residual_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Residual", 0.0);
  summary->jacobian_evaluation_time_in_seconds =
      FindWithDefault(evaluator_time_statistics, "Evaluator::Jacobian", 0.0);

  summary->postprocessor_time_in_seconds =
      WallTimeInSeconds() - post_process_start_time;
}

}  // namespace internal
}  // namespace ceres
