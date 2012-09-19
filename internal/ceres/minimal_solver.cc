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
// Author: keir@google.com (Keir Mierle)

#include "ceres/minimal_solver.h"

#include <cstdio>
#include <iostream>  // NOLINT
#include <numeric>
#include "ceres/evaluator.h"
#include "ceres/linear_solver.h"
#include "ceres/minimizer.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/solver.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/wall_time.h"

namespace ceres {
namespace internal {

Solver::Summary MinimalSolver::Solve(const Solver::Options& options,
                                     Program* program,
                                     double* parameters) {
  Solver::Summary summary;
  summary.initial_cost = 0.0;
  summary.fixed_cost = 0.0;
  summary.final_cost = 0.0;
  string error;

  LinearSolver::Options linear_solver_options;
  linear_solver_options.type = DENSE_QR;
  scoped_ptr<LinearSolver>
      linear_solver(LinearSolver::Create(linear_solver_options));

  Evaluator::Options evaluator_options;
  evaluator_options.linear_solver_type = DENSE_QR;
  evaluator_options.num_eliminate_blocks = 0;
  evaluator_options.num_threads = 1;
  scoped_ptr<Evaluator> evaluator(Evaluator::Create(evaluator_options, program, &error));
  CHECK_NOTNULL(evaluator.get());

  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());
  CHECK_NOTNULL(jacobian.get());

  TrustRegionStrategy::Options trust_region_strategy_options;
  trust_region_strategy_options.linear_solver = linear_solver.get();
  scoped_ptr<TrustRegionStrategy> strategy(
      TrustRegionStrategy::Create(trust_region_strategy_options));
  CHECK_NOTNULL(strategy.get());

  Minimizer::Options minimizer_options(options);
  minimizer_options.evaluator = evaluator.get();
  minimizer_options.jacobian = jacobian.get();
  minimizer_options.trust_region_strategy = strategy.get();

  TrustRegionMinimizer minimizer;
  minimizer.Minimize(minimizer_options, parameters, &summary);
  return summary;
}


}  // namespace internal
}  // namespace ceres
