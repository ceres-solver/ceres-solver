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

#include "ceres/augmented_lagrangian_cost_function.h"
#include "ceres/constrained_solver.h"
#include "ceres/solver.h"
#include "glog/logging.h"
#include "ceres/casts.h"

namespace ceres {
namespace experimental {

void ConstrainedSolver::Solve(const Solver::Options& options,
                              experimental::ConstrainedProblem* problem,
                              Solver::Summary* summary) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(summary);
  map<ConstraintBlockId, CostFunction*>& constraints = *problem->mutable_constraints();
  Problem* unconstrained_problem = problem->mutable_problem();

  double mu = 0.1;
  double omega  = 0.1;
  double eta = 1.0;

  for (map<ConstraintBlockId, CostFunction*>::const_iterator it = constraints.begin();
       it != constraints.end();
       ++it) {
    AugmentedLagrangianCostFunction* cost_function =
        down_cast<AugmentedLagrangianCostFunction*>(it->second);
    cost_function->mutable_lambda()->setZero();
    cost_function->set_mu(mu);
  }

  // Check convergence
  while (true) {
    // Update convergence tolerance;
    ceres::Solve(options, unconstrained_problem, summary);
    // some error checking.
    bool constraints_are_reduced_sufficiently = false;
    // Evaluate constraints
    double scratch[1000];
    if (constraints_are_reduced_sufficiently) {
      // Update lagrange variables;
      for (map<ConstraintBlockId, CostFunction*>::iterator it = constraints.begin();
           it != constraints.end();
           ++it) {
        AugmentedLagrangianCostFunction* cost_function =
            down_cast<AugmentedLagrangianCostFunction*>(it->second);
        vector<double*> parameter_blocks;
        unconstrained_problem->GetParameterBlocksForResidualBlock(it->first, &parameter_blocks);
        cost_function->UpdateLambda(parameter_blocks, scratch);
      }

      omega *= mu;
      eta *= std::pow(mu, 0.9);
    } else {
      mu *= 0.1;
      for (map<ConstraintBlockId, CostFunction*>::iterator it = constraints.begin();
           it != constraints.end();
           ++it) {
        AugmentedLagrangianCostFunction* cost_function =
            down_cast<AugmentedLagrangianCostFunction*>(it->second);
        cost_function->set_mu(mu);
      }

      omega = mu;
      eta = 0.1258925 * std::pow(mu, 0.1);
    }
  };
}

}  // namespace experimental
}  // namespace ceres
