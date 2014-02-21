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
double ConstrainedSolver::EvaluateConstraintMaxNorm(experimental::ConstrainedProblem* problem) {
  map<ConstraintBlockId, CostFunction*>& constraints = *problem->mutable_constraints();
  Problem* unconstrained_problem = problem->mutable_problem();
  Vector residuals;
  double max_norm = 0.0;
  for (map<ConstraintBlockId, CostFunction*>::const_iterator it = constraints.begin();
       it != constraints.end();
       ++it) {
    AugmentedLagrangianCostFunction* cost_function =
            down_cast<AugmentedLagrangianCostFunction*>(it->second);
    vector<double*> parameter_blocks;
    unconstrained_problem
        ->GetParameterBlocksForResidualBlock(it->first, &parameter_blocks);
    const CostFunction* constraint = cost_function->constraint();
    if (constraint->num_residuals() > residuals.size()) {
      residuals.resize(constraint->num_residuals());
    }
    CHECK(constraint->Evaluate(&parameter_blocks[0], residuals.data(), NULL));
    max_norm = std::max(max_norm, residuals.lpNorm<Eigen::Infinity>());
  }
  return max_norm;
};

void ConstrainedSolver::Solve(const Solver::Options& original_options,
                              experimental::ConstrainedProblem* problem,
                              Solver::Summary* summary) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(summary);
  ceres::Solver::Options options(original_options);

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

  LOG(INFO) << "Initial constraint max norm: "
            << EvaluateConstraintMaxNorm(problem);

  while (true) {
    LOG(INFO) << "mu: " << mu << " omega: " << omega << " eta: " << eta;

    *summary = Solver::Summary();

    options.gradient_tolerance = omega;
    ceres::Solve(options, unconstrained_problem, summary);
    LOG(INFO) << summary->BriefReport();

    if (summary->termination_type != CONVERGENCE ||
        summary->termination_type != NO_CONVERGENCE) {
      break;
    }

    for (map<ConstraintBlockId, CostFunction*>::const_iterator it = constraints.begin();
         it != constraints.end();
         ++it) {
      AugmentedLagrangianCostFunction* cost_function =
          down_cast<AugmentedLagrangianCostFunction*>(it->second);
      cost_function->mutable_lambda()->setZero();
      cost_function->set_mu(mu);
    }

    const double constraint_max_norm = EvaluateConstraintMaxNorm(problem);
    LOG(INFO) << "Constraint max norm: " << constraint_max_norm
              << "eta: " << eta;

    if (constraint_max_norm < eta) {
      LOG(INFO) << "Updating lambda";
      for (map<ConstraintBlockId, CostFunction*>::iterator it = constraints.begin();
           it != constraints.end();
           ++it) {
        AugmentedLagrangianCostFunction* cost_function =
            down_cast<AugmentedLagrangianCostFunction*>(it->second);
        vector<double*> parameter_blocks;
        unconstrained_problem
            ->GetParameterBlocksForResidualBlock(it->first, &parameter_blocks);
        cost_function->UpdateLambda(parameter_blocks);
      }

      omega *= mu;
      eta *= std::pow(mu, 0.9);
    } else {
      LOG(INFO) << "Updating mu";
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
