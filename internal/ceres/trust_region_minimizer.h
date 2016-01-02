// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
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

#ifndef CERES_INTERNAL_TRUST_REGION_MINIMIZER_H_
#define CERES_INTERNAL_TRUST_REGION_MINIMIZER_H_

#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/minimizer.h"
#include "ceres/solver.h"
#include "ceres/sparse_matrix.h"
#include "ceres/trust_region_step_evaluator.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

// Generic trust region minimization algorithm.
//
// For example usage, see SolverImpl::Minimize.
class TrustRegionMinimizer : public Minimizer {
 public:
  ~TrustRegionMinimizer() {}
  virtual void Minimize(const Minimizer::Options& options,
                        double* parameters,
                        Solver::Summary* solver_summary);

 private:
  void Init(const Minimizer::Options& options,
            double* parameters,
            Solver::Summary* solver_summary);
  bool IterationZero();
  bool FinalizeIterationAndCheckIfMinimizerCanContinue();
  bool ComputeTrustRegionStep();

  bool EvaluateGradientAndJacobian();
  void ComputeCandidatePointAndEvaluateCost();

  void DoLineSearch(const Vector& x,
                    const Vector& gradient,
                    const double cost,
                    Vector* delta);
  void DoInnerIterationsIfNeeded();

  bool ParameterToleranceReached();
  bool FunctionToleranceReached();
  bool GradientToleranceReached();
  bool MaxSolverTimeReached();
  bool MaxSolverIterationsReached();
  bool MinTrustRegionRadiusReached();

  bool IsStepSuccessful();
  void HandleUnsuccessfulStep();
  bool HandleSuccessfulStep();
  bool HandleInvalidStep();

  Minimizer::Options options_;

  double* parameters_;
  Solver::Summary* solver_summary_;
  Evaluator* evaluator_;
  SparseMatrix* jacobian_;
  TrustRegionStrategy* strategy_;
  scoped_ptr<TrustRegionStepEvaluator> step_evaluator_;

  IterationSummary iteration_summary_;

  int num_parameters_;
  int num_effective_parameters_;
  int num_residuals_;

  Vector delta_;
  Vector gradient_;
  Vector inner_iteration_x_;
  Vector model_residuals_;
  Vector negative_gradient_;
  Vector projected_gradient_step_;
  Vector residuals_;
  Vector scale_;
  Vector trust_region_step_;
  Vector x_;
  Vector candidate_x_;
  bool is_not_silent_;
  bool inner_iterations_are_enabled_;
  bool inner_iterations_were_useful_;

  double x_norm_;
  double x_cost_;
  double minimum_cost_;
  double model_cost_change_;
  double candidate_cost_;

  double start_time_;
  double iteration_start_time_;
  int num_consecutive_invalid_steps_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_TRUST_REGION_MINIMIZER_H_
