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

#ifndef CERES_INTERNAL_TRUST_REGION_STEP_EVALUATOR_H_
#define CERES_INTERNAL_TRUST_REGION_STEP_EVALUATOR_H_

namespace ceres {
namespace internal {

class TrustRegionStepEvaluator {
 public:
  virtual ~TrustRegionStepEvaluator();
  virtual double StepQuality(double cost, double model_cost_change) const = 0;
  virtual void StepAccepted(double cost, double model_cost_change) = 0;
};

class MonotonicStepEvaluator : public TrustRegionStepEvaluator {
 public:
  MonotonicStepEvaluator(double initial_cost);
  virtual ~MonotonicStepEvaluator();
  virtual double StepQuality(double cost, double model_cost_change) const;
  virtual void StepAccepted(double cost, double model_cost_change);

 private:
  double current_cost_;
};

class TointNonMonotonicStepEvaluator : public TrustRegionStepEvaluator {
 public:
  TointNonMonotonicStepEvaluator(double initial_cost,
                                 int max_consecutive_nonmonotonic_steps,
                                 bool is_not_silent);
  virtual ~TointNonMonotonicStepEvaluator();
  virtual double StepQuality(double cost, double model_cost_change) const;
  virtual void StepAccepted(double cost, double model_cost_change);

 private:
  const int max_consecutive_nonmonotonic_steps_;
  const bool is_not_silent_;
  double minimum_cost_;
  double current_cost_;
  double reference_cost_;
  double candidate_cost_;
  double accumulated_reference_model_cost_change_;
  double accumulated_candidate_model_cost_change_;
  int num_consecutive_nonmonotonic_steps_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_TRUST_REGION_STEP_EVALUATOR_H_
