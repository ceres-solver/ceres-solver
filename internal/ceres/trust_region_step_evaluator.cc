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

#include <algorithm>
#include "ceres/trust_region_step_evaluator.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

TrustRegionStepEvaluator::~TrustRegionStepEvaluator() {}

MonotonicStepEvaluator::MonotonicStepEvaluator(const double initial_cost)
    : current_cost_(initial_cost) {
}

MonotonicStepEvaluator::~MonotonicStepEvaluator() {}

double MonotonicStepEvaluator::StepQuality(
    const double cost,
    const double model_cost_change) const {
    return (current_cost_ - cost) / model_cost_change;
}

void MonotonicStepEvaluator::StepAccepted(const double cost,
                                          double model_cost_change) {
  CHECK_LT(cost, current_cost_);
  current_cost_ = cost;
}

TointNonMonotonicStepEvaluator::TointNonMonotonicStepEvaluator(
    const double initial_cost,
    const int max_consecutive_nonmonotonic_steps,
    const bool is_not_silent)
    : max_consecutive_nonmonotonic_steps_(max_consecutive_nonmonotonic_steps),
      is_not_silent_(is_not_silent),
      minimum_cost_(initial_cost),
      current_cost_(initial_cost),
      reference_cost_(initial_cost),
      candidate_cost_(initial_cost),
      accumulated_reference_model_cost_change_(0.0),
      accumulated_candidate_model_cost_change_(0.0),
      num_consecutive_nonmonotonic_steps_(0){
}

TointNonMonotonicStepEvaluator::~TointNonMonotonicStepEvaluator() {}

double TointNonMonotonicStepEvaluator::StepQuality(
    const double cost,
    const double model_cost_change) const {
    const double relative_decrease = (current_cost_ - cost) / model_cost_change;
    const double historical_relative_decrease =
        (reference_cost_ - cost) /
        (accumulated_reference_model_cost_change_ + model_cost_change);
    return std::max(relative_decrease, historical_relative_decrease);
}

void TointNonMonotonicStepEvaluator::StepAccepted(
    const double cost,
    const double model_cost_change) {
    current_cost_ = cost;
    accumulated_candidate_model_cost_change_ += model_cost_change;
    accumulated_reference_model_cost_change_ += model_cost_change;
    if (current_cost_ < minimum_cost_) {
      minimum_cost_ = current_cost_;
      candidate_cost_ = current_cost_;
      accumulated_candidate_model_cost_change_ = 0.0;
      num_consecutive_nonmonotonic_steps_ = 0;
      return;
    }

    ++num_consecutive_nonmonotonic_steps_;
    if (current_cost_ > candidate_cost_) {
      // The current iterate is has a higher cost than the
      // candidate iterate. Set the candidate to this point.
      VLOG_IF(2, is_not_silent_)
          << "Updating the candidate iterate to the current point.";
      candidate_cost_ = current_cost_;
      accumulated_candidate_model_cost_change_ = 0.0;
    }

    // At this point we have made too many non-monotonic steps and
    // we are going to reset the value of the reference iterate so
    // as to force the algorithm to descend.
    //
    // This is the case because the candidate iterate has a value
    // greater than minimum_cost but smaller than the reference
    // iterate.
    if (num_consecutive_nonmonotonic_steps_ ==
        max_consecutive_nonmonotonic_steps_) {
      VLOG_IF(2, is_not_silent_)
          << "Resetting the reference point to the candidate point";
      reference_cost_ = candidate_cost_;
      accumulated_reference_model_cost_change_ =
          accumulated_candidate_model_cost_change_;
    }
}


}  // namespace internal
}  // namespace ceres
