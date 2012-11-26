// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include "ceres/line_search.h"

#include <glog/logging.h>
#include "ceres/fpclassify.h"
#include "ceres/evaluator.h"
#include "ceres/internal/eigen.h"
#include "ceres/polynomial.h"


namespace ceres {
namespace internal {
namespace {
inline FunctionSample ValueSample(const double x, const double value) {
  FunctionSample sample;
  sample.x = x;
  sample.value = value;
  sample.value_is_valid = true;
  return sample;
};

inline FunctionSample ValueAndGradientSample(const double x,
                                      const double value,
                                      const double gradient) {
  FunctionSample sample;
  sample.x = x;
  sample.value = value;
  sample.gradient = gradient;
  sample.value_is_valid = true;
  sample.gradient_is_valid = true;
  return sample;
};

}  // namespace

LineSearchEvaluator::LineSearchEvaluator(Evaluator* evaluator)
    : evaluator_(evaluator),
      position_(evaluator->NumParameters()),
      direction_(evaluator->NumEffectiveParameters()),
      evaluation_point_(evaluator->NumParameters()),
      scaled_direction_(evaluator->NumEffectiveParameters()),
      gradient_(evaluator->NumEffectiveParameters()) {
}

void LineSearchEvaluator::Init(const Vector& position,
                               const Vector& direction) {
  position_ = position;
  direction_ = direction;
}

bool LineSearchEvaluator::Evaluate(const double x, double* f, double* g) {
  scaled_direction_ = x * direction_;
  if (evaluator_->Plus(position_.data(),
                      scaled_direction_.data(),
                      evaluation_point_.data()) &&
      evaluator_->Evaluate(evaluation_point_.data(),
                          f,
                          NULL,
                          gradient_.data(), NULL)) {
    *g = direction_.dot(gradient_);
    return IsFinite(*f) && IsFinite(*g);
  }
  return false;
}

void ArmijoLineSearch::Search(LineSearch::Options& options,
                              LineSearch::EvaluatorBase* evaluator,
                              double initial_step_size,
                              const double cost_reference,
                              Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();

  double cost_0 = 0.0;
  double gradient_0 = 0.0;
  summary->num_evaluations = 1;
  if (!evaluator->Evaluate(0.0, &cost_0, &gradient_0)) {
    LOG(WARNING) << "Line search failed. "
                 << "Evaluation at the initial point failed.";
    return;
  }

  double cost_prev = 0.0;
  double step_size_prev = 0.0;
  double gradient_prev = 0.0;
  bool x_prev_is_valid = false;

  double cost_new = 0.0;
  double step_size_new = initial_step_size;
  double gradient_new = 0.0;
  bool x_new_is_valid = false;

  summary->num_evaluations += 1;
  x_new_is_valid = evaluator->Evaluate(step_size_new, &cost_new, &gradient_new);
  summary->success = true;

  while (!x_new_is_valid || cost_new > (cost_reference
                                        + options.sufficient_decrease
                                        * gradient_0
                                        * step_size_new)) {
    const double step_size_current = step_size_new;

    if ((options.interpolation_degree == 0) || !x_new_is_valid) {
      // Pure backtracking search.
      step_size_new *= 0.5;
    } else {
      // Interpolation.
      vector<FunctionSample> samples;
      samples.push_back(ValueAndGradientSample(0.0, cost_0, gradient_0));

      if (options.interpolation_degree == 1) {
        samples.push_back(ValueSample(step_size_new, cost_new));

        if (options.use_higher_degree_interpolation_when_possible &&
            summary->num_evaluations > 2 &&
            x_prev_is_valid) {
          samples.push_back(ValueSample(step_size_prev, cost_prev));
        }
      } else {
        samples.push_back(
            ValueAndGradientSample(step_size_new, cost_new, gradient_new));

        if (options.use_higher_degree_interpolation_when_possible &&
            summary->num_evaluations > 2 &&
            x_prev_is_valid) {
          samples.push_back(
              ValueAndGradientSample(step_size_prev, cost_prev, gradient_prev));
        }
      }

      double min_interpolated_value;
      MinimizeInterpolatingPolynomial(samples, 0.0, step_size_current,
                                      &step_size_new, &min_interpolated_value);
      step_size_new =
          min(max(step_size_new,
                  options.min_relative_step_size_change * step_size_current),
              options.max_relative_step_size_change * step_size_current);
    }

    summary->step_size = step_size_new;
    if (fabs(gradient_0) * summary->step_size < options.step_size_tolerance) {
      LOG(WARNING) << "Step size too small. Truncating to zero.";
      summary->step_size = 0.0;
      break;
    }

    step_size_prev = step_size_current;
    cost_prev = cost_new;
    gradient_prev = gradient_new;

    summary->num_evaluations += 1;
    x_new_is_valid = evaluator->Evaluate(step_size_new, &cost_new, &gradient_new);
  }
}

}  // namespace internal
}  // namespace ceres
