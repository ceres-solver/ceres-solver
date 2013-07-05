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

#ifndef CERES_NO_LINE_SEARCH_MINIMIZER
#include "ceres/line_search.h"

#include "ceres/fpclassify.h"
#include "ceres/evaluator.h"
#include "ceres/internal/eigen.h"
#include "ceres/polynomial.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {
namespace {

FunctionSample ValueSample(const double x, const double value) {
  FunctionSample sample;
  sample.x = x;
  sample.value = value;
  sample.value_is_valid = true;
  return sample;
};

FunctionSample ValueAndGradientSample(const double x,
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

// Result: step_size \in [min_step_size, max_step_size].
void PolynomialInterpolationStepSizeUpdate(
   const LineSearchInterpolationType& interpolation_type,
   const FunctionSample& lowerbound_line_sample,
   const FunctionSample& previous_line_sample,
   const FunctionSample& current_line_sample,
   const double& min_step_size,
   const double& max_step_size,
   double* step_size) {
  if ((interpolation_type == BISECTION) || 
      !current_line_sample.value_is_valid) {
    *step_size = 
      min(max(current_line_sample.x * 0.5, min_step_size), max_step_size);
  } else {
    // Only check if lower-bound is valid here, where it is required
    // to avoid replicating current_line_sample.value_is_valid == false
    // behaviour in WolfeLineSearch.
    CHECK(lowerbound_line_sample.value_is_valid);

    // Backtrack by interpolating the function and gradient values
    // and minimizing the corresponding polynomial.
    vector<FunctionSample> samples;
    samples.push_back(lowerbound_line_sample);

    if (interpolation_type == QUADRATIC) {
      // Two point interpolation using function values and the
      // gradient at the lower bound.
      samples.push_back(ValueSample(current_line_sample.x,
                                    current_line_sample.value));

      if (previous_line_sample.value_is_valid) {
        // Three point interpolation, using function values and the
        // gradient at the lower bound.
        samples.push_back(ValueSample(previous_line_sample.x,
                                      previous_line_sample.value));
      }
    } else {  // interpolation_type == CUBIC
      // Two point interpolation using the function values and the gradients.
      samples.push_back(current_line_sample);

      if (previous_line_sample.value_is_valid) {
        // Three point interpolation using the function values and
        // the gradients.
        samples.push_back(previous_line_sample);
      }
    }

    double unused_min_value = 0.0;
    MinimizeInterpolatingPolynomial(samples, min_step_size, max_step_size,
                                    step_size, &unused_min_value);
  }
};

}  // namespace

LineSearchFunction::LineSearchFunction(Evaluator* evaluator)
    : evaluator_(evaluator),
      position_(evaluator->NumParameters()),
      direction_(evaluator->NumEffectiveParameters()),
      evaluation_point_(evaluator->NumParameters()),
      scaled_direction_(evaluator->NumEffectiveParameters()),
      gradient_(evaluator->NumEffectiveParameters()) {
}

void LineSearchFunction::Init(const Vector& position,
                              const Vector& direction) {
  position_ = position;
  direction_ = direction;
}

bool LineSearchFunction::Evaluate(const double x, double* f, double* g) {
  scaled_direction_ = x * direction_;
  if (!evaluator_->Plus(position_.data(),
                        scaled_direction_.data(),
                        evaluation_point_.data())) {
    return false;
  }

  if (g == NULL) {
    return (evaluator_->Evaluate(evaluation_point_.data(),
                                  f, NULL, NULL, NULL) &&
            IsFinite(*f));
  }

  if (!evaluator_->Evaluate(evaluation_point_.data(),
                            f,
                            NULL,
                            gradient_.data(), NULL)) {
    return false;
  }

  *g = direction_.dot(gradient_);
  return IsFinite(*f) && IsFinite(*g);
}

void ArmijoLineSearch::Search(const LineSearch::Options& options,
                              const double initial_step_size,
                              const double initial_cost,
                              const double initial_gradient,
                              Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  CHECK_GT(options.sufficient_decrease, 0.0);
  CHECK_LT(options.sufficient_decrease, 1.0);
  Function* function = options.function;

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // _not_ initial_step_size, which is our starting guess.
  const FunctionSample initial_line_sample = 
    ValueAndGradientSample(0.0, initial_cost, initial_gradient);

  FunctionSample previous_line_sample = ValueAndGradientSample(0.0, 0.0, 0.0);
  previous_line_sample.value_is_valid = false;

  FunctionSample current_line_sample = 
    ValueAndGradientSample(initial_step_size, 0.0, 0.0);
  current_line_sample.value_is_valid = false;

  ++summary->num_evaluations;
  current_line_sample.value_is_valid =
      function->Evaluate(current_line_sample.x,
                         &current_line_sample.value,
                         options.interpolation_type != CUBIC ? 
                         NULL : &current_line_sample.gradient);
  while (!current_line_sample.value_is_valid || 
         current_line_sample.value > (initial_cost
                                      + options.sufficient_decrease
                                      * initial_gradient
                                      * current_line_sample.x)) {
    // If current_line_sample.value_is_valid is not true we treat it as if the
    // cost at that point is not large enough to satisfy the sufficient
    // decrease condition.
    double step_size = 0.0;
    // TODO(alexstewart): Should bounding apply if using BISECTION
    // interpolation and/or when the current step-size is invalid?
    PolynomialInterpolationStepSizeUpdate(options.interpolation_type,
                                          initial_line_sample,
                                          previous_line_sample,
                                          current_line_sample,
                                          (options.min_relative_step_size_change
                                           * current_line_sample.x),
                                          (options.max_relative_step_size_change
                                           * current_line_sample.x),
                                          &step_size);

    if (fabs(initial_gradient) * step_size < options.min_step_size) {
      LOG(WARNING) << "Line search failed: step_size too small: " << step_size
                   << " with initial_gradient: " << initial_gradient;
      return;
    }

    previous_line_sample = current_line_sample;
    current_line_sample.x = step_size;

    ++summary->num_evaluations;
    current_line_sample.value_is_valid =
      function->Evaluate(current_line_sample.x,
                         &current_line_sample.value,
                         options.interpolation_type != CUBIC ? 
                         NULL : &current_line_sample.gradient);
  }

  summary->optimal_step_size = current_line_sample.x;
  summary->success = true;
}

void WolfeLineSearch::Search(const LineSearch::Options& options,
                             const double initial_step_size,
                             const double initial_cost,
                             const double initial_gradient,
                             Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  CHECK_GT(options.sufficient_decrease, 0.0);
  CHECK_GT(options.sufficient_curvature_decrease, options.sufficient_decrease);
  CHECK_LT(options.sufficient_curvature_decrease, 1.0);
  CHECK_GT(options.expansion_max_relative_step_size_change, 1.0);
  Function* function = options.function;

  // Placeholder to allow reuse of PolynomialInterpolationStepSizeUpdate()
  // when performing interpolation between two points only (no previous)
  // as value_is_valid is false, never used in interpolation.
  const FunctionSample ignored_placeholder_line_sample;
  DCHECK(!ignored_placeholder_line_sample.value_is_valid);

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // _not_ initial_step_size, which is our starting guess.
  FunctionSample previous_line_sample = 
    ValueAndGradientSample(0.0, initial_cost, initial_gradient);
  FunctionSample current_line_sample =
    ValueAndGradientSample(initial_step_size, 0.0, 0.0);
  current_line_sample.value_is_valid = false;

  bool perform_zoom_search = false;
  FunctionSample f_low_line_sample, f_high_line_sample;

  // Wolfe Bracketing phase: Increases step_size until either it finds a point
  // that satisfies the (strong) Wolfe conditions, or an interval that brackets
  // step sizes which satisfy the conditions.  From Nocedal & Wright [1] p61 the
  // interval: (step_size_{k-1}, step_size_{k}) contains step lengths satisfying
  // the strong Wolfe conditions if one of the following conditions are met:
  // 
  //   1. step_size_{k} violates the sufficient decrease (Armijo) condition.
  //   2. f(step_size_{k}) >= f(step_size_{k-1}).
  //   3. f'(step_size_{k}) >= 0.
  // 
  // Caveat: If f(step_size_{k}) is invalid, then step_size is reduced, ignoring
  // this special case, step_size monotonically increases during Bracketing.
  ++summary->num_evaluations;
  current_line_sample.value_is_valid =
      function->Evaluate(current_line_sample.x,
                         &current_line_sample.value,
                         &current_line_sample.gradient);
  bool found_bracket = false;
  while (!found_bracket) {
    if (current_line_sample.value_is_valid && 
        (current_line_sample.value > (initial_cost
                                      + options.sufficient_decrease
                                      * initial_gradient
                                      * current_line_sample.x) ||
         (previous_line_sample.value_is_valid && 
          current_line_sample.value > previous_line_sample.value))) {
      // Bracket found: current step size violates Armijo sufficient decrease
      // condition, or has stepped past an inflection point of f() relative to
      // previous step size.
      found_bracket = true;
      perform_zoom_search = true;
      f_low_line_sample = previous_line_sample;
      f_high_line_sample = current_line_sample;

    } else if (current_line_sample.value_is_valid && 
               fabs(current_line_sample.gradient) <= 
               -options.sufficient_curvature_decrease*initial_gradient) {
      // Current step size satisfies strong Wolfe conditions, Zoom not required.
      found_bracket = true;

    } else if (current_line_sample.value_is_valid && 
               current_line_sample.gradient >= 0) {
      // Bracket found: current step size has stepped past an inflection point
      // of f(), but Armijo sufficient decrease is still satisfied and
      // f(current) is our best minimum thus far.  Remember step size
      // monotonically increases, thus previous_step_size < current_step_size
      // even though f(previous) > f(current).
      found_bracket = true;
      perform_zoom_search = true;
      // Note inverse ordering from first bracket case.
      f_low_line_sample = current_line_sample;
      f_high_line_sample = previous_line_sample;

    } else {
      // If f(current) is valid, (but meets no criteria) expand the search by
      // increasing the step size.
      const double max_step_size = current_line_sample.value_is_valid ?
        current_line_sample.x * options.expansion_max_relative_step_size_change
        : current_line_sample.x;

      double step_size = 0.0;
      // Contracts step size if f(current) is not valid.
      PolynomialInterpolationStepSizeUpdate(options.interpolation_type,
                                            previous_line_sample,
                                            ignored_placeholder_line_sample,
                                            current_line_sample,
                                            previous_line_sample.x,
                                            max_step_size,
                                            &step_size);
      if (fabs(initial_gradient) * step_size < options.min_step_size) {
        LOG(WARNING) << "Line search failed: step_size too small: " << step_size
                     << " with initial_gradient: " << initial_gradient;
        return;
      }

      previous_line_sample = current_line_sample;
      current_line_sample.x = step_size;

      ++summary->num_evaluations;
      current_line_sample.value_is_valid =
        function->Evaluate(current_line_sample.x,
                           &current_line_sample.value,
                           &current_line_sample.gradient);
    }
  }

  // Wolfe Zoom phase: Called when the Bracketing phase finds an interval that
  // brackets step sizes which satisfy the (strong) Wolfe conditions (before
  // finding a step size that satisfies the conditions).  Zoom successively
  // decreases the size of the interval until a step size which satisfies the
  // the conditions is found.  The interval is defined by low & high, which
  // satisfy:
  // 
  //   1. The interval bounded by low & high contains step sizes that satsify
  //      the strong Wolfe conditions.
  //   2. low is of all the step sizes evaluated *which satisifed the Armijo
  //      sufficient decrease condition*, the one which generated the smallest
  //      function value, i.e. f(low) < f(all other steps satisfying Armijo).
  //        - Note that this does _not_ (necessarily) mean that f(low) < f(high)
  //          (although this is typical) e.g. when low = initial, and high is
  //          the first sample, and which does not satisfy the Armijo condition,
  //          but still has f(high) < f(initial).
  //   3. high is chosen after low, s.t. f'(low)*(high - low) < 0.
  // 
  // Important: high & low step sizes are defined by their _function_ values,
  // i.e. it is _not_ required that low_step < high_step.
  if (perform_zoom_search) {
    CHECK(f_low_line_sample.value_is_valid);
    CHECK(f_high_line_sample.value_is_valid);
    CHECK_LT(f_low_line_sample.gradient*(f_high_line_sample.x 
                                       -f_low_line_sample.x), 0.0);

    bool found_optimal_point = false;
    while (!found_optimal_point) {
      // Polynomial interpolation requires inputs ordered according to
      // step size, not f(step size).
      const FunctionSample& lower_bound_step_size_line_sample = 
        f_low_line_sample.x < f_high_line_sample.x ? 
        f_low_line_sample : f_high_line_sample;
      const FunctionSample& upper_bound_step_size_line_sample = 
        f_low_line_sample.x < f_high_line_sample.x ? 
        f_high_line_sample : f_low_line_sample;
      PolynomialInterpolationStepSizeUpdate(options.interpolation_type,
                                            lower_bound_step_size_line_sample,
                                            ignored_placeholder_line_sample,
                                            upper_bound_step_size_line_sample,
                                            lower_bound_step_size_line_sample.x,
                                            upper_bound_step_size_line_sample.x,
                                            &current_line_sample.x);
      // No check on magnitude of step size being too small here as it is
      // lower-bounded by the initial bracket start point, which was valid.
      ++summary->num_evaluations;
      current_line_sample.value_is_valid =
        function->Evaluate(current_line_sample.x,
                           &current_line_sample.value,
                           &current_line_sample.gradient);
      if (!current_line_sample.value_is_valid) {
        LOG(WARNING) << "Line search failed: Wolfe Zoom phase found step_size: "
                     << current_line_sample.x << " for which function is "
                     << "invalid, between low_step: " << f_low_line_sample.x
                     << "& high_step: " << f_high_line_sample.x << " at which "
                     << "function is valid.";
        return;
      }

      if ((current_line_sample.value > (initial_cost
                                        + options.sufficient_decrease
                                        * initial_gradient
                                        * current_line_sample.x)) ||
          (current_line_sample.value >= f_low_line_sample.value)) {
        // Armijo sufficient decrease not satisfied, or not better
        // than current lowest sample, use as new upper bound.
        f_high_line_sample = current_line_sample;

      } else {
        // Armijo sufficient decrease satisfied, check Wolfe condition.

        if (fabs(current_line_sample.gradient) <= 
            -options.sufficient_curvature_decrease*initial_gradient) {
          // (strong) Wolfe conditions satisfied.
          found_optimal_point = true;

        } else if (current_line_sample.gradient *
                   (f_high_line_sample.x - f_low_line_sample.x) >= 0) {
          f_high_line_sample = f_low_line_sample;
        }

        if (!found_optimal_point) {
          f_low_line_sample = current_line_sample;

          if (fabs(f_high_line_sample.x - f_low_line_sample.x)
              < options.min_step_size) {
            // Bracket width has been reduced below tolerance, and no
            // point satisfying strong Wolfe conditions has been found.
            return;
          }
        }

      }
    }
  }

  summary->optimal_step_size = current_line_sample.x;
  summary->success = true;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_LINE_SEARCH_MINIMIZER
