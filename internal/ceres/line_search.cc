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
#include "ceres/stringprintf.h"
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

// Convenience stream operator for pushing FunctionSamples into log messages.
std::ostream & operator<<(std::ostream &os,
                          const FunctionSample& sample) {
  os << "[x: " << sample.x << ", value: " << sample.value
     << ", gradient: " << sample.gradient << ", value_is_valid: "
     << std::boolalpha << sample.value_is_valid << ", gradient_is_valid: "
     << std::boolalpha << sample.gradient_is_valid << "]";
  return os;
};

}  // namespace

LineSearch* LineSearch::Create(const LineSearchType line_search_type,
                               const LineSearch::Options& options,
                               string* error) {
  LineSearch* line_search = NULL;
  switch (line_search_type) {
  case ceres::ARMIJO:
    line_search = new ArmijoLineSearch;
    break;
  case ceres::WOLFE:
    line_search = new WolfeLineSearch;
    break;
  default:
    *error = string("Invalid line search algorithm type: ") +
        LineSearchTypeToString(line_search_type) +
        string(", unable to create line search.");
    return NULL;
  }
  line_search->Init(options);
  return line_search;
}

void LineSearch::Init(const LineSearch::Options& options) {
  options_ = options;
}

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

// Returns step_size \in [min_step_size, max_step_size] which minimizes the
// polynomial of degree defined by interpolation_type which interpolates all
// of the provided samples with valid values.
double LineSearch::InterpolatingPolynomialMinimizingStepSize(
    const LineSearchInterpolationType& interpolation_type,
    const FunctionSample& lowerbound,
    const FunctionSample& previous,
    const FunctionSample& current,
    const double min_step_size,
    const double max_step_size) const {
  if (!current.value_is_valid ||
      (interpolation_type == BISECTION &&
       max_step_size <= current.x)) {
    // Either: sample is invalid; or we are using BISECTION and contracting
    // the step size.
    return min(max(current.x * 0.5, min_step_size), max_step_size);
  } else if (interpolation_type == BISECTION) {
    // We are expanding the search using BISECTION, which is an oxymoron, but is
    // understood to mean always taking the maximum step size.
    CHECK_GT(max_step_size, current.x);
    return max_step_size;
  }
  // Only check if lower-bound is valid here, where it is required
  // to avoid replicating current.value_is_valid == false
  // behaviour in WolfeLineSearch.
  CHECK(lowerbound.value_is_valid)
      << "Ceres bug: lower-bound sample for interpolation is invalid, "
      << "please contact the developers!, interpolation_type: "
      << LineSearchInterpolationTypeToString(interpolation_type)
      << ", lowerbound: " << lowerbound << ", previous: " << previous
      << ", current: " << current;

  // Select step size by interpolating the function and gradient values
  // and minimizing the corresponding polynomial.
  vector<FunctionSample> samples;
  samples.push_back(lowerbound);

  if (interpolation_type == QUADRATIC) {
    // Two point interpolation using function values and the
    // gradient at the lower bound.
    samples.push_back(ValueSample(current.x, current.value));

    if (previous.value_is_valid) {
      // Three point interpolation, using function values and the
      // gradient at the lower bound.
      samples.push_back(ValueSample(previous.x, previous.value));
    }
  } else if (interpolation_type == CUBIC) {
    // Two point interpolation using the function values and the gradients.
    samples.push_back(current);

    if (previous.value_is_valid) {
      // Three point interpolation using the function values and
      // the gradients.
      samples.push_back(previous);
    }
  } else {
    LOG(FATAL) << "Ceres bug: No handler for interpolation_type: "
               << LineSearchInterpolationTypeToString(interpolation_type)
               << ", please contact the developers!";
  }

  double step_size = 0.0, unused_min_value = 0.0;
  MinimizeInterpolatingPolynomial(samples, min_step_size, max_step_size,
                                  &step_size, &unused_min_value);
  return step_size;
}

void ArmijoLineSearch::Search(const double step_size_estimate,
                              const double initial_cost,
                              const double initial_gradient,
                              Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  CHECK_GT(options().sufficient_decrease, 0.0);
  CHECK_LT(options().sufficient_decrease, 1.0);
  CHECK_GT(options().max_num_iterations, 0);
  Function* function = options().function;

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // not step_size_estimate, which is our starting guess.
  const FunctionSample initial_position =
      ValueAndGradientSample(0.0, initial_cost, initial_gradient);

  FunctionSample previous = ValueAndGradientSample(0.0, 0.0, 0.0);
  previous.value_is_valid = false;

  FunctionSample current = ValueAndGradientSample(step_size_estimate, 0.0, 0.0);
  current.value_is_valid = false;

  const bool interpolation_uses_gradients =
      options().interpolation_type == CUBIC;

  ++summary->num_evaluations;
  current.value_is_valid =
      function->Evaluate(current.x,
                         &current.value,
                         interpolation_uses_gradients
                         ? &current.gradient : NULL);
  current.gradient_is_valid =
      interpolation_uses_gradients && current.value_is_valid;
  while (!current.value_is_valid ||
         current.value > (initial_cost
                          + options().sufficient_decrease
                          * initial_gradient
                          * current.x)) {
    // If current.value_is_valid is not true we treat it as if the
    // cost at that point is not large enough to satisfy the sufficient
    // decrease condition.
    const double step_size =
        this->InterpolatingPolynomialMinimizingStepSize(
            options().interpolation_type,
            initial_position,
            previous,
            current,
            (options().max_step_contraction * current.x),
            (options().min_step_contraction * current.x));

    if (fabs(initial_gradient) * step_size < options().min_step_size) {
      summary->error =
          StringPrintf("Line search failed: step_size too small: %.5e "
                       " with initial_gradient: %.5e.", step_size,
                       initial_gradient);
      LOG(WARNING) << summary->error;
      return;
    }

    previous = current;
    current.x = step_size;

    ++summary->num_evaluations;
    current.value_is_valid =
      function->Evaluate(current.x,
                         &current.value,
                         interpolation_uses_gradients
                         ? &current.gradient : NULL);
    current.gradient_is_valid =
        interpolation_uses_gradients && current.value_is_valid;
  }

  summary->optimal_step_size = current.x;
  summary->success = true;
}

void WolfeLineSearch::Search(const double step_size_estimate,
                             const double initial_cost,
                             const double initial_gradient,
                             Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  // All parameters should have been validated by the Solver, but as
  // invalid values would produce crazy nonsense, hard check them here.
  CHECK_GT(options().sufficient_decrease, 0.0);
  CHECK_GT(options().sufficient_curvature_decrease,
           options().sufficient_decrease);
  CHECK_LT(options().sufficient_curvature_decrease, 1.0);
  CHECK_GT(options().max_step_expansion, 1.0);

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // not step_size_estimate, which is our starting guess.
  const FunctionSample initial_position =
      ValueAndGradientSample(0.0, initial_cost, initial_gradient);

  bool do_zoom_search = false;
  FunctionSample zoom_solution;
  FunctionSample bracket_low, bracket_high;

  // Wolfe bracketing phase: Increases step_size until either it finds a point
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
  // this special case, step_size monotonically increases during bracketing.
  if (!this->BracketingPhase(initial_position,
                             step_size_estimate,
                             &bracket_low,
                             &bracket_high,
                             &do_zoom_search,
                             summary)) {
    // Failed to find either a valid point or a valid bracket.
    return;
  }

  if (!do_zoom_search) {
    // Bracketing phase already found a point satisfying the strong Wolfe
    // conditions, no Zoom required.
    summary->optimal_step_size = bracket_low.x;
    summary->success = true;
    return;
  }

  // Bracketing phase found a valid bracket (of non-zero, finite width),
  // which should contain a point satisfying the Wolfe conditions.

  // Wolfe Zoom phase: Called when the Bracketing phase finds an interval that
  // brackets step sizes which satisfy the (strong) Wolfe conditions (before
  // finding a step size that satisfies the conditions).  Zoom successively
  // decreases the size of the interval until a step size which satisfies the
  // Wolfe conditions is found.  The interval is defined by bracket_low &
  // bracket_high, which satisfy:
  //
  //   1. The interval bounded by step sizes: bracket_low.x & bracket_high.x
  //      contains step sizes that satsify the strong Wolfe conditions.
  //   2. bracket_low.x is of all the step sizes evaluated *which satisifed the
  //      Armijo sufficient decrease condition*, the one which generated the
  //      smallest function value, i.e. bracket_low.value <
  //      f(all other steps satisfying Armijo).
  //        - Note that this does _not_ (necessarily) mean that initially
  //          bracket_low.value < bracket_high.value (although this is typical)
  //          e.g. when bracket_low = initial_position, and bracket_high is the
  //          first sample, and which does not satisfy the Armijo condition,
  //          but still has bracket_high.value < initial_position.value.
  //   3. bracket_high is chosen after bracket_low, s.t.
  //      bracket_low.gradient * (bracket_high.x - bracket_low.x) < 0.
  //
  // Important: The high/low in bracket_high & bracket_low refer to their
  // _function_ values, not their step sizes i.e. it is _not_ required that
  // bracket_low.x < bracket_high.x.
  if (!this->ZoomPhase(initial_position,
                       bracket_low,
                       bracket_high,
                       &zoom_solution,
                       summary)) {
    // Failed to find a valid point (given the specified decrease parameters)
    // within the specified bracket.
    return;
  }
  summary->optimal_step_size = zoom_solution.x;
  summary->success = true;
}

bool WolfeLineSearch::BracketingPhase(
    const FunctionSample& initial_position,
    const double step_size_estimate,
    FunctionSample* bracket_low,
    FunctionSample* bracket_high,
    bool* do_zoom_search,
    Summary* summary) {
  Function* function = options().function;

  FunctionSample previous = initial_position;
  FunctionSample current = ValueAndGradientSample(step_size_estimate, 0.0, 0.0);
  current.value_is_valid = false;

  const bool interpolation_uses_gradients =
      options().interpolation_type == CUBIC;

  *do_zoom_search = false;

  ++summary->num_evaluations;
  current.value_is_valid =
      function->Evaluate(current.x,
                         &current.value,
                         interpolation_uses_gradients
                         ? &current.gradient : NULL);
  current.gradient_is_valid =
      interpolation_uses_gradients && current.value_is_valid;

  while (true) {
    if (current.value_is_valid &&
        (current.value > (initial_position.value
                          + options().sufficient_decrease
                          * initial_position.gradient
                          * current.x) ||
         (previous.value_is_valid &&
          current.value > previous.value))) {
      // Bracket found: current step size violates Armijo sufficient decrease
      // condition, or has stepped past an inflection point of f() relative to
      // previous step size.
      *do_zoom_search = true;
      *bracket_low = previous;
      *bracket_high = current;
      break;
    }

    // Irrespective of the interpolation type we are using, we now need the
    // gradient at the current point (which satisfies the Armijo condition)
    // in order to check the strong Wolfe conditions.
    if (!interpolation_uses_gradients) {
      ++summary->num_evaluations;
      current.value_is_valid =
          function->Evaluate(current.x,
                             &current.value,
                             &current.gradient);
      current.gradient_is_valid = current.value_is_valid;
    }

    if (current.value_is_valid &&
        fabs(current.gradient) <=
        -options().sufficient_curvature_decrease * initial_position.gradient) {
      // Current step size satisfies the strong Wolfe conditions, and is thus a
      // valid termination point, therefore a Zoom not required.
      *bracket_low = current;
      *bracket_high = current;
      break;

    } else if (current.value_is_valid &&
               current.gradient >= 0) {
      // Bracket found: current step size has stepped past an inflection point
      // of f(), but Armijo sufficient decrease is still satisfied and
      // f(current) is our best minimum thus far.  Remember step size
      // monotonically increases, thus previous_step_size < current_step_size
      // even though f(previous) > f(current).
      *do_zoom_search = true;
      // Note inverse ordering from first bracket case.
      *bracket_low = current;
      *bracket_high = previous;
      break;

    } else if (summary->num_evaluations >= options().max_num_iterations) {
      // Check num iterations bound here so that we always evaluate the
      // max_num_iterations-th iteration against all conditions, and
      // then perform no additional (unused) evaluations.
      summary->error =
          StringPrintf("Line search failed: Wolfe bracketing phase failed to "
                       "find a point satisfying strong Wolfe conditions, or a "
                       "bracket containing such a point within specified "
                       "max_num_iterations: %d", options().max_num_iterations);
      LOG(WARNING) << summary->error;
      return false;

    } else {
      // Either: f(current) is invalid; or, f(current) is valid, but does not
      // satisfy the strong Wolfe conditions itself, or the conditions for
      // being a boundary of a bracket.

      // If f(current) is valid, (but meets no criteria) expand the search by
      // increasing the step size.
      const double max_step_size =
          current.value_is_valid
          ? (current.x * options().max_step_expansion) : current.x;

      // We are performing 2-point interpolation only here, but the API of
      // InterpolatingPolynomialMinimizingStepSize() allows for up to
      // 3-point interpolation, so pad call with a sample with an invalid
      // value that will therefore be ignored.
      const FunctionSample unused_previous;
      DCHECK(!unused_previous.value_is_valid);
      // Contracts step size if f(current) is not valid.
      const double step_size =
          this->InterpolatingPolynomialMinimizingStepSize(
              options().interpolation_type,
              previous,
              unused_previous,
              current,
              previous.x,
              max_step_size);
      if (fabs(initial_position.gradient) * step_size
          < options().min_step_size) {
        summary->error =
            StringPrintf("Line search failed: step_size too small: %.5e "
                         "with initial_gradient: %.5e", step_size,
                         initial_position.gradient);
        LOG(WARNING) << summary->error;
        return false;
      }

      previous = current.value_is_valid ? current : previous;
      current.x = step_size;

      ++summary->num_evaluations;
      current.value_is_valid =
          function->Evaluate(current.x,
                             &current.value,
                             interpolation_uses_gradients
                             ? &current.gradient : NULL);
      current.gradient_is_valid =
          interpolation_uses_gradients && current.value_is_valid;
    }
  }
  // Either we have a valid point, defined as a bracket of zero width, in which
  // case no zoom is required, or a valid bracket in which to zoom.
  return true;
}

bool WolfeLineSearch::ZoomPhase(const FunctionSample& initial_position,
                                FunctionSample bracket_low,
                                FunctionSample bracket_high,
                                FunctionSample* solution,
                                Summary* summary) {
  Function* function = options().function;

  CHECK(bracket_low.value_is_valid && bracket_low.gradient_is_valid)
      << "Ceres bug: f_low input to Wolfe Zoom invalid, please contact "
      << "the developers!, initial_position: " << initial_position
      << ", bracket_low: " << bracket_low
      << ", bracket_high: "<< bracket_high;
  // We do not require bracket_high.gradient_is_valid as the gradient condition
  // for a valid bracket is only dependent upon bracket_low.gradient, and
  // in order to minimize jacobian evaluations, bracket_high.gradient may
  // not have been calculated (if bracket_high.value does not satisfy the
  // Armijo sufficient decrease condition and interpolation method does not
  // require it).
  CHECK(bracket_high.value_is_valid)
      << "Ceres bug: f_high input to Wolfe Zoom invalid, please "
      << "contact the developers!, initial_position: " << initial_position
      << ", bracket_low: " << bracket_low
      << ", bracket_high: "<< bracket_high;
  CHECK_LT(bracket_low.gradient *
           (bracket_high.x - bracket_low.x), 0.0)
      << "Ceres bug: f_high input to Wolfe Zoom does not satisfy gradient "
      << "condition combined with f_low, please contact the developers!"
      << ", initial_position: " << initial_position
      << ", bracket_low: " << bracket_low
      << ", bracket_high: "<< bracket_high;

  const int num_bracketing_iterations = summary->num_evaluations;
  const bool interpolation_uses_gradients =
      options().interpolation_type == CUBIC;

  while (true) {
    if (summary->num_evaluations >= options().max_num_iterations) {
      summary->error =
          StringPrintf("Line search failed: Wolfe zoom phase failed to "
                       "find a point satisfying strong Wolfe conditions "
                       "within specified max_num_iterations: %d "
                       ", (num iterations taken for bracketing: %d).",
                       options().max_num_iterations, num_bracketing_iterations);
      LOG(WARNING) << summary->error;
      return false;
    }

    // Polynomial interpolation requires inputs ordered according to
    // step size, not f(step size).
    const FunctionSample& lower_bound_step =
        bracket_low.x < bracket_high.x ? bracket_low : bracket_high;
    const FunctionSample& upper_bound_step =
        bracket_low.x < bracket_high.x ? bracket_high : bracket_low;
    // We are performing 2-point interpolation only here, but the API of
    // InterpolatingPolynomialMinimizingStepSize() allows for up to
    // 3-point interpolation, so pad call with a sample with an invalid
    // value that will therefore be ignored.
    const FunctionSample unused_previous;
    DCHECK(!unused_previous.value_is_valid);
    solution->x =
        this->InterpolatingPolynomialMinimizingStepSize(
            options().interpolation_type,
            lower_bound_step,
            unused_previous,
            upper_bound_step,
            lower_bound_step.x,
            upper_bound_step.x);
    // No check on magnitude of step size being too small here as it is
    // lower-bounded by the initial bracket start point, which was valid.
    ++summary->num_evaluations;
    solution->value_is_valid =
        function->Evaluate(solution->x,
                           &solution->value,
                           interpolation_uses_gradients
                           ? &solution->gradient : NULL);
    solution->gradient_is_valid =
        interpolation_uses_gradients && solution->value_is_valid;
    if (!solution->value_is_valid) {
      summary->error =
          StringPrintf("Line search failed: Wolfe Zoom phase found "
                       "step_size: %.5e, for which function is invalid, "
                       "between low_step: %.5e and high_step: %.5e "
                       "at which function is valid.",
                       solution->x, bracket_low.x, bracket_high.x);
      LOG(WARNING) << summary->error;
      return false;
    }

    if ((solution->value > (initial_position.value
                            + options().sufficient_decrease
                            * initial_position.gradient
                            * solution->x)) ||
        (solution->value >= bracket_low.value)) {
      // Armijo sufficient decrease not satisfied, or not better
      // than current lowest sample, use as new upper bound.
      bracket_high = *solution;
      continue;
    }

    // Armijo sufficient decrease satisfied, check strong Wolfe condition.
    if (!interpolation_uses_gradients) {
      // Irrespective of the interpolation type we are using, we now need the
      // gradient at the current point (which satisfies the Armijo condition)
      // in order to check the strong Wolfe conditions.
      ++summary->num_evaluations;
      solution->value_is_valid =
          function->Evaluate(solution->x,
                             &solution->value,
                             &solution->gradient);
      solution->gradient_is_valid = solution->value_is_valid;
      if (!solution->value_is_valid) {
        summary->error =
            StringPrintf("Line search failed: Wolfe Zoom phase found "
                         "step_size: %.5e, for which function is invalid, "
                         "between low_step: %.5e and high_step: %.5e "
                         "at which function is valid.",
                         solution->x, bracket_low.x, bracket_high.x);
        LOG(WARNING) << summary->error;
        return false;
      }
    }
    if (fabs(solution->gradient) <=
        -options().sufficient_curvature_decrease * initial_position.gradient) {
      // Found a valid termination point satisfying strong Wolfe conditions.
      break;

    } else if (solution->gradient * (bracket_high.x - bracket_low.x) >= 0) {
      bracket_high = bracket_low;
    }

    bracket_low = *solution;

    if (fabs(bracket_high.x - bracket_low.x) < options().min_step_size) {
      // Bracket width has been reduced below tolerance, and no
      // point satisfying strong Wolfe conditions has been found.
      return false;
    }
  }
  // solution contains a valid point which satisfies the strong Wolfe
  // conditions.
  return true;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_LINE_SEARCH_MINIMIZER
