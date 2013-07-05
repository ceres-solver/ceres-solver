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
                               string* error) {
  switch (line_search_type) {
  case ceres::ARMIJO:
    return new ArmijoLineSearch;
  case ceres::WOLFE:
    return new WolfeLineSearch;
  default:
    *error = string("Invalid line search algorithm type: ") +
        LineSearchTypeToString(line_search_type) +
        string("Unable to create line search.");
    return NULL;
  }
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
    const FunctionSample& lowerbound_sample,
    const FunctionSample& previous_sample,
    const FunctionSample& current_sample,
    const double min_step_size,
    const double max_step_size) const {
  if (!current_sample.value_is_valid ||
      (interpolation_type == BISECTION &&
       max_step_size <= current_sample.x)) {
    // Either: sample is invalid; or we are using BISECTION and contracting
    // the step size.
    return min(max(current_sample.x * 0.5, min_step_size), max_step_size);
  } else if (interpolation_type == BISECTION) {
    // We are expanding the search using BISECTION, which is an oxymoron, but is
    // understood to mean always taking the maximum step size.
    CHECK(max_step_size > current_sample.x);
    return max_step_size;
  }
  // Only check if lower-bound is valid here, where it is required
  // to avoid replicating current_sample.value_is_valid == false
  // behaviour in WolfeLineSearch.
  CHECK(lowerbound_sample.value_is_valid)
      << "Ceres bug: lower-bound sample for interpolation is invalid, "
      << "please contact the developers!, interpolation_type: "
      << LineSearchInterpolationTypeToString(interpolation_type)
      << ", lowerbound_sample: " << lowerbound_sample
      << ", previous_sample: " << previous_sample
      << ", current_sample: " << current_sample;

  // Select step size by interpolating the function and gradient values
  // and minimizing the corresponding polynomial.
  vector<FunctionSample> samples;
  samples.push_back(lowerbound_sample);

  if (interpolation_type == QUADRATIC) {
    // Two point interpolation using function values and the
    // gradient at the lower bound.
    samples.push_back(ValueSample(current_sample.x,
                                  current_sample.value));

    if (previous_sample.value_is_valid) {
      // Three point interpolation, using function values and the
      // gradient at the lower bound.
      samples.push_back(ValueSample(previous_sample.x,
                                    previous_sample.value));
    }
  } else if (interpolation_type == CUBIC) {
    // Two point interpolation using the function values and the gradients.
    samples.push_back(current_sample);

    if (previous_sample.value_is_valid) {
      // Three point interpolation using the function values and
      // the gradients.
      samples.push_back(previous_sample);
    }
  } else {
    CHECK(false) << "Ceres bug: No handler for interpolation_type: "
                 << LineSearchInterpolationTypeToString(interpolation_type)
                 << ", please contact the developers!";
  }

  double step_size = 0.0, unused_min_value = 0.0;
  MinimizeInterpolatingPolynomial(samples, min_step_size, max_step_size,
                                  &step_size, &unused_min_value);
  return step_size;
}

void ArmijoLineSearch::Search(const LineSearch::Options& options,
                              const double step_size_estimate,
                              const double initial_cost,
                              const double initial_gradient,
                              Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  CHECK_GT(options.sufficient_decrease, 0.0);
  CHECK_LT(options.sufficient_decrease, 1.0);
  CHECK_GT(options.max_num_step_size_iterations, 0);
  Function* function = options.function;

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // not step_size_estimate, which is our starting guess.
  const FunctionSample initial_sample =
      ValueAndGradientSample(0.0, initial_cost, initial_gradient);

  FunctionSample previous_sample = ValueAndGradientSample(0.0, 0.0, 0.0);
  previous_sample.value_is_valid = false;

  FunctionSample current_sample =
      ValueAndGradientSample(step_size_estimate, 0.0, 0.0);
  current_sample.value_is_valid = false;

  ++summary->num_evaluations;
  current_sample.value_is_valid =
      function->Evaluate(current_sample.x,
                         &current_sample.value,
                         options.interpolation_type != CUBIC
                         ? NULL
                         : &current_sample.gradient);
  while (!current_sample.value_is_valid ||
         current_sample.value > (initial_cost
                                 + options.sufficient_decrease
                                 * initial_gradient
                                 * current_sample.x)) {
    // If current_sample.value_is_valid is not true we treat it as if the
    // cost at that point is not large enough to satisfy the sufficient
    // decrease condition.
    const double step_size =
        this->InterpolatingPolynomialMinimizingStepSize(
            options.interpolation_type,
            initial_sample,
            previous_sample,
            current_sample,
            (options.max_step_contraction * current_sample.x),
            (options.min_step_contraction * current_sample.x));

    if (fabs(initial_gradient) * step_size < options.min_step_size) {
      LOG(WARNING) << "Line search failed: step_size too small: " << step_size
                   << " with initial_gradient: " << initial_gradient;
      return;
    }

    previous_sample = current_sample;
    current_sample.x = step_size;

    ++summary->num_evaluations;
    current_sample.value_is_valid =
      function->Evaluate(current_sample.x,
                         &current_sample.value,
                         options.interpolation_type != CUBIC
                         ? NULL
                         : &current_sample.gradient);
  }

  summary->optimal_step_size = current_sample.x;
  summary->success = true;
}

void WolfeLineSearch::Search(const LineSearch::Options& options,
                             const double step_size_estimate,
                             const double initial_cost,
                             const double initial_gradient,
                             Summary* summary) {
  *CHECK_NOTNULL(summary) = LineSearch::Summary();
  // All parameters should have been validated by the Solver, but as
  // invalid values would produce crazy nonsense, hard check them here.
  CHECK_GT(options.sufficient_decrease, 0.0);
  CHECK_GT(options.sufficient_curvature_decrease, options.sufficient_decrease);
  CHECK_LT(options.sufficient_curvature_decrease, 1.0);
  CHECK_GT(options.max_step_expansion, 1.0);

  // Note initial_cost & initial_gradient are evaluated at step_size = 0,
  // not step_size_estimate, which is our starting guess.
  const FunctionSample initial_sample =
      ValueAndGradientSample(0.0, initial_cost, initial_gradient);

  bool do_zoom_search = false;
  FunctionSample final_sample;
  FunctionSample bracket_f_low_sample, bracket_f_high_sample;

  if (!this->BracketingPhase(options,
                             initial_sample,
                             step_size_estimate,
                             &bracket_f_low_sample,
                             &bracket_f_high_sample,
                             &do_zoom_search,
                             summary)) {
    // Failed to find either a valid point or a valid bracket.
    return;
  }

  if (do_zoom_search) {
    // Bracketing phase found a valid bracket (of non-zero, finite width),
    // which should contain a point satisfying the Wolfe conditions.
    if (!this->ZoomPhase(options,
                         initial_sample,
                         bracket_f_low_sample,
                         bracket_f_high_sample,
                         &final_sample,
                         summary)) {
      // Failed to find a valid point (given the specified decrease parameters)
      // within the specified bracket.
      return;
    }
  } else {
    // Bracketing phase already found a point satisfying the strong Wolfe
    // conditions, no Zoom required.
    final_sample = bracket_f_low_sample;
  }

  summary->optimal_step_size = final_sample.x;
  summary->success = true;
}

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
bool WolfeLineSearch::BracketingPhase(
    const LineSearch::Options& options,
    const FunctionSample& initial_sample,
    const double& step_size_estimate,
    FunctionSample* bracket_f_low_sample,
    FunctionSample* bracket_f_high_sample,
    bool* do_zoom_search,
    Summary* summary) {
  Function* function = options.function;

  FunctionSample previous_sample = initial_sample;
  FunctionSample current_sample =
      ValueAndGradientSample(step_size_estimate, 0.0, 0.0);
  current_sample.value_is_valid = false;

  *do_zoom_search = false;

  ++summary->num_evaluations;
  current_sample.value_is_valid =
      function->Evaluate(current_sample.x,
                         &current_sample.value,
                         &current_sample.gradient);
  bool found_bracket_or_valid_point = false;
  while (!found_bracket_or_valid_point) {
    if (current_sample.value_is_valid &&
        (current_sample.value > (initial_sample.value
                                 + options.sufficient_decrease
                                 * initial_sample.gradient
                                 * current_sample.x) ||
         (previous_sample.value_is_valid &&
          current_sample.value > previous_sample.value))) {
      // Bracket found: current step size violates Armijo sufficient decrease
      // condition, or has stepped past an inflection point of f() relative to
      // previous step size.
      found_bracket_or_valid_point = true;
      *do_zoom_search = true;
      *bracket_f_low_sample = previous_sample;
      *bracket_f_high_sample = current_sample;

    } else if (current_sample.value_is_valid &&
               fabs(current_sample.gradient) <=
               -options.sufficient_curvature_decrease *
               initial_sample.gradient) {
      // Current step size satisfies the strong Wolfe conditions, and is thus a
      // valid termination point, therefore a Zoom not required.
      found_bracket_or_valid_point = true;
      *bracket_f_low_sample = current_sample;
      *bracket_f_high_sample = current_sample;

    } else if (current_sample.value_is_valid &&
               current_sample.gradient >= 0) {
      // Bracket found: current step size has stepped past an inflection point
      // of f(), but Armijo sufficient decrease is still satisfied and
      // f(current) is our best minimum thus far.  Remember step size
      // monotonically increases, thus previous_step_size < current_step_size
      // even though f(previous) > f(current).
      found_bracket_or_valid_point = true;
      *do_zoom_search = true;
      // Note inverse ordering from first bracket case.
      *bracket_f_low_sample = current_sample;
      *bracket_f_high_sample = previous_sample;

    } else if (summary->num_evaluations >=
               options.max_num_step_size_iterations) {
      // Check num iterations bound here so that we always evaluate the
      // max_num_step_size_iterations-th iteration against all conditions, and
      // then perform no additional (unused) evaluations.
      LOG(WARNING) << "Line search failed: Wolfe bracketing phase failed to "
                   << "find a point satisfying strong Wolfe conditions, or a "
                   << "bracket containing such a point within specified "
                   << "max_num_step_size_iterations: "
                   << options.max_num_step_size_iterations;
      return false;

    } else {
      // Either: f(current) is invalid; or, f(current) is valid, but does not
      // satisfy the strong Wolfe conditions itself, or the conditions for
      // being a boundary of a bracket.

      // If f(current) is valid, (but meets no criteria) expand the search by
      // increasing the step size.
      const double max_step_size =
          current_sample.value_is_valid
          ? (current_sample.x *
             options.max_step_expansion)
          : current_sample.x;

      // We are performing 2-point interpolation only here, but the API of
      // InterpolatingPolynomialMinimizingStepSize() allows for up to
      // 3-point interpolation, so pad call with a sample with an invalid
      // value that will therefore be ignored.
      const FunctionSample unused_previous_sample;
      DCHECK(!unused_previous_sample.value_is_valid);
      // Contracts step size if f(current) is not valid.
      const double step_size =
          this->InterpolatingPolynomialMinimizingStepSize(
              options.interpolation_type,
              previous_sample,
              unused_previous_sample,
              current_sample,
              previous_sample.x,
              max_step_size);
      if (fabs(initial_sample.gradient) * step_size < options.min_step_size) {
        LOG(WARNING) << "Line search failed: step_size too small: " << step_size
                     << " with initial_gradient: " << initial_sample.gradient;
        return false;
      }

      previous_sample = current_sample;
      current_sample.x = step_size;

      ++summary->num_evaluations;
      current_sample.value_is_valid =
        function->Evaluate(current_sample.x,
                           &current_sample.value,
                           &current_sample.gradient);
    }
  }
  // Either we have a valid point, defined as a bracket of zero width, in which
  // case no zoom is required, or a valid bracket in which to zoom.
  return true;
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
bool WolfeLineSearch::ZoomPhase(const LineSearch::Options& options,
                                const FunctionSample& initial_sample,
                                FunctionSample bracket_f_low_sample,
                                FunctionSample bracket_f_high_sample,
                                FunctionSample* final_sample,
                                Summary* summary) {
  Function* function = options.function;

  CHECK(bracket_f_low_sample.value_is_valid)
      << "Ceres bug: f_low input to Wolfe Zoom invalid, please contact "
      << "the developers!, initial_sample: " << initial_sample
      << ", bracket_f_low_sample: " << bracket_f_low_sample
      << ", bracket_f_high_sample: "<< bracket_f_high_sample;
  CHECK(bracket_f_high_sample.value_is_valid)
      << "Ceres bug: f_high input to Wolfe Zoom invalid, please "
      << "contact the developers!, initial_sample: " << initial_sample
      << ", bracket_f_low_sample: " << bracket_f_low_sample
      << ", bracket_f_high_sample: "<< bracket_f_high_sample;
  CHECK_LT(bracket_f_low_sample.gradient *
           (bracket_f_high_sample.x - bracket_f_low_sample.x), 0.0)
      << "Ceres bug: f_high input to Wolfe Zoom does not satisfy gradient "
      << "condition combined with f_low, please contact the developers!"
      << ", initial_sample: " << initial_sample
      << ", bracket_f_low_sample: " << bracket_f_low_sample
      << ", bracket_f_high_sample: "<< bracket_f_high_sample;

  const int num_bracketing_iterations = summary->num_evaluations;

  bool found_optimal_point = false;
  while (!found_optimal_point) {
    if (summary->num_evaluations >=
        options.max_num_step_size_iterations) {
      LOG(WARNING) << "Line search failed: Wolfe zoom phase failed to "
                   << "find a point satisfying strong Wolfe conditions "
                   << "within specified max_num_step_size_iterations: "
                   << options.max_num_step_size_iterations
                   << ", (num iterations taken for bracketing: "
                   << num_bracketing_iterations << ").";
      return false;
    }

    // Polynomial interpolation requires inputs ordered according to
    // step size, not f(step size).
    const FunctionSample& lower_bound_step_size_sample =
        bracket_f_low_sample.x < bracket_f_high_sample.x
        ? bracket_f_low_sample
        : bracket_f_high_sample;
    const FunctionSample& upper_bound_step_size_sample =
        bracket_f_low_sample.x < bracket_f_high_sample.x
        ? bracket_f_high_sample
        : bracket_f_low_sample;
    // We are performing 2-point interpolation only here, but the API of
    // InterpolatingPolynomialMinimizingStepSize() allows for up to
    // 3-point interpolation, so pad call with a sample with an invalid
    // value that will therefore be ignored.
    const FunctionSample unused_previous_sample;
    DCHECK(!unused_previous_sample.value_is_valid);
    final_sample->x =
        this->InterpolatingPolynomialMinimizingStepSize(
            options.interpolation_type,
            lower_bound_step_size_sample,
            unused_previous_sample,
            upper_bound_step_size_sample,
            lower_bound_step_size_sample.x,
            upper_bound_step_size_sample.x);
    // No check on magnitude of step size being too small here as it is
    // lower-bounded by the initial bracket start point, which was valid.
    ++summary->num_evaluations;
    final_sample->value_is_valid =
        function->Evaluate(final_sample->x,
                           &final_sample->value,
                           &final_sample->gradient);
    if (!final_sample->value_is_valid) {
      LOG(WARNING) << "Line search failed: Wolfe Zoom phase found step_size: "
                   << final_sample->x << " for which function is "
                   << "invalid, between low_step: "
                   << bracket_f_low_sample.x
                   << "& high_step: " << bracket_f_high_sample.x
                   << " at which function is valid.";
      return false;
    }

    if ((final_sample->value > (initial_sample.value
                                + options.sufficient_decrease
                                * initial_sample.gradient
                                * final_sample->x)) ||
        (final_sample->value >= bracket_f_low_sample.value)) {
      // Armijo sufficient decrease not satisfied, or not better
      // than current lowest sample, use as new upper bound.
      bracket_f_high_sample = *final_sample;

    } else {
      // Armijo sufficient decrease satisfied, check Wolfe condition.

      if (fabs(final_sample->gradient) <=
          -options.sufficient_curvature_decrease * initial_sample.gradient) {
        // Valid termination point satisfying strong Wolfe conditions found.
        found_optimal_point = true;

      } else if (final_sample->gradient *
                 (bracket_f_high_sample.x - bracket_f_low_sample.x) >= 0) {
        bracket_f_high_sample = bracket_f_low_sample;
      }

      if (!found_optimal_point) {
        bracket_f_low_sample = *final_sample;

        if (fabs(bracket_f_high_sample.x - bracket_f_low_sample.x)
            < options.min_step_size) {
          // Bracket width has been reduced below tolerance, and no
          // point satisfying strong Wolfe conditions has been found.
          return false;
        }
      }
    }
  }
  // final_sample contains a valid point which satisfies the strong Wolfe
  // conditions.
  return true;
}

}  // namespace internal
}  // namespace ceres

#endif  // CERES_NO_LINE_SEARCH_MINIMIZER
