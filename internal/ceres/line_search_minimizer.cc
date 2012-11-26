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

#include "ceres/line_search_minimizer.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <iostream>

#include "Eigen/Dense"
#include "ceres/array_utils.h"
#include "ceres/evaluator.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/line_search.h"
#include "ceres/stringprintf.h"
#include "ceres/types.h"
#include "ceres/wall_time.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {
namespace {
// Small constant for various floating point issues.
const double kEpsilon = 1e-12;
}  // namespace

// Execute the list of IterationCallbacks sequentially. If any one of
// the callbacks does not return SOLVER_CONTINUE, then stop and return
// its status.
CallbackReturnType LineSearchMinimizer::RunCallbacks(
    const IterationSummary& iteration_summary) {
  for (int i = 0; i < options_.callbacks.size(); ++i) {
    const CallbackReturnType status =
        (*options_.callbacks[i])(iteration_summary);
    if (status != SOLVER_CONTINUE) {
      return status;
    }
  }
  return SOLVER_CONTINUE;
}

void LineSearchMinimizer::Init(const Minimizer::Options& options) {
  options_ = options;
}

void LineSearchMinimizer::Minimize(const Minimizer::Options& options,
                                   double* parameters,
                                   Solver::Summary* summary) {
  double start_time = WallTimeInSeconds();
  double iteration_start_time =  start_time;
  Init(options);

  Evaluator* evaluator = CHECK_NOTNULL(options_.evaluator);
  const int num_parameters = evaluator->NumParameters();
  const int num_effective_parameters = evaluator->NumEffectiveParameters();

  summary->termination_type = NO_CONVERGENCE;
  summary->num_successful_steps = 0;
  summary->num_unsuccessful_steps = 0;

  VectorRef x(parameters, num_parameters);

  Vector gradient(num_effective_parameters);
  double gradient_squared_norm;
  Vector previous_gradient(num_effective_parameters);
  Vector gradient_change(num_effective_parameters);
  double previous_gradient_squared_norm = 0.0;

  Vector search_direction(num_effective_parameters);
  Vector previous_search_direction(num_effective_parameters);

  Vector delta(num_effective_parameters);
  Vector x_plus_delta(num_parameters);

  double directional_derivative = 0.0;
  double previous_directional_derivative = 0.0;

  IterationSummary iteration_summary;
  iteration_summary.iteration = 0;
  iteration_summary.step_is_valid = false;
  iteration_summary.step_is_successful = false;
  iteration_summary.cost_change = 0.0;
  iteration_summary.gradient_max_norm = 0.0;
  iteration_summary.step_norm = 0.0;
  iteration_summary.linear_solver_iterations = 0;
  iteration_summary.step_solver_time_in_seconds = 0;

  // Do initial cost and Jacobian evaluation.
  double cost = 0.0;
  double previous_cost = 0.0;
  if (!evaluator->Evaluate(x.data(), &cost, NULL, gradient.data(), NULL)) {
    LOG(WARNING) << "Terminating: Cost and gradient evaluation failed.";
    summary->termination_type = NUMERICAL_FAILURE;
    return;
  }

  gradient_squared_norm = gradient.squaredNorm();
  iteration_summary.cost = cost + summary->fixed_cost;
  iteration_summary.gradient_max_norm = gradient.lpNorm<Eigen::Infinity>();

  // The initial gradient max_norm is bounded from below so that we do
  // not divide by zero.
  const double gradient_max_norm_0 =
      max(iteration_summary.gradient_max_norm, kEpsilon);
  const double absolute_gradient_tolerance =
      options_.gradient_tolerance * gradient_max_norm_0;

  if (iteration_summary.gradient_max_norm <= absolute_gradient_tolerance) {
    summary->termination_type = GRADIENT_TOLERANCE;
    VLOG(1) << "Terminating: Gradient tolerance reached."
            << "Relative gradient max norm: "
            << iteration_summary.gradient_max_norm / gradient_max_norm_0
            << " <= " << options_.gradient_tolerance;
    return;
  }

  iteration_summary.iteration_time_in_seconds =
      WallTimeInSeconds() - iteration_start_time;
  iteration_summary.cumulative_time_in_seconds =
      WallTimeInSeconds() - start_time
      + summary->preprocessor_time_in_seconds;
  summary->iterations.push_back(iteration_summary);

  // Call the various callbacks.
  switch (RunCallbacks(iteration_summary)) {
    case SOLVER_TERMINATE_SUCCESSFULLY:
      summary->termination_type = USER_SUCCESS;
      VLOG(1) << "Terminating: User callback returned USER_SUCCESS.";
      return;
    case SOLVER_ABORT:
      summary->termination_type = USER_ABORT;
      VLOG(1) << "Terminating: User callback returned  USER_ABORT.";
      return;
    case SOLVER_CONTINUE:
      break;
    default:
      LOG(FATAL) << "Unknown type of user callback status";
  }

  LineSearchFunction line_search_function(evaluator);
  LineSearch::Options line_search_options;
  line_search_options.function = &line_search_function;
  ArmijoLineSearch line_search;
  LineSearch::Summary line_search_summary;

  while (true) {
    iteration_start_time = WallTimeInSeconds();
    if (iteration_summary.iteration >= options_.max_num_iterations) {
      summary->termination_type = NO_CONVERGENCE;
      VLOG(1) << "Terminating: Maximum number of iterations reached.";
      break;
    }

    const double total_solver_time = iteration_start_time - start_time +
        summary->preprocessor_time_in_seconds;
    if (total_solver_time >= options_.max_solver_time_in_seconds) {
      summary->termination_type = NO_CONVERGENCE;
      VLOG(1) << "Terminating: Maximum solver time reached.";
      break;
    }

    previous_search_direction = search_direction;

    iteration_summary = IterationSummary();
    iteration_summary.iteration = summary->iterations.back().iteration + 1;
    iteration_summary.step_is_valid = false;
    iteration_summary.step_is_successful = false;

    if (iteration_summary.iteration == 1) {
      search_direction = -gradient;
      directional_derivative = -gradient_squared_norm;
    } else {
      // TODO(sameeragarwal): This should probably be refactored into
      // a set of functions. But we will do that once things settle
      // down in this solver.
      switch (options_.line_search_direction_type) {
        case STEEPEST_DESCENT:
          search_direction = -gradient;
          directional_derivative = -gradient_squared_norm;
          break;

        case NONLINEAR_CONJUGATE_GRADIENT:
          {
            double beta = 0.0;

            switch (options_.nonlinear_conjugate_gradient_type) {
              case FLETCHER_REEVES:
                beta = gradient.squaredNorm() / previous_gradient_squared_norm;
                break;

              case POLAK_RIBIRERE:
                gradient_change = gradient - previous_gradient;
                beta = gradient.dot(gradient_change) / previous_gradient_squared_norm;
                break;

              case HESTENES_STIEFEL:
                gradient_change = gradient - previous_gradient;
                beta = gradient.dot(gradient_change) / previous_search_direction.dot(gradient_change);
                break;

              default:
                LOG(FATAL) << "Unknown nonlinear conjugate gradient type: "
                           << options_.nonlinear_conjugate_gradient_type;
            }

            search_direction = -gradient + beta * previous_search_direction;
          }

          directional_derivative =  gradient.dot(search_direction);
          if (directional_derivative > -options.function_tolerance) {
            LOG(WARNING) << "Restarting non-linear conjugate gradients: "
                         << directional_derivative;
            search_direction = -gradient;
            directional_derivative = -gradient_squared_norm;
          }
          break;

        default:
          LOG(FATAL) << "Unknown line search direction type: "
                     << options_.line_search_direction_type;
      }
    }

    const double initial_step_size = (iteration_summary.iteration == 1)
        ? min(1.0, 1.0/gradient.lpNorm<Eigen::Infinity>())
        : min(1.0 ,2.0 * (cost - previous_cost)/directional_derivative);

    previous_cost = cost;
    previous_gradient = gradient;
    previous_gradient_squared_norm = gradient_squared_norm;
    previous_directional_derivative = directional_derivative;

    line_search_function.Init(x, search_direction);
    line_search.Search(line_search_options,
                       initial_step_size,
                       cost,
                       directional_derivative,
                       &line_search_summary);

    delta = line_search_summary.optimal_step_size * search_direction;
    evaluator->Plus(x.data(), delta.data(), x_plus_delta.data());
    evaluator->Evaluate(x_plus_delta.data(), &cost, NULL, gradient.data(), NULL);
    gradient_squared_norm = gradient.squaredNorm();
    x = x_plus_delta;

    iteration_summary.cost = cost + summary->fixed_cost;
    iteration_summary.cost_change = previous_cost - cost;
    iteration_summary.step_norm = delta.norm();
    iteration_summary.gradient_max_norm = gradient.lpNorm<Eigen::Infinity>();
    iteration_summary.step_is_valid = true;
    iteration_summary.step_is_successful = true;
    iteration_summary.step_norm = delta.norm();
    iteration_summary.step_size =  line_search_summary.optimal_step_size;
    iteration_summary.line_search_function_evaluations =
        line_search_summary.num_evaluations;

    if (iteration_summary.gradient_max_norm <= absolute_gradient_tolerance) {
      summary->termination_type = GRADIENT_TOLERANCE;
      VLOG(1) << "Terminating: Gradient tolerance reached."
              << "Relative gradient max norm: "
              << iteration_summary.gradient_max_norm / gradient_max_norm_0
              << " <= " << options_.gradient_tolerance;
      break;
    }

    const double absolute_function_tolerance =
        options_.function_tolerance * previous_cost;
    if (fabs(iteration_summary.cost_change) < absolute_function_tolerance) {
      VLOG(1) << "Terminating. Function tolerance reached. "
              << "|cost_change|/cost: "
              << fabs(iteration_summary.cost_change) / previous_cost
              << " <= " << options_.function_tolerance;
      summary->termination_type = FUNCTION_TOLERANCE;
      return;
    }

    iteration_summary.iteration_time_in_seconds =
        WallTimeInSeconds() - iteration_start_time;
    iteration_summary.cumulative_time_in_seconds =
        WallTimeInSeconds() - start_time
        + summary->preprocessor_time_in_seconds;
    summary->iterations.push_back(iteration_summary);

    switch (RunCallbacks(iteration_summary)) {
      case SOLVER_TERMINATE_SUCCESSFULLY:
        summary->termination_type = USER_SUCCESS;
        VLOG(1) << "Terminating: User callback returned USER_SUCCESS.";
        return;
      case SOLVER_ABORT:
        summary->termination_type = USER_ABORT;
        VLOG(1) << "Terminating: User callback returned  USER_ABORT.";
        return;
      case SOLVER_CONTINUE:
        break;
      default:
        LOG(FATAL) << "Unknown type of user callback status";
    }
  }
}

}  // namespace internal
}  // namespace ceres
