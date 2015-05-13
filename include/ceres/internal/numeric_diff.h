// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
//         mierle@gmail.com (Keir Mierle)
//         tbennun@gmail.com (Tal Ben-Nun)
//
// Finite differencing routines used by NumericDiffCostFunction.

#ifndef CERES_PUBLIC_INTERNAL_NUMERIC_DIFF_H_
#define CERES_PUBLIC_INTERNAL_NUMERIC_DIFF_H_

#include <cstring>

#include "Eigen/Dense"
#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/internal/variadic_evaluate.h"
#include "ceres/numeric_diff_options.h"
#include "ceres/types.h"
#include "glog/logging.h"


namespace ceres {
namespace internal {

// Helper templates that allow evaluation of a variadic functor or a
// CostFunction object.
template <typename CostFunctor,
          int N0, int N1, int N2, int N3, int N4,
          int N5, int N6, int N7, int N8, int N9 >
bool EvaluateImpl(const CostFunctor* functor,
                  double const* const* parameters,
                  double* residuals,
                  const void* /* NOT USED */) {
  return VariadicEvaluate<CostFunctor,
                          double,
                          N0, N1, N2, N3, N4, N5, N6, N7, N8, N9>::Call(
                              *functor,
                              parameters,
                              residuals);
}

template <typename CostFunctor,
          int N0, int N1, int N2, int N3, int N4,
          int N5, int N6, int N7, int N8, int N9 >
bool EvaluateImpl(const CostFunctor* functor,
                  double const* const* parameters,
                  double* residuals,
                  const CostFunction* /* NOT USED */) {
  return functor->Evaluate(parameters, residuals, NULL);
}

// This is split from the main class because C++ doesn't allow partial template
// specializations for member functions. The alternative is to repeat the main
// class for differing numbers of parameters, which is also unfortunate.
template <typename CostFunctor,
          NumericDiffMethodType kMethod,
          int kNumResiduals,
          int N0, int N1, int N2, int N3, int N4,
          int N5, int N6, int N7, int N8, int N9,
          int kParameterBlock,
          int kParameterBlockSize>
struct NumericDiff {
  // Mutates parameters but must restore them before return.
  static bool EvaluateJacobianForParameterBlock(
      const CostFunctor* functor,
      double const* residuals_at_eval_point,
      const NumericDiffOptions options,
      int num_residuals,
      int parameter_block_index,
      int parameter_block_size,
      double **parameters,
      double *jacobian) {
    using Eigen::Map;
    using Eigen::Matrix;
    using Eigen::RowMajor;
    using Eigen::ColMajor;

    const int num_residuals_internal =
        (kNumResiduals != ceres::DYNAMIC ? kNumResiduals : num_residuals);
    const int parameter_block_index_internal =
        (kParameterBlock != ceres::DYNAMIC ? kParameterBlock :
                                             parameter_block_index);
    const int parameter_block_size_internal =
        (kParameterBlockSize != ceres::DYNAMIC ? kParameterBlockSize :
                                                 parameter_block_size);

    typedef Matrix<double, kNumResiduals, 1> ResidualVector;
    typedef Matrix<double, kParameterBlockSize, 1> ParameterVector;

    // The convoluted reasoning for choosing the Row/Column major
    // ordering of the matrix is an artifact of the restrictions in
    // Eigen that prevent it from creating RowMajor matrices with a
    // single column. In these cases, we ask for a ColMajor matrix.
    typedef Matrix<double,
                   kNumResiduals,
                   kParameterBlockSize,
                   (kParameterBlockSize == 1) ? ColMajor : RowMajor>
        JacobianMatrix;

    Map<JacobianMatrix> parameter_jacobian(jacobian,
                                           num_residuals_internal,
                                           parameter_block_size_internal);

    Map<ParameterVector> x_plus_delta(
        parameters[parameter_block_index_internal],
        parameter_block_size_internal);
    ParameterVector x(x_plus_delta);
    ParameterVector step_size = x.array().abs() *
        ((kMethod == RIDDERS) ? options.adaptive_initial_step_size :
        options.relative_step_size);

    // It is not a good idea to make the step size arbitrarily
    // small. This will lead to problems with round off and numerical
    // instability when dividing by the step size. The general
    // recommendation is to not go down below sqrt(epsilon).
    // For Ridders' method, the initial step size is required to be large,
    // thus relative_step_size is used.
    const double min_step_size = ((kMethod == RIDDERS) ?
        options.adaptive_initial_step_size :
        std::sqrt(std::numeric_limits<double>::epsilon()));

    // For each parameter in the parameter block, use finite differences to
    // compute the derivative for that parameter.
    ResidualVector temp_residuals(num_residuals_internal);
    ResidualVector residuals(num_residuals_internal);
    for (int j = 0; j < parameter_block_size_internal; ++j) {
      const double delta = std::max(min_step_size, step_size(j));

      if (kMethod == RIDDERS) {
        if (!EvaluateAdaptiveNumericDiff(functor, j, delta,
                                         options,
                                         num_residuals_internal,
                                         parameter_block_size_internal,
                                         parameters, x.data(),
                                         x_plus_delta.data(),
                                         residuals_at_eval_point,
                                         temp_residuals.data(),
                                         residuals.data())) {
          return false;
        }
      } else {
        if (!EvaluateNumericDiff(functor, j, delta, num_residuals_internal,
                                 parameter_block_size_internal,
                                 parameters, x.data(), x_plus_delta.data(),
                                 residuals_at_eval_point,
                                 temp_residuals.data(),
                                 residuals.data())) {
          return false;
        }
      }

      parameter_jacobian.col(j).matrix() = residuals;
    }
    return true;
  }

  static bool EvaluateNumericDiff(const CostFunctor* functor,
                                  int parameter_index, double delta,
                                  int num_residuals,
                                  int parameter_block_size,
                                  double** parameters,
                                  double const* x_ptr,
                                  double* x_plus_delta_ptr,
                                  double const* residuals_at_eval_point,
                                  double* temp_residuals_ptr,
                                  double* residuals_ptr) {
    using Eigen::Map;
    using Eigen::Matrix;

    typedef Matrix<double, kNumResiduals, 1> ResidualVector;
    typedef Matrix<double, kParameterBlockSize, 1> ParameterVector;

    Map<const ParameterVector> x(x_ptr, parameter_block_size);
    Map<ParameterVector> x_plus_delta(x_plus_delta_ptr,
                                      parameter_block_size);

    Map<ResidualVector> residuals(residuals_ptr, num_residuals);
    Map<ResidualVector> temp_residuals(temp_residuals_ptr, num_residuals);

    // Mutate 1 element at a time and then restore.
    x_plus_delta(parameter_index) = x(parameter_index) + delta;

    if (!EvaluateImpl<CostFunctor, N0, N1, N2, N3, N4, N5, N6, N7, N8, N9>(
            functor, parameters, residuals.data(), functor)) {
      return false;
    }

    // Compute this column of the jacobian in 3 steps:
    // 1. Store residuals for the forward part.
    // 2. Subtract residuals for the backward (or 0) part.
    // 3. Divide out the run.
    double one_over_delta = 1.0 / delta;
    if (kMethod == CENTRAL || kMethod == RIDDERS) {
      // Compute the function on the other side of x(parameter_index).
      x_plus_delta(parameter_index) = x(parameter_index) - delta;

      if (!EvaluateImpl<CostFunctor, N0, N1, N2, N3, N4, N5, N6, N7, N8, N9>(
              functor, parameters, temp_residuals.data(), functor)) {
        return false;
      }

      residuals -= temp_residuals;
      one_over_delta /= 2;
    } else {
      // Forward difference only; reuse existing residuals evaluation.
      residuals -=
          Map<const ResidualVector>(residuals_at_eval_point,
                                    num_residuals);
    }

    // Restore x_plus_delta.
    x_plus_delta(parameter_index) = x(parameter_index);

    // Divide out the run to get slope.
    residuals *= one_over_delta;

    return true;
  }

  // This numeric difference implementation uses adaptive differentiation
  // on the parameters to obtain the Jacobian matrix. The adaptive algorithm
  // is based on Ridders' method for adaptive differentiation, which creates
  // a Romberg tableau from varying step sizes and extrapolates the
  // intermediate results to obtain the current computational error.
  //
  // References:
  // C.J.F. Ridders, Accurate computation of F'(x) and F'(x) F"(x), Advances
  // in Engineering Software (1978), Volume 4, Issue 2, April 1982,
  // Pages 75-76, ISSN 0141-1195,
  // http://dx.doi.org/10.1016/S0141-1195(82)80057-0.
  static bool EvaluateAdaptiveNumericDiff(
      const CostFunctor* functor,
      int parameter_index, double delta,
      const NumericDiffOptions options,
      int num_residuals,
      int parameter_block_size,
      double** parameters,
      double const* x_ptr,
      double* x_plus_delta_ptr,
      double const* residuals_at_eval_point,
      double* temp_residuals_ptr,
      double* residuals_ptr) {
    using Eigen::Map;
    using Eigen::Matrix;

    typedef Matrix<double, kNumResiduals, 1> ResidualVector;
    typedef Matrix<double, kParameterBlockSize, 1> ParameterVector;

    Map<const ParameterVector> x(x_ptr, parameter_block_size);
    Map<ParameterVector> x_plus_delta(x_plus_delta_ptr,
                                      parameter_block_size);

    Map<ResidualVector> residuals(residuals_ptr, num_residuals);
    Map<ResidualVector> temp_residuals(temp_residuals_ptr, num_residuals);

    // In order for the algorithm to converge, the step size 
    // should be initialized to a value that is large enough to produce
    // a significant change in the function.
    // As the derivative is estimated, the step size decreases.
    // By default, the step sizes are chosen so that the middle column
    // of the Romberg tableau uses the input delta.
    double current_step_size = delta *
        pow(options.adaptive_step_shrink_factor,
            options.max_adaptive_extrapolations / 2);

    // Double-buffering temporary differential candidate vectors
    // from previous step size.
    std::vector<ResidualVector> stepsize_candidates_a(
        options.max_adaptive_extrapolations);
    std::vector<ResidualVector> stepsize_candidates_b(
        options.max_adaptive_extrapolations);
    ResidualVector* current_candidates = &stepsize_candidates_a[0];
    ResidualVector* previous_candidates = &stepsize_candidates_b[0];

    // Initialize first candidate vector sizes.
    stepsize_candidates_a[0].resize(num_residuals);
    stepsize_candidates_b[0].resize(num_residuals);

    // Represents the computational error of the derivative. This variable
    // is initially set to a large value, and is set to the difference between
    // current and previous finite difference extrapolations.
    // norm_error is supposed to decrease as the finite difference 
    // tableau generation progresses, serving both as an estimate for 
    // differentiation error and as a measure of differentiation numerical
    // stability.
    double norm_error = std::numeric_limits<double>::max();

    // Loop over decreasing step sizes until:
    //  1. Error is smaller than a given value (adaptive_epsilon),
    //  2. Maximal order of extrapolation reached, or
    //  3. Extrapolation becomes numerically unstable.
    for (int i = 0; i < options.max_adaptive_extrapolations; ++i) {
      // Compute the numerical derivative at this step size.
      if (!EvaluateNumericDiff(functor, parameter_index, current_step_size,
                               num_residuals,
                               parameter_block_size,
                               parameters, x.data(), x_plus_delta.data(),
                               residuals_at_eval_point,
                               temp_residuals.data(),
                               current_candidates[0].data())) {
        // Something went wrong; bail.
        return false;
      }

      // Store initial results.
      if (i == 0) {
        residuals = current_candidates[0];
      }

      // Shrink differentiation step size.
      current_step_size /= options.adaptive_step_shrink_factor;

      double factor = options.adaptive_step_shrink_factor *
          options.adaptive_step_shrink_factor;
      for (int k = 1; k <= i; ++k) {
        current_candidates[k].resize(num_residuals);

        ResidualVector& candidate = current_candidates[k];

        // Extrapolate the various orders of finite differences using
        // the Richardson acceleration method.
        candidate =
            (factor * current_candidates[k - 1] -
             previous_candidates[k - 1]) / (factor - 1.0);

        factor *= options.adaptive_step_shrink_factor *
            options.adaptive_step_shrink_factor;

        // Compute the difference between the previous value and the current.
        double candidate_error;
        if (options.adaptive_relative_error) {
          // Compare using relative error: ||1 - a/b||.
          candidate_error = std::max(
              (1 - candidate.array() / current_candidates[k - 1].array()).
                  matrix().norm(),
              (1 - candidate.array() / previous_candidates[k - 1].array()).
                  matrix().norm());
        } else {
          // Compare values using absolute error: ||a - b||.
          candidate_error = std::max(
            (candidate - current_candidates[k - 1]).norm(),
            (candidate - previous_candidates[k - 1]).norm());
        }

        // If the error has decreased, update results.
        if (candidate_error <= norm_error) {
          norm_error = candidate_error;
          residuals = candidate;

          // If the error is small enough, stop.
          if (norm_error < options.adaptive_epsilon) {
            break;
          }
        }
      }

      // After breaking out of the inner loop, declare convergence.
      if (norm_error < options.adaptive_epsilon) {
        break;
      }

      // Check to see if the current gradient estimate is numerically unstable.
      // If so, bail out and return the last stable result.
      if (i > 0) {
        // Compare current error to chosen candidate error using ||1 - a/b||.
        if (options.adaptive_relative_error &&
            (1 - current_candidates[i].array() / previous_candidates[i - 1].
                array()).matrix().norm() >= 2 * norm_error) {
          break;
        // Compare current error to chosen candidate error using ||a - b||.
        } else if ((current_candidates[i] - 
                    previous_candidates[i - 1]).norm() >= 2 * norm_error) {
          break;
        }
      }

      std::swap(current_candidates, previous_candidates);
    }
    return true;
  }
};

template <typename CostFunctor,
          NumericDiffMethodType kMethod,
          int kNumResiduals,
          int N0, int N1, int N2, int N3, int N4,
          int N5, int N6, int N7, int N8, int N9,
          int kParameterBlock>
struct NumericDiff<CostFunctor, kMethod, kNumResiduals,
                   N0, N1, N2, N3, N4, N5, N6, N7, N8, N9,
                   kParameterBlock, 0> {
  // Mutates parameters but must restore them before return.
  static bool EvaluateJacobianForParameterBlock(
      const CostFunctor* functor,
      double const* residuals_at_eval_point,
      const NumericDiffOptions options,
      const int num_residuals,
      const int parameter_block_index,
      const int parameter_block_size,
      double **parameters,
      double *jacobian) {
    LOG(FATAL) << "Control should never reach here.";
    return true;
  }
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_NUMERIC_DIFF_H_
