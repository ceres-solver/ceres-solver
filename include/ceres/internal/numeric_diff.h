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
// Finite differencing routine used by NumericDiffCostFunction.

#ifndef CERES_PUBLIC_INTERNAL_NUMERIC_DIFF_H_
#define CERES_PUBLIC_INTERNAL_NUMERIC_DIFF_H_

#include <cstring>

#include "Eigen/Dense"
#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/internal/variadic_evaluate.h"
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
          NumericDiffMethod kMethod,
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
      const double relative_step_size,
      const int adaptive_max_extrapolations,
      const double adaptive_epsilon,
      const double adaptive_step_shrink_factor,
      const bool adaptive_relative_error,
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
    ParameterVector step_size = x.array().abs() * relative_step_size;

    // To handle cases where a parameter is exactly zero, instead use
    // the mean step_size for the other dimensions. If all the
    // parameters are zero, there's no good answer. Take
    // relative_step_size as a guess and hope for the best.
    const double fallback_step_size =
        (step_size.sum() == 0)
        ? relative_step_size
        : step_size.sum() / step_size.rows();

    // For each parameter in the parameter block, use finite differences to
    // compute the derivative for that parameter.

    ResidualVector temp_residuals(num_residuals_internal);
    ResidualVector residuals(num_residuals_internal);
    for (int j = 0; j < parameter_block_size_internal; ++j) {
      const double delta =
          (step_size(j) == 0.0) ? fallback_step_size : step_size(j);

      if (kMethod == ADAPTIVE) {
        if (!EvaluateAdaptiveNumericDiff(functor, j, delta,
                                         adaptive_max_extrapolations,
                                         adaptive_epsilon,
                                         adaptive_step_shrink_factor,
                                         adaptive_relative_error,
                                         num_residuals_internal,
                                         parameter_block_size_internal,
                                         parameters, x.data(),
                                         x_plus_delta.data(),
                                         residuals_at_eval_point,
                                         temp_residuals.data(),
                                         residuals.data()))
          return false;
      } else {
        if (!EvaluateNumericDiff(functor, j, delta, num_residuals_internal,
                                 parameter_block_size_internal,
                                 parameters, x.data(), x_plus_delta.data(),
                                 residuals_at_eval_point,
                                 temp_residuals.data(),
                                 residuals.data()))
          return false;
      }

      // Store the obtained value.
      parameter_jacobian.col(j).matrix() = residuals;
    }
    return true;
  }

  static bool EvaluateNumericDiff(const CostFunctor* functor,
                                  int ind, double delta,
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
    x_plus_delta(ind) = x(ind) + delta;

    if (!EvaluateImpl<CostFunctor, N0, N1, N2, N3, N4, N5, N6, N7, N8, N9>(
            functor, parameters, residuals.data(), functor)) {
      return false;
    }

    // Compute this column of the jacobian in 3 steps:
    // 1. Store residuals for the forward part.
    // 2. Subtract residuals for the backward (or 0) part.
    // 3. Divide out the run.
    double one_over_delta = 1.0 / delta;
    if (kMethod == CENTRAL || kMethod == ADAPTIVE) {
      // Compute the function on the other side of x(ind).
      x_plus_delta(ind) = x(ind) - delta;

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
    x_plus_delta(ind) = x(ind);  // Restore x_plus_delta.

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
      int ind, double delta,
      int adaptive_max_extrapolations,
      double adaptive_epsilon,
      double adaptive_step_shrink_factor,
      bool adaptive_relative_error,
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

    // Initialize parameters for adaptive differentiation.
    double current_step_size = delta;

    // Initialize constant vector of ones (for error computations).
    ResidualVector ones(num_residuals);
    ones.setConstant(1.0);

    // Double-buffering temporary differential candidate vectors
    // from previous step size.
    std::vector<ResidualVector> stepsize_candidates_a
      (adaptive_max_extrapolations);
    std::vector<ResidualVector> stepsize_candidates_b
      (adaptive_max_extrapolations);
    ResidualVector* cur_candidates = &stepsize_candidates_a[0];
    ResidualVector* prev_candidates = &stepsize_candidates_b[0];

    double norm_err = std::numeric_limits<double>::max();

    // Loop over decreasing step sizes until:
    //  1. Error is smaller than a given value (epsilon), or
    //  2. Maximal order of extrapolation reached, or
    //  3. Extrapolation becomes numerically unstable.
    for(int i = 0; i < adaptive_max_extrapolations; ++i) {
      cur_candidates[0].resize(num_residuals);

      // Compute the numerical derivative at this step size.
      if (!EvaluateNumericDiff(functor, ind, current_step_size,
                               num_residuals,
                               parameter_block_size,
                               parameters, x.data(), x_plus_delta.data(),
                               residuals_at_eval_point,
                               temp_residuals.data(),
                               cur_candidates[0].data())) {
        // Something went wrong; bail.
        return false;
      }

      // Store initial results.
      if (i == 0)
        residuals = cur_candidates[0];

      // Shrink differentiation step size.
      current_step_size /= adaptive_step_shrink_factor;

      double factor = adaptive_step_shrink_factor *
        adaptive_step_shrink_factor;
      for(int k = 1; k <= i; ++k) {
        cur_candidates[k].resize(num_residuals);

        ResidualVector& candidate = cur_candidates[k];

        // Extrapolate various orders of finite differences
        // using Neville's algorithm.
        candidate =
          (factor * cur_candidates[k - 1] -
           prev_candidates[k - 1]) / (factor - 1.0);

        factor *= adaptive_step_shrink_factor * adaptive_step_shrink_factor;

        // Compute the difference between the previous
        // value and the current.
        double temp_error;
        if(adaptive_relative_error) {
          // Compare using relative error: (1 - a/b).
          temp_error = std::max(
            (ones - candidate.cwiseQuotient(cur_candidates[k - 1])).norm(),
            (ones - candidate.cwiseQuotient(prev_candidates[k - 1])).norm());
        } else {
          // Compare values using absolute error: (a - b).
          temp_error = std::max(
            (candidate - cur_candidates[k - 1]).norm(),
            (candidate - prev_candidates[k - 1]).norm());
        }

          residuals = candidate;
        // If the error has decreased, update results.
        if(temp_error <= norm_err) {
          norm_err = temp_error;
          residuals = candidate;

          // If the error is small enough, stop.
          if(norm_err < adaptive_epsilon)
            break;
        }
      }

      // After breaking out of the inner loop, declare convergence.
      if(norm_err < adaptive_epsilon)
        break;

      // Testing for numerical stability of the results.
      if(i > 0) {
        if(adaptive_relative_error) {
          if((ones - cur_candidates[i].cwiseQuotient(
              prev_candidates[i - 1])).norm() >= 2 * norm_err) {
            break;
          }
        } else {
          if((cur_candidates[i] - prev_candidates[i - 1]).norm() >=
             2 * norm_err) {
            break;
          }
        }
      }

      // Swap the two vector pointers.
      std::swap(cur_candidates, prev_candidates);
    }
    return true;
  }
};

template <typename CostFunctor,
          NumericDiffMethod kMethod,
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
      const double relative_step_size,
      const int adaptive_max_extrapolations,
      const double adaptive_epsilon,
      const double adaptive_step_shrink_factor,
      const bool adaptive_relative_error,
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
