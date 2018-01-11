// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)

#include "ceres/solver.h"

#include <limits>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/sized_cost_function.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"

namespace ceres {
namespace internal {

const double kUninitialized = 1e302;

// Generally multiple inheritance is a terrible idea, but in this (test)
// case it makes for a relatively elegant test implementation.
struct QuadraticCostFunctionAndEvaluationCallback :
      SizedCostFunction<1, 1>, EvaluationCallback  {

  explicit QuadraticCostFunctionAndEvaluationCallback(double *parameter)
      : EvaluationCallback(),
        parameter(parameter),
        prepare_num_calls(0),
        evaluate_num_calls(0),
        evaluate_last_parameter_value(kUninitialized) {}

  virtual ~QuadraticCostFunctionAndEvaluationCallback() {}

  // Evaluation callback interface. This checks that all the preconditions are
  // met at the point that Ceres calls into it.
  virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) {
    // Check: Prepare() & Evaluate() come in pairs, in that order.
    EXPECT_EQ(prepare_num_calls, evaluate_num_calls);

    // Check: new_evaluation_point indicates that the parameter has changed.
    if (new_evaluation_point) {
      // If it's a new evaluation point, then the parameter should have
      // changed. Technically, it's not required that it must change but
      // in practice it does, and that helps with testing.
      EXPECT_NE(evaluate_last_parameter_value, *parameter);
      EXPECT_NE(prepare_parameter_value, *parameter);
    } else {
      // If this is the same evaluation point as last time, ensure that
      // the parameters match both from the previous evaluate, the
      // previous prepare, and the current prepare.
      EXPECT_EQ(evaluate_last_parameter_value, prepare_parameter_value);
      EXPECT_EQ(evaluate_last_parameter_value, *parameter);
    }

    // Save details for to check at the next call to Evaluate().
    prepare_num_calls++;
    prepare_requested_jacobians = evaluate_jacobians;
    prepare_new_evaluation_point = new_evaluation_point;
    prepare_parameter_value = *parameter;
  }

  // Cost function interface. This checks that preconditions that were
  // set as part of the PrepareForEvaluation() call are met in this one.
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    // Cost function implementation.
    double x = **parameters;
    residuals[0] = 6.987654321 - x;
    if (jacobians != NULL) {
      **jacobians = - 1.0;
    }

    // Check: PrepareForEvaluation() & Evaluate() come in pairs, in that order.
    EXPECT_EQ(prepare_num_calls, evaluate_num_calls + 1);

    // Check: if new_evaluation_point indicates that the parameter has
    // changed, it has changed; otherwise it is the same.
    if (prepare_new_evaluation_point) {
      EXPECT_NE(evaluate_last_parameter_value, x);
    } else {
      EXPECT_NE(evaluate_last_parameter_value, kUninitialized);
      EXPECT_EQ(evaluate_last_parameter_value, x);
    }

    // Check: Parameter matches value in in parameter blocks during prepare.
    EXPECT_EQ(prepare_parameter_value, x);

    // Check: jacobians are requested if they were in PrepareForEvaluation().
    EXPECT_EQ(prepare_requested_jacobians, jacobians != NULL);

    evaluate_num_calls++;
    evaluate_last_parameter_value = x;
    return true;
  }

  // Pointer to the parameter block associated with this cost function.
  // Contents should get set by Ceres before calls to PrepareForEvaluation()
  // and Evaluate().
  double* parameter;


  // Track state: PrepareForEvaluation().
  //
  // These track details from the PrepareForEvaluation() call (hence the
  // "prepare_" prefix), which are checked for consistency in Evaluate().
  int prepare_num_calls;
  bool prepare_requested_jacobians;
  bool prepare_new_evaluation_point;
  double prepare_parameter_value;

  // Track state: Evaluate().
  //
  // These track details from the Evaluate() call (hence the "evaluate_"
  // prefix), which are then checked for consistency in the calls to
  // PrepareForEvaluation(). Mutable is reasonable for this case.
  mutable int evaluate_num_calls;
  mutable double evaluate_last_parameter_value;
};

TEST(EvaluationCallback, WithTrustRegionMinimizer) {
  double x = 50.123456789;
  const double original_x = x;

  QuadraticCostFunctionAndEvaluationCallback cost_function(&x);
  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  Problem problem(problem_options);
  problem.AddResidualBlock(&cost_function, NULL, &x);

  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.evaluation_callback = &cost_function;

  Solver::Summary summary;

  // Run the solve. Checking is done inside the cost function / callback.
  Solve(options, &problem, &summary);

  LOG(ERROR) << summary.FullReport();

  int num_iterations = summary.num_successful_steps +
                       summary.num_unsuccessful_steps;
  EXPECT_GT(num_iterations, 1);
  EXPECT_GT(cost_function.prepare_num_calls, 0);
  EXPECT_GT(cost_function.evaluate_num_calls, 0);
  EXPECT_EQ(cost_function.prepare_num_calls,
            cost_function.evaluate_num_calls);
  EXPECT_NE(x, original_x);
}

TEST(EvaluationCallback, WithLineSearchMinimizer) {
  double x = 50.123456789;
  const double original_x = x;

  QuadraticCostFunctionAndEvaluationCallback cost_function(&x);
  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  Problem problem(problem_options);
  problem.AddResidualBlock(&cost_function, NULL, &x);

  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.minimizer_type = ceres::LINE_SEARCH;
  options.evaluation_callback = &cost_function;

  Solver::Summary summary;

  // Run the solve. Checking is done inside the cost function / callback.
  Solve(options, &problem, &summary);

  LOG(ERROR) << summary.FullReport();

  int num_iterations = summary.num_successful_steps +
                       summary.num_unsuccessful_steps;
  EXPECT_GT(num_iterations, 1);
  EXPECT_GT(cost_function.prepare_num_calls, 0);
  EXPECT_GT(cost_function.evaluate_num_calls, 0);
  EXPECT_EQ(cost_function.prepare_num_calls,
            cost_function.evaluate_num_calls);
  EXPECT_NE(x, original_x);
}
}  // namespace internal
}  // namespace ceres
