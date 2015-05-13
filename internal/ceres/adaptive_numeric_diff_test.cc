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
// Author: tbennun@gmail.com (Tal Ben-Nun)

#include "ceres/numeric_diff_cost_function.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "ceres/internal/macros.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/cost_function.h"
#include "ceres/numeric_diff_test_utils.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

const double kNoiseFactor = 0.01;

class ExponentialFunctor {
 public:
  bool operator()(const double* x1, double* residuals) const {
      residuals[0] = exp(x1[0]);
      return true;
  }

  void ExpectCostFunctionEvaluationIsNearlyCorrect(
      const CostFunction& cost_function) const {
    double x[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    double *parameters[] = { &x[0] };

    double dydx[5];
    double *jacobians[] = { &dydx[0] };

    double residuals[5] = { -1e-100, -2e-100, -3e-100, -4e-100, -5e-100 };

    const double tolerance = 1e-12;
    
    // Evaluate functor at all locations.
    for (int i = 0; i < 5; ++i) {
      parameters[0] = &x[i];
      jacobians[0] = &dydx[i];
      ASSERT_TRUE(cost_function.Evaluate(&parameters[0],
                                         &residuals[i],
                                         &jacobians[0]));

      ExpectClose(residuals[i], exp(x[i]), tolerance);
    }

    // Check evaluated differences.
    for (int i = 0; i < 5; ++i)
      ExpectClose(exp(x[i]), dydx[i], tolerance);
  }
};

class ExponentialCostFunction : public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** /* not used */) const {
    return functor_(parameters[0], residuals);
  }

 private:
  ExponentialFunctor functor_;
};

// Test adaptive numeric difference by synthetically adding random noise
// to the functor.
class RandomizedFunctor {
 public:
  RandomizedFunctor(double noise_factor) : noise_factor_(noise_factor) {
  }

  bool operator()(const double* x1, double* residuals) const {
      double random_value = static_cast<double>(rand()) /
                            static_cast<double>(RAND_MAX);
      // Normalize noise to [-factor, factor]
      random_value *= 2.0;
      random_value -= 1.0;
      random_value *= noise_factor_;

      residuals[0] = x1[0] * x1[0] + random_value;
      return true;
  }

  void ExpectCostFunctionEvaluationIsNearlyCorrect(
      const CostFunction& cost_function) const {
    double x[] = { 0.0, 1.0, 3.0, 4.0, 5.0 };
    double *parameters[] = { &x[0] };

    double dydx[5];
    double *jacobians[] = { &dydx[0] };

    double residuals[5] = { -1e-100, -2e-100, -3e-100, -4e-100, -5e-100 };

    // Evaluate functor at all locations.
    for (int i = 0; i < 5; ++i) {
      parameters[0] = &x[i];
      jacobians[0] = &dydx[i];
      ASSERT_TRUE(cost_function.Evaluate(&parameters[0],
                                         &residuals[i],
                                         &jacobians[0]));

      ExpectClose(residuals[i], x[i] * x[i], noise_factor_);
    }

    const double tolerance = 2e-4;

    // Check evaluated differences.
    for (int i = 0; i < 5; ++i)
      ExpectClose(2.0 * x[i], dydx[i], tolerance);
  }

 private:
  double noise_factor_;
};

class RandomizedCostFunction : public SizedCostFunction<1, 1> {
 public:
  RandomizedCostFunction(double noise_factor) : functor_(noise_factor) {
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** /* not used */) const {
    return functor_(parameters[0], residuals);
  }

 private:
  RandomizedFunctor functor_;
};

// Test fixture to set up randomization.
class AdaptiveNumericDiff : public ::testing::Test {
 protected:
  virtual void SetUp() {
    srand(static_cast<unsigned int>(time(NULL)));
  }
};

TEST_F(AdaptiveNumericDiff, ExponentialFunctorCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<ExponentialFunctor,
                                  ADAPTIVE,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new ExponentialFunctor, TAKE_OWNERSHIP, 1, 1e-2));
  ExponentialFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST_F(AdaptiveNumericDiff, ExponentialCostFunctionCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<ExponentialCostFunction,
                                  ADAPTIVE,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new ExponentialCostFunction, TAKE_OWNERSHIP, 1, 1e-2));
  ExponentialFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST_F(AdaptiveNumericDiff, RandomizedFunctorCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<RandomizedFunctor,
                                  ADAPTIVE,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new RandomizedFunctor(kNoiseFactor), TAKE_OWNERSHIP,
             1, 1000.0));
  RandomizedFunctor functor (kNoiseFactor);
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST_F(AdaptiveNumericDiff, RandomizedCostFunctionCentralDifferences) {
  internal::scoped_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<RandomizedCostFunction,
                                  ADAPTIVE,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new RandomizedCostFunction(kNoiseFactor), TAKE_OWNERSHIP,
             1, 1000.0));
  RandomizedFunctor functor (kNoiseFactor);
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

}  // namespace internal
}  // namespace ceres
