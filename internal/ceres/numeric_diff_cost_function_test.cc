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
// Author: keir@google.com (Keir Mierle)
//         tbennun@gmail.com (Tal Ben-Nun)

#include "ceres/numeric_diff_cost_function.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "ceres/array_utils.h"
#include "ceres/numeric_diff_test_utils.h"
#include "ceres/test_util.h"
#include "ceres/types.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(NumericDiffCostFunction, EasyCaseFunctorCentralDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  CENTRAL,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyFunctor));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, EasyCaseFunctorForwardDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  FORWARD,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyFunctor));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, EasyCaseFunctorRidders) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  RIDDERS,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyFunctor));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, RIDDERS);
}

TEST(NumericDiffCostFunction, EasyCaseCostFunctionCentralDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyCostFunction,
                                  CENTRAL,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyCostFunction, TAKE_OWNERSHIP));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, EasyCaseCostFunctionForwardDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyCostFunction,
                                  FORWARD,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyCostFunction, TAKE_OWNERSHIP));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction, EasyCaseCostFunctionRidders) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyCostFunction,
                                  RIDDERS,
                                  3,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new EasyCostFunction, TAKE_OWNERSHIP));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, RIDDERS);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseFunctorCentralDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalFunctor,
                                  CENTRAL,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalFunctor));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseFunctorForwardDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalFunctor,
                                  FORWARD,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalFunctor));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseFunctorRidders) {
  NumericDiffOptions options;

  // Using a smaller initial step size to overcome oscillatory function
  // behavior.
  options.ridders_relative_initial_step_size = 1e-3;

  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalFunctor,
                                  RIDDERS,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalFunctor, TAKE_OWNERSHIP, 2, options));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, RIDDERS);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseCostFunctionCentralDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalCostFunction,
                                  CENTRAL,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalCostFunction, TAKE_OWNERSHIP));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseCostFunctionForwardDifferences) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalCostFunction,
                                  FORWARD,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalCostFunction, TAKE_OWNERSHIP));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, FORWARD);
}

TEST(NumericDiffCostFunction,
     TranscendentalCaseCostFunctionRidders) {
  NumericDiffOptions options;

  // Using a smaller initial step size to overcome oscillatory function
  // behavior.
  options.ridders_relative_initial_step_size = 1e-3;

  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<TranscendentalCostFunction,
                                  RIDDERS,
                                  2,  /* number of residuals */
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
          new TranscendentalCostFunction, TAKE_OWNERSHIP, 2, options));
  TranscendentalFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, RIDDERS);
}

template<int num_rows, int num_cols>
class SizeTestingCostFunction : public SizedCostFunction<num_rows, num_cols> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    return true;
  }
};

// As described in
// http://forum.kde.org/viewtopic.php?f=74&t=98536#p210774
// Eigen3 has restrictions on the Row/Column major storage of vectors,
// depending on their dimensions. This test ensures that the correct
// templates are instantiated for various shapes of the Jacobian
// matrix.
TEST(NumericDiffCostFunction, EigenRowMajorColMajorTest) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<1,1>,  CENTRAL, 1, 1>(
          new SizeTestingCostFunction<1,1>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<2,1>,  CENTRAL, 2, 1>(
          new SizeTestingCostFunction<2,1>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<1,2>,  CENTRAL, 1, 2>(
          new SizeTestingCostFunction<1,2>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<SizeTestingCostFunction<2,2>,  CENTRAL, 2, 2>(
          new SizeTestingCostFunction<2,2>, ceres::TAKE_OWNERSHIP));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 1, 1>(
          new EasyFunctor, TAKE_OWNERSHIP, 1));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 1, 1>(
          new EasyFunctor, TAKE_OWNERSHIP, 2));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 1, 2>(
          new EasyFunctor, TAKE_OWNERSHIP, 1));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 1, 2>(
          new EasyFunctor, TAKE_OWNERSHIP, 2));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 2, 1>(
          new EasyFunctor, TAKE_OWNERSHIP, 1));

  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor, CENTRAL, ceres::DYNAMIC, 2, 1>(
          new EasyFunctor, TAKE_OWNERSHIP, 2));
}

TEST(NumericDiffCostFunction,
     EasyCaseFunctorCentralDifferencesAndDynamicNumResiduals) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<EasyFunctor,
                                  CENTRAL,
                                  ceres::DYNAMIC,
                                  5,  /* size of x1 */
                                  5   /* size of x2 */>(
                                      new EasyFunctor, TAKE_OWNERSHIP, 3));
  EasyFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function, CENTRAL);
}

TEST(NumericDiffCostFunction, ExponentialFunctorRidders) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<ExponentialFunctor,
                                  RIDDERS,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new ExponentialFunctor));
  ExponentialFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST(NumericDiffCostFunction, ExponentialCostFunctionRidders) {
  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(
      new NumericDiffCostFunction<ExponentialCostFunction,
                                  RIDDERS,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new ExponentialCostFunction));
  ExponentialFunctor functor;
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST(NumericDiffCostFunction, RandomizedFunctorRidders) {
  std::unique_ptr<CostFunction> cost_function;
  NumericDiffOptions options;
  // Larger initial step size is chosen to produce robust results in the
  // presence of random noise.
  options.ridders_relative_initial_step_size = 10.0;

  cost_function.reset(
      new NumericDiffCostFunction<RandomizedFunctor,
                                  RIDDERS,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new RandomizedFunctor(kNoiseFactor, kRandomSeed), TAKE_OWNERSHIP,
             1, options));
  RandomizedFunctor functor (kNoiseFactor, kRandomSeed);
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

TEST(NumericDiffCostFunction, RandomizedCostFunctionRidders) {
  std::unique_ptr<CostFunction> cost_function;
  NumericDiffOptions options;
  // Larger initial step size is chosen to produce robust results in the
  // presence of random noise.
  options.ridders_relative_initial_step_size = 10.0;

  cost_function.reset(
      new NumericDiffCostFunction<RandomizedCostFunction,
                                  RIDDERS,
                                  1,  /* number of residuals */
                                  1   /* size of x1 */>(
             new RandomizedCostFunction(kNoiseFactor, kRandomSeed),
             TAKE_OWNERSHIP, 1, options));
  RandomizedFunctor functor (kNoiseFactor, kRandomSeed);
  functor.ExpectCostFunctionEvaluationIsNearlyCorrect(*cost_function);
}

struct OnlyFillsOneOutputFunctor {
  bool operator()(const double* x, double* output) const {
    output[0] = x[0];
    return true;
  }
};

TEST(NumericDiffCostFunction, PartiallyFilledResidualShouldFailEvaluation) {
  double parameter = 1.0;
  double jacobian[2];
  double residuals[2];
  double* parameters[] = {&parameter};
  double* jacobians[] = {jacobian};

  std::unique_ptr<CostFunction> cost_function(
      new NumericDiffCostFunction<OnlyFillsOneOutputFunctor, CENTRAL, 2, 1>(
          new OnlyFillsOneOutputFunctor));
  InvalidateArray(2, jacobian);
  InvalidateArray(2, residuals);
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, jacobians));
  EXPECT_FALSE(IsArrayValid(2, residuals));
  InvalidateArray(2, residuals);
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals, NULL));
  // We are only testing residuals here, because the Jacobians are
  // computed using finite differencing from the residuals, so unless
  // we introduce a validation step after every evaluation of
  // residuals inside NumericDiffCostFunction, there is no way of
  // ensuring that the Jacobian array is invalid.
  EXPECT_FALSE(IsArrayValid(2, residuals));
}

TEST(NumericDiffCostFunction, ParameterBlockConstant) {
  constexpr int kNumResiduals = 3;
  constexpr int kX1 = 5;
  constexpr int kX2 = 5;

  std::unique_ptr<CostFunction> cost_function;
  cost_function.reset(new NumericDiffCostFunction<EasyFunctor,
                                                  CENTRAL,
                                                  kNumResiduals,
                                                  kX1,
                                                  kX2>(new EasyFunctor));

  // Prepare the parameters and residuals.
  std::array<double, kX1> x1{1e-64, 2.0, 3.0, 4.0, 5.0};
  std::array<double, kX2> x2{9.0, 9.0, 5.0, 5.0, 1.0};
  std::array<double*, 2> parameter_blocks{x1.data(), x2.data()};

  std::vector<double> residuals(kNumResiduals, -100000);

  // Evaluate the full jacobian.
  std::vector<std::vector<double>> jacobian_full_vect(2);
  jacobian_full_vect[0].resize(kNumResiduals * kX1, -100000);
  jacobian_full_vect[1].resize(kNumResiduals * kX2, -100000);
  {
    std::array<double*, 2> jacobian{jacobian_full_vect[0].data(),
                                    jacobian_full_vect[1].data()};
    ASSERT_TRUE(cost_function->Evaluate(
        parameter_blocks.data(), residuals.data(), jacobian.data()));
  }

  // Evaluate and check jacobian when first parameter block is constant.
  {
    std::vector<double> jacobian_vect(kNumResiduals * kX2, -100000);
    std::array<double*, 2> jacobian{nullptr, jacobian_vect.data()};

    ASSERT_TRUE(cost_function->Evaluate(
        parameter_blocks.data(), residuals.data(), jacobian.data()));

    for (int i = 0; i < kNumResiduals * kX2; ++i) {
      EXPECT_DOUBLE_EQ(jacobian_full_vect[1][i], jacobian_vect[i]);
    }
  }

  // Evaluate and check jacobian when second parameter block is constant.
  {
    std::vector<double> jacobian_vect(kNumResiduals * kX1, -100000);
    std::array<double*, 2> jacobian{jacobian_vect.data(), nullptr};

    ASSERT_TRUE(cost_function->Evaluate(
        parameter_blocks.data(), residuals.data(), jacobian.data()));

    for (int i = 0; i < kNumResiduals * kX1; ++i) {
      EXPECT_DOUBLE_EQ(jacobian_full_vect[0][i], jacobian_vect[i]);
    }
  }
}

}  // namespace internal
}  // namespace ceres
