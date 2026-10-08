// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2024 Google Inc. All rights reserved.
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
// Author: thadh@gmail.com (Thad Hughes)
//         mierle@gmail.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/dynamic_autodiff_cost_function.h"

#include <memory>
#include <utility>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/dynamic_cost_function_test_utils.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres::internal {

struct AutoDiffTraits {
  template <typename Functor>
  using CostFunction = DynamicAutoDiffCostFunction<Functor, 3>;
  static constexpr double kTolerance = 0.0;
};

INSTANTIATE_TYPED_TEST_SUITE_P(DynamicAutoDiffCostFunction,
                               DynamicCostFunctionTest,
                               ::testing::Types<AutoDiffTraits>);

class ValueError {
 public:
  explicit ValueError(double target_value) : target_value_(target_value) {}

  template <typename T>
  bool operator()(const T* value, T* residual) const {
    *residual = *value - T(target_value_);
    return true;
  }

 protected:
  double target_value_;
};

class DynamicValueError {
 public:
  explicit DynamicValueError(double target_value)
      : target_value_(target_value) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residual) const {
    residual[0] = T(target_value_) - parameters[0][0];
    return true;
  }

 protected:
  double target_value_;
};

TEST(DynamicAutoDiffCostFunction,
     EvaluateWithEmptyJacobiansArrayComputesResidual) {
  const double target_value = 1.0;
  double parameter = 0;
  ceres::DynamicAutoDiffCostFunction<DynamicValueError, 1> cost_function(
      new DynamicValueError(target_value));
  cost_function.AddParameterBlock(1);
  cost_function.SetNumResiduals(1);

  double* parameter_blocks[1] = {&parameter};
  double* jacobians[1] = {nullptr};
  double residual;

  ASSERT_TRUE(cost_function.Evaluate(parameter_blocks, &residual, jacobians));
  EXPECT_EQ(residual, target_value);
}

TEST(DynamicAutoDiffCostFunctionTest, DeductionTemplateCompilationTest) {
  // Ensure deduction guide to be working
  (void)DynamicAutoDiffCostFunction(new MyCostFunctor());
  (void)DynamicAutoDiffCostFunction(new MyCostFunctor(), TAKE_OWNERSHIP);
  (void)DynamicAutoDiffCostFunction(std::make_unique<MyCostFunctor>());
}

TEST(DynamicAutoDiffCostFunctionTest, ArgumentForwarding) {
  (void)DynamicAutoDiffCostFunction<MyCostFunctor>();
}

TEST(DynamicAutoDiffCostFunctionTest, UniquePtr) {
  (void)DynamicAutoDiffCostFunction(std::make_unique<MyCostFunctor>());
}

TEST(DynamicAutoDiffCostFunctionTest, Ownership) {
  MyCostFunctor functor;
  {
    DynamicAutoDiffCostFunction<MyCostFunctor> cost_function(
        &functor, DO_NOT_TAKE_OWNERSHIP);
    cost_function.AddParameterBlock(3);
    cost_function.SetNumResiduals(3);
  }
}

TEST(DynamicAutoDiffCostFunctionTest, ExplicitUniquePtr) {
  auto functor = std::make_unique<MyCostFunctor>();
  DynamicAutoDiffCostFunction<MyCostFunctor> cost_function(std::move(functor));
  cost_function.AddParameterBlock(3);
  cost_function.SetNumResiduals(3);
}

}  // namespace ceres::internal
