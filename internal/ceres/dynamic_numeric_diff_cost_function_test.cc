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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//         mierle@gmail.com (Keir Mierle)

#include "ceres/dynamic_numeric_diff_cost_function.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "ceres/dynamic_cost_function_test_utils.h"
#include "ceres/numeric_diff_options.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres::internal {

struct NumericDiffTraits {
  template <typename Functor>
  using CostFunction = DynamicNumericDiffCostFunction<Functor>;
  static constexpr double kTolerance = 1e-6;
};

INSTANTIATE_TYPED_TEST_SUITE_P(DynamicNumericDiffCostFunction,
                               DynamicCostFunctionTest,
                               ::testing::Types<NumericDiffTraits>);

TEST(DynamicNumericDiffCostFunctionTest, DeductionTemplateCompilationTest) {
  // Ensure deduction guide to be working
  (void)DynamicNumericDiffCostFunction{std::make_unique<MyCostFunctor>()};
  (void)DynamicNumericDiffCostFunction{std::make_unique<MyCostFunctor>(),
                                       NumericDiffOptions{}};
  (void)DynamicNumericDiffCostFunction{new MyCostFunctor};
  (void)DynamicNumericDiffCostFunction{new MyCostFunctor, TAKE_OWNERSHIP};
  (void)DynamicNumericDiffCostFunction{
      new MyCostFunctor, TAKE_OWNERSHIP, NumericDiffOptions{}};
}

TEST(DynamicNumericDiffCostFunctionTest, ArgumentForwarding) {
  (void)DynamicNumericDiffCostFunction<MyCostFunctor>();
}

TEST(DynamicNumericDiffCostFunctionTest, UniquePtr) {
  (void)DynamicNumericDiffCostFunction<MyCostFunctor>(
      std::make_unique<MyCostFunctor>());
}

TEST(DynamicNumericDiffCostFunctionTest, Ownership) {
  MyCostFunctor functor;
  {
    DynamicNumericDiffCostFunction<MyCostFunctor> cost_function(
        &functor, DO_NOT_TAKE_OWNERSHIP);
    cost_function.AddParameterBlock(3);
    cost_function.SetNumResiduals(3);
  }
}

}  // namespace ceres::internal
