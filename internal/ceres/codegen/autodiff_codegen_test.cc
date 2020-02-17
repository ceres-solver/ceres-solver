// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// This file tests the Expression class. For each member function one test is
// included here.
//
#include "autodiff_codegen_test.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/codegen/internal/expression.h"
#include "ceres/internal/autodiff.h"
#include "ceres/random.h"
#include "gtest/gtest.h"
#include "test_util.h"
namespace ceres {
namespace internal {

// This struct is used to convert combined cost functions (evaluate method +
// operator()) to cost functors (only operator()). As a result, putting such a
// type into ceres::Autodiff will use the operator() instead of evaluate().
template <typename CostFunction>
struct CostFunctionToFunctor {
  template <typename... _Args>
  CostFunctionToFunctor(_Args&&... __args)
      : cost_function(std::forward<_Args>(__args)...) {}

  template <typename... _Args>
  bool operator()(_Args&&... __args) const {
    return cost_function(std::forward<_Args>(__args)...);
  }

  CostFunction cost_function;
};

// Similar to ExpectClose in test_util.cc, but with inf and nan checks.
static bool ExpectCloseWithInfNan(double x,
                                  double y,
                                  double max_abs_relative_difference) {
  // If both are inf or nan it's fine too!
  if (std::isinf(x) && std::isinf(y)) {
    return true;
  }

  if (std::isnan(x) && std::isnan(y)) {
    return true;
  }

  double absolute_difference = fabs(x - y);
  double relative_difference = absolute_difference / std::max(fabs(x), fabs(y));
  if (x == 0 || y == 0) {
    // If x or y is exactly zero, then relative difference doesn't have any
    // meaning. Take the absolute difference instead.
    relative_difference = absolute_difference;
  }
  if (relative_difference > max_abs_relative_difference) {
    VLOG(1) << StringPrintf("x=%17g y=%17g abs=%17g rel=%17g",
                            x,
                            y,
                            absolute_difference,
                            relative_difference);
  }

  EXPECT_NEAR(relative_difference, 0.0, max_abs_relative_difference);
  return relative_difference <= max_abs_relative_difference;
}

template <int kNumResiduals, int... Ns>
std::pair<std::vector<double>, std::vector<double>> EvaluateCostFunction(
    CostFunction* f1, bool random_values = true, double value = 0) {
  using Params = StaticParameterDims<Ns...>;

  std::vector<double> params_array(Params::kNumParameters);
  std::vector<double*> params(Params::kNumParameters);
  std::vector<double> residuals_0(kNumResiduals, 0);
  std::vector<double> jacobians_array_0(kNumResiduals * Params::kNumParameters,
                                        0);

  for (auto& p : params_array) {
    if (random_values) {
      p = ceres::RandDouble() * 2.0 - 1.0;
    } else {
      p = value;
    }
  }
  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    params[i] = &params_array[k];
  }

  std::vector<double*> jacobians_0(Params::kNumParameterBlocks);
  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    jacobians_0[i] = &jacobians_array_0[k * kNumResiduals];
  }

  f1->Evaluate(params.data(), residuals_0.data(), jacobians_0.data());

  return std::make_pair(residuals_0, jacobians_array_0);
}

template <int kNumResiduals, int... Ns>
void compare_cost_functions(CostFunction* f1,
                            CostFunction* f2,
                            bool random_values = true,
                            double value = 0) {
  ceres::SetRandomState(956113);
  auto residuals_jacobians_1 =
      EvaluateCostFunction<kNumResiduals, Ns...>(f1, random_values, value);
  ceres::SetRandomState(956113);
  auto residuals_jacobians_2 =
      EvaluateCostFunction<kNumResiduals, Ns...>(f2, random_values, value);

  for (int i = 0; i < residuals_jacobians_1.first.size(); ++i) {
    ExpectCloseWithInfNan(
        residuals_jacobians_1.first[i], residuals_jacobians_2.first[i], 1e-20);
  }
  for (int i = 0; i < residuals_jacobians_1.second.size(); ++i) {
    ExpectCloseWithInfNan(residuals_jacobians_1.second[i],
                          residuals_jacobians_2.second[i],
                          1e-20);
  }
}

template <typename FunctorType, int kNumResiduals, int... Ns>
void TestFunctor() {
  FunctorType cost_function_generated;
  CostFunctionToFunctor<FunctorType> cost_functor;
  auto* cost_function_ad =
      new AutoDiffCostFunction<CostFunctionToFunctor<FunctorType>,
                               kNumResiduals,
                               Ns...>(&cost_functor);

  // Run the tests with a few fixed values to check edge-cases
  std::vector<double> input_values = {0,
                                      -1,
                                      1,
                                      -10,
                                      10,
                                      1e-50,
                                      1e50,
                                      std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::quiet_NaN()};
  for (auto v : input_values) {
    compare_cost_functions<kNumResiduals, Ns...>(
        &cost_function_generated, cost_function_ad, false, v);
  }

  // Run N times with random values in the range [-1,1]
  for (int i = 0; i < 100; ++i) {
    compare_cost_functions<kNumResiduals, Ns...>(
        &cost_function_generated, cost_function_ad, true);
  }
}

TEST(AutodiffCodeGen, InputOutputAssignment) {
  TestFunctor<test::InputOutputAssignment, 7, 4, 2, 1>();
}

TEST(AutodiffCodeGen, CompileTimeConstants) {
  TestFunctor<test::CompileTimeConstants, 7, 1>();
}

TEST(AutodiffCodeGen, Assignments) { TestFunctor<test::Assignments, 8, 2>(); }
TEST(AutodiffCodeGen, BinaryArithmetic) {
  TestFunctor<test::BinaryArithmetic, 9, 2>();
}
TEST(AutodiffCodeGen, UnaryArithmetic) {
  TestFunctor<test::UnaryArithmetic, 3, 1>();
}
TEST(AutodiffCodeGen, BinaryComparison) {
  TestFunctor<test::BinaryComparison, 12, 2>();
}
TEST(AutodiffCodeGen, LogicalOperators) {
  TestFunctor<test::LogicalOperators, 8, 3>();
}
TEST(AutodiffCodeGen, ScalarFunctions) {
  TestFunctor<test::ScalarFunctions, 20, 22>();
}
TEST(AutodiffCodeGen, LogicalFunctions) {
  TestFunctor<test::LogicalFunctions, 4, 4>();
}
TEST(AutodiffCodeGen, Branches) { TestFunctor<test::Branches, 4, 3>(); }

}  // namespace internal
}  // namespace ceres
