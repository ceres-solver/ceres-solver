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
namespace ceres {
namespace internal {

// This struct is used to convert combined cost functions (evaluate method +
// operator()) to cost functors (only operator()). As a result, putting such a
// type into ceres::Autodiff will use the operator() instead of evaluate().
template <typename CostFunction>
struct CostFunctionToFunctor {
  template <typename... _Args>
  CostFunctionToFunctor(_Args&&... __args)
      : data(std::forward<_Args>(__args)...) {}

  template <typename... _Args>
  bool operator()(_Args&&... __args) const {
    return data(std::forward<_Args>(__args)...);
  }

  CostFunction data;
};

inline void double_compare(double v1, double v2) {
  // If both are inf or nan it's fine too!
  if (std::isinf(v1) && std::isinf(v2)) {
    return;
  }

  if (std::isnan(v1) && std::isnan(v2)) {
    return;
  }

  EXPECT_NEAR(v1, v2, 1e-40);
}

template <int kNumResiduals, int... Ns>
void compare_cost_functions(CostFunction* f1,
                            CostFunction* f2,
                            bool random_values = true,
                            double value = 0) {
  using Params = StaticParameterDims<Ns...>;

  std::array<double, Params::kNumParameters> params_array;
  std::array<double*, Params::kNumParameters> params;

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

  std::array<double, kNumResiduals> residuals_0, residuals_1;
  std::fill(residuals_0.begin(), residuals_0.end(), 0.0);
  std::fill(residuals_1.begin(), residuals_1.end(), 0.0);
  std::array<double, kNumResiduals * Params::kNumParameters> jacobians_array_0,
      jacobians_array_1;
  std::fill(jacobians_array_0.begin(), jacobians_array_0.end(), 0.0);
  std::fill(jacobians_array_1.begin(), jacobians_array_1.end(), 0.0);

  std::array<double*, Params::kNumParameterBlocks> jacobians_0, jacobians_1;

  for (int i = 0, k = 0; i < Params::kNumParameterBlocks;
       k += Params::GetDim(i), ++i) {
    jacobians_0[i] = &jacobians_array_0[k * kNumResiduals];
    jacobians_1[i] = &jacobians_array_1[k * kNumResiduals];
  }

  f1->Evaluate(params.data(), residuals_0.data(), jacobians_0.data());
  f2->Evaluate(params.data(), residuals_1.data(), jacobians_1.data());

  for (int i = 0; i < kNumResiduals; ++i) {
    double_compare(residuals_0[i], residuals_1[i]);
  }
  for (int i = 0; i < kNumResiduals * Params::kNumParameters; ++i) {
    double_compare(jacobians_array_0[i], jacobians_array_1[i]);
  }
}

template <typename FunctorType, int kNumResiduals, int... Ns>
void test_functor() {
  FunctorType cost_function_generated;
  CostFunctionToFunctor<FunctorType> cost_functor;
  auto* cost_function_ad =
      new ceres::AutoDiffCostFunction<CostFunctionToFunctor<FunctorType>,
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
    ceres::internal::compare_cost_functions<kNumResiduals, Ns...>(
        &cost_function_generated, cost_function_ad, false, v);
  }

  // Run N times with random values in the range [-1,1]
  for (int i = 0; i < 100; ++i) {
    ceres::internal::compare_cost_functions<kNumResiduals, Ns...>(
        &cost_function_generated, cost_function_ad, true);
  }
}

TEST(AutodiffCodeGen, InputOutputAssignment) {
  test_functor<test::InputOutputAssignment, 7, 4, 2, 1>();
}

TEST(AutodiffCodeGen, CompileTimeConstants) {
  test_functor<test::CompileTimeConstants, 7, 1>();
}

TEST(AutodiffCodeGen, Assignments) { test_functor<test::Assignments, 8, 2>(); }
TEST(AutodiffCodeGen, BinaryArithmetic) {
  test_functor<test::BinaryArithmetic, 9, 2>();
}
TEST(AutodiffCodeGen, UnaryArithmetic) {
  test_functor<test::UnaryArithmetic, 3, 1>();
}
TEST(AutodiffCodeGen, BinaryComparison) {
  test_functor<test::BinaryComparison, 12, 2>();
}
TEST(AutodiffCodeGen, LogicalOperators) {
  test_functor<test::LogicalOperators, 8, 3>();
}
TEST(AutodiffCodeGen, ScalarFunctions) {
  test_functor<test::ScalarFunctions, 20, 22>();
}
TEST(AutodiffCodeGen, LogicalFunctions) {
  test_functor<test::LogicalFunctions, 4, 4>();
}
TEST(AutodiffCodeGen, Branches) { test_functor<test::Branches, 4, 3>(); }

}  // namespace internal
}  // namespace ceres
