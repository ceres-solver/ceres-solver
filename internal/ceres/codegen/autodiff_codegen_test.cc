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
#include "codegen/test_utils.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class CompareAutodiffToCodegen : public testing::TestWithParam<double> {
 public:
  template <typename FunctorType, int kNumResiduals, int... Ns>
  void TestCostFunction() {
    FunctorType cost_function_generated;
    CostFunctionToFunctor<FunctorType> cost_functor;
    auto* cost_function_ad =
        new AutoDiffCostFunction<CostFunctionToFunctor<FunctorType>,
                                 kNumResiduals,
                                 Ns...>(&cost_functor);
    auto value = GetParam();
    CompareCostFunctions(
        &cost_function_generated, cost_function_ad, value, 1e-20);
  }
};

TEST_P(CompareAutodiffToCodegen, InputOutputAssignment) {
  TestCostFunction<test::InputOutputAssignment, 7, 4, 2, 1>();
}

TEST_P(CompareAutodiffToCodegen, CompileTimeConstants) {
  TestCostFunction<test::CompileTimeConstants, 7, 1>();
}

TEST_P(CompareAutodiffToCodegen, Assignments) {
  TestCostFunction<test::Assignments, 8, 2>();
}

TEST_P(CompareAutodiffToCodegen, BinaryArithmetic) {
  TestCostFunction<test::BinaryArithmetic, 9, 2>();
}

TEST_P(CompareAutodiffToCodegen, UnaryArithmetic) {
  TestCostFunction<test::UnaryArithmetic, 3, 1>();
}

TEST_P(CompareAutodiffToCodegen, BinaryComparison) {
  TestCostFunction<test::BinaryComparison, 12, 2>();
}

TEST_P(CompareAutodiffToCodegen, LogicalOperators) {
  TestCostFunction<test::LogicalOperators, 8, 3>();
}

TEST_P(CompareAutodiffToCodegen, ScalarFunctions) {
  TestCostFunction<test::ScalarFunctions, 20, 22>();
}

TEST_P(CompareAutodiffToCodegen, LogicalFunctions) {
  TestCostFunction<test::LogicalFunctions, 4, 4>();
}

TEST_P(CompareAutodiffToCodegen, Branches) {
  TestCostFunction<test::Branches, 4, 3>();
}

INSTANTIATE_TEST_SUITE_P(
    AutoDiffCodeGenTest,
    CompareAutodiffToCodegen,
    testing::Values(0,
                    -1,
                    1,
                    0.5,
                    -0.5,
                    10,
                    -10,
                    1e20,
                    1e-20,
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()));
}  // namespace internal
}  // namespace ceres
