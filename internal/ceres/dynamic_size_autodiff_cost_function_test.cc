// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
// Author: thadh@google.com (Thad Hughes)

#include "ceres/dynamic_size_autodiff_cost_function.h"

#include <cstddef>

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Takes 2 parameter blocks:
//     parameters[0] is size 10.
//     parameters[1] is size 5.
// Emits 21 residuals:
//     A: i - parameters[0][i], for i in [0,10)  -- this is 10 residuals
//     B: parameters[0][i] - i, for i in [0,10)  -- this is another 10.
//     C: sum(parameters[0][i]^2 - 8*parameters[0][i]) + sum(parameters[1][i])
class MyCostFunction {
 public:
  template <typename T, typename R>
  bool operator()(T const* const* parameters, R* residual_emitter) const {
    const T* params0 = parameters[0];
    for (int i = 0; i < 10; ++i) {
      // Emit "A" residuals.
      residual_emitter->EmitResidual(T(i) - params0[i]);
      // Emit "B" residuals.
      residual_emitter->EmitResidual(params0[i] - T(i));
    }

    // Emit "C" residual.
    T c_residual(0.0);
    for (int i = 0; i < 10; ++i) {
      c_residual += pow(params0[i], 2) - T(8) * params0[i];
    }
    // And another parameter block, just to test.
    const T* params1 = parameters[1];
    for (int i = 0; i < 5; ++i) {
      c_residual += params1[i];
    }
    residual_emitter->EmitResidual(c_residual);
    return true;
  }
};

TEST(DynamicSizeAutodiffCostFunctionTest, TestResiduals) {
  // Test the residual counting.
  vector<double> param_block_0(10, 0.0);
  vector<double> param_block_1(5, 0.0);
  typedef Jet<double, 3> MyJet;
  DynamicSizeAutodiffCostFunction<MyCostFunction, MyJet> cost_function(
      new MyCostFunction());
  cost_function.AddParameterBlock(param_block_0.data(), param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.data(), param_block_1.size());
  cost_function.CountResiduals();
  EXPECT_EQ(21, cost_function.NumResiduals());

  // Test residual computation.
  vector<double> residuals(21, -100000);
  EXPECT_TRUE(cost_function.Evaluate(
      cost_function.parameters(), residuals.data(), NULL));
  for (int r = 0; r < 10; ++r) {
    EXPECT_EQ(1.0 * r, residuals.at(r * 2));
    EXPECT_EQ(-1.0 * r, residuals.at(r * 2 + 1));
  }
  EXPECT_EQ(0, residuals.at(20));
}

TEST(DynamicSizeAutodiffCostFunctionTest, TestJacobian) {
  // Test the residual counting.
  vector<double> param_block_0(10, 0.0);
  for (int i = 0; i < 10; ++i) {
    param_block_0[i] = 2 * i;
  }
  vector<double> param_block_1(5, 0.0);
  typedef Jet<double,3> MyJet;
  DynamicSizeAutodiffCostFunction<MyCostFunction, MyJet> cost_function(
      new MyCostFunction());
  cost_function.AddParameterBlock(param_block_0.data(), param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.data(), param_block_1.size());
  cost_function.CountResiduals();
  EXPECT_EQ(21, cost_function.NumResiduals());

  // Test jacobian computation.
  vector<double> residuals(21, -100000);
  vector<vector<double> > jacobian_vect(2);
  jacobian_vect[0].resize(21 * 10, -100000);
  jacobian_vect[1].resize(21 * 5, -100000);
  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(jacobian_vect[1].data());
  EXPECT_TRUE(cost_function.Evaluate(
      cost_function.parameters(), residuals.data(), jacobian.data()));

  for (int r = 0; r < 10; ++r) {
    EXPECT_EQ(-1.0 * r, residuals.at(r * 2));
    EXPECT_EQ(+1.0 * r, residuals.at(r * 2 + 1));
  }
  EXPECT_EQ(420, residuals.at(20));
  for (int p = 0; p < 10; ++p) {
    // Check "A" Jacobian.
    EXPECT_EQ(-1.0, jacobian_vect[0][2*p * 10 + p]);
    // Check "B" Jacobian.
    EXPECT_EQ(+1.0, jacobian_vect[0][(2*p+1) * 10 + p]);
    jacobian_vect[0][2*p * 10 + p] = 0.0;
    jacobian_vect[0][(2*p+1) * 10 + p] = 0.0;
  }

  // Check "C" Jacobian for first parameter block.
  for (int p = 0; p < 10; ++p) {
    EXPECT_EQ(4 * p - 8, jacobian_vect[0][20 * 10 + p]);
    jacobian_vect[0][20 * 10 + p] = 0.0;
  }
  for (int i = 0; i < jacobian_vect[0].size(); ++i) {
    EXPECT_EQ(0.0, jacobian_vect[0][i]);
  }

  // Check "C" Jacobian for second parameter block.
  for (int p = 0; p < 5; ++p) {
    EXPECT_EQ(1.0, jacobian_vect[1][20 * 5 + p]);
    jacobian_vect[1][20 * 5 + p] = 0.0;
  }
  for (int i = 0; i < jacobian_vect[1].size(); ++i) {
    EXPECT_EQ(0.0, jacobian_vect[1][i]);
  }
}

}  // namespace internal
}  // namespace ceres
