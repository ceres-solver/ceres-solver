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
// Author: thadh@gmail.com (Thad Hughes)
//         mierle@gmail.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/dynamic_autodiff_cost_function.h"

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
class MyCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* params0 = parameters[0];
    int r = 0;
    for (int i = 0; i < 10; ++i) {
      residuals[r++] = T(i) - params0[i];
      residuals[r++] = params0[i] - T(i);
    }

    T c_residual(0.0);
    for (int i = 0; i < 10; ++i) {
      c_residual += pow(params0[i], 2) - T(8) * params0[i];
    }

    const T* params1 = parameters[1];
    for (int i = 0; i < 5; ++i) {
      c_residual += params1[i];
    }
    residuals[r++] = c_residual;
    return true;
  }
};

TEST(DynamicAutodiffCostFunctionTest, TestResiduals) {
  vector<double> param_block_0(10, 0.0);
  vector<double> param_block_1(5, 0.0);
  DynamicAutoDiffCostFunction<MyCostFunctor, 3> cost_function(
      new MyCostFunctor());
  cost_function.AddParameterBlock(param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.size());
  cost_function.SetNumResiduals(21);

  // Test residual computation.
  vector<double> residuals(21, -100000);
  vector<double*> parameter_blocks(2);
  parameter_blocks[0] = &param_block_0[0];
  parameter_blocks[1] = &param_block_1[0];
  EXPECT_TRUE(cost_function.Evaluate(&parameter_blocks[0],
                                     residuals.data(),
                                     NULL));
  for (int r = 0; r < 10; ++r) {
    EXPECT_EQ(1.0 * r, residuals.at(r * 2));
    EXPECT_EQ(-1.0 * r, residuals.at(r * 2 + 1));
  }
  EXPECT_EQ(0, residuals.at(20));
}

TEST(DynamicAutodiffCostFunctionTest, TestJacobian) {
  // Test the residual counting.
  vector<double> param_block_0(10, 0.0);
  for (int i = 0; i < 10; ++i) {
    param_block_0[i] = 2 * i;
  }
  vector<double> param_block_1(5, 0.0);
  DynamicAutoDiffCostFunction<MyCostFunctor, 3> cost_function(
      new MyCostFunctor());
  cost_function.AddParameterBlock(param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.size());
  cost_function.SetNumResiduals(21);

  // Prepare the residuals.
  vector<double> residuals(21, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(2);
  parameter_blocks[0] = &param_block_0[0];
  parameter_blocks[1] = &param_block_1[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(2);
  jacobian_vect[0].resize(21 * 10, -100000);
  jacobian_vect[1].resize(21 * 5, -100000);
  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(jacobian_vect[1].data());

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

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

TEST(DynamicAutodiffCostFunctionTest, JacobianWithFirstParameterBlockConstant) {
  // Test the residual counting.
  vector<double> param_block_0(10, 0.0);
  for (int i = 0; i < 10; ++i) {
    param_block_0[i] = 2 * i;
  }
  vector<double> param_block_1(5, 0.0);
  DynamicAutoDiffCostFunction<MyCostFunctor, 3> cost_function(
      new MyCostFunctor());
  cost_function.AddParameterBlock(param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.size());
  cost_function.SetNumResiduals(21);

  // Prepare the residuals.
  vector<double> residuals(21, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(2);
  parameter_blocks[0] = &param_block_0[0];
  parameter_blocks[1] = &param_block_1[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(2);
  jacobian_vect[0].resize(21 * 10, -100000);
  jacobian_vect[1].resize(21 * 5, -100000);
  vector<double*> jacobian;
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[1].data());

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  for (int r = 0; r < 10; ++r) {
    EXPECT_EQ(-1.0 * r, residuals.at(r * 2));
    EXPECT_EQ(+1.0 * r, residuals.at(r * 2 + 1));
  }
  EXPECT_EQ(420, residuals.at(20));

  // Check "C" Jacobian for second parameter block.
  for (int p = 0; p < 5; ++p) {
    EXPECT_EQ(1.0, jacobian_vect[1][20 * 5 + p]);
    jacobian_vect[1][20 * 5 + p] = 0.0;
  }
  for (int i = 0; i < jacobian_vect[1].size(); ++i) {
    EXPECT_EQ(0.0, jacobian_vect[1][i]);
  }
}

TEST(DynamicAutodiffCostFunctionTest, JacobianWithSecondParameterBlockConstant) {
  // Test the residual counting.
  vector<double> param_block_0(10, 0.0);
  for (int i = 0; i < 10; ++i) {
    param_block_0[i] = 2 * i;
  }
  vector<double> param_block_1(5, 0.0);
  DynamicAutoDiffCostFunction<MyCostFunctor, 3> cost_function(
      new MyCostFunctor());
  cost_function.AddParameterBlock(param_block_0.size());
  cost_function.AddParameterBlock(param_block_1.size());
  cost_function.SetNumResiduals(21);

  // Prepare the residuals.
  vector<double> residuals(21, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(2);
  parameter_blocks[0] = &param_block_0[0];
  parameter_blocks[1] = &param_block_1[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(2);
  jacobian_vect[0].resize(21 * 10, -100000);
  jacobian_vect[1].resize(21 * 5, -100000);
  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(NULL);

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

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
}

// Takes 3 parameter blocks:
//     parameters[0] (x) is size 1.
//     parameters[1] (y) is size 2.
//     parameters[2] (z) is size 3.
// Emits 7 residuals:
//     A: x[0] (= sum_x)
//     B: y[0] + 2.0 * y[1] (= sum_y)
//     C: z[0] + 3.0 * z[1] + 6.0 * z[2] (= sum_z)
//     D: sum_x * sum_y
//     E: sum_y * sum_z
//     F: sum_x * sum_z
//     G: sum_x * sum_y * sum_z
class MyThreeParameterCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* x = parameters[0];
    const T* y = parameters[1];
    const T* z = parameters[2];

    T sum_x = x[0];
    T sum_y = y[0] + 2.0 * y[1];
    T sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];

    residuals[0] = sum_x;
    residuals[1] = sum_y;
    residuals[2] = sum_z;
    residuals[3] = sum_x * sum_y;
    residuals[4] = sum_y * sum_z;
    residuals[5] = sum_x * sum_z;
    residuals[6] = sum_x * sum_y * sum_z;
    return true;
  }
};

TEST(DynamicAutodiffCostFunctionTest, TestThreeParameterResiduals) {
  vector<double> x(1, 0.0);
  vector<double> y(2);
  y[0] = 1.0;
  y[1] = 3.0;
  vector<double> z(3);
  z[0] = 2.0;
  z[1] = 4.0;
  z[2] = 6.0;

  DynamicAutoDiffCostFunction<MyThreeParameterCostFunctor, 3> cost_function(
    new MyThreeParameterCostFunctor());
  cost_function.AddParameterBlock(x.size());
  cost_function.AddParameterBlock(y.size());
  cost_function.AddParameterBlock(z.size());
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(3);
  parameter_blocks[0] = &x[0];
  parameter_blocks[1] = &y[0];
  parameter_blocks[2] = &z[0];

  // Test residual computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     NULL));

  const double sum_x = x[0];
  const double sum_y = y[0] + 2.0 * y[1];
  const double sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];
  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);
}

TEST(DynamicAutodiffCostFunctionTest, TestThreeParameterJacobians) {
  vector<double> x(1, 0.0);
  vector<double> y(2);
  y[0] = 1.0;
  y[1] = 3.0;
  vector<double> z(3);
  z[0] = 2.0;
  z[1] = 4.0;
  z[2] = 6.0;

  DynamicAutoDiffCostFunction<MyThreeParameterCostFunctor, 3> cost_function(
    new MyThreeParameterCostFunctor());
  cost_function.AddParameterBlock(x.size());
  cost_function.AddParameterBlock(y.size());
  cost_function.AddParameterBlock(z.size());
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(3);
  parameter_blocks[0] = &x[0];
  parameter_blocks[1] = &y[0];
  parameter_blocks[2] = &z[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(3);
  jacobian_vect[0].resize(7 * x.size(), -100000);
  jacobian_vect[1].resize(7 * y.size(), -100000);
  jacobian_vect[2].resize(7 * z.size(), -100000);

  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(jacobian_vect[1].data());
  jacobian.push_back(jacobian_vect[2].data());

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x[0];
  const double sum_y = y[0] + 2.0 * y[1];
  const double sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];

  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);

  const double expected_jacobian_x[7] = {
    1.0,
    0.0,
    0.0,
    sum_y,
    0.0,
    sum_z,
    sum_y * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_x[i], jacobian_vect[0][i]);
  }

  const double expected_jacobian_y[14] = {
    0.0, 0.0,
    1.0, 2.0,
    0.0, 0.0,
    sum_x, 2.0 * sum_x,
    sum_z, 2.0 * sum_z,
    0.0, 0.0,
    sum_x * sum_z, 2.0 * sum_x * sum_z
  };
  for (int i = 0; i < 14; ++i) {
    EXPECT_EQ(expected_jacobian_y[i], jacobian_vect[1][i]);
  }

  const double expected_jacobian_z[21] = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 3.0, 6.0,
    0.0, 0.0, 0.0,
    sum_y, 3.0 * sum_y, 6.0 * sum_y,
    sum_x, 3.0 * sum_x, 6.0 * sum_x,
    sum_x * sum_y, 3.0 * sum_x * sum_y, 6.0 * sum_x * sum_y
  };
  for (int i = 0; i < 21; ++i) {
    EXPECT_EQ(expected_jacobian_z[i], jacobian_vect[2][i]);
  }
}

TEST(DynamicAutodiffCostFunctionTest,
     ThreeParameterJacobianWithFirstAndLastParameterBlockConstant) {
  vector<double> x(1, 0.0);
  vector<double> y(2);
  y[0] = 1.0;
  y[1] = 3.0;
  vector<double> z(3);
  z[0] = 2.0;
  z[1] = 4.0;
  z[2] = 6.0;

  DynamicAutoDiffCostFunction<MyThreeParameterCostFunctor, 3> cost_function(
    new MyThreeParameterCostFunctor());
  cost_function.AddParameterBlock(x.size());
  cost_function.AddParameterBlock(y.size());
  cost_function.AddParameterBlock(z.size());
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(3);
  parameter_blocks[0] = &x[0];
  parameter_blocks[1] = &y[0];
  parameter_blocks[2] = &z[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(3);
  jacobian_vect[0].resize(7 * x.size(), -100000);
  jacobian_vect[1].resize(7 * y.size(), -100000);
  jacobian_vect[2].resize(7 * z.size(), -100000);

  vector<double*> jacobian;
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[1].data());
  jacobian.push_back(NULL);

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x[0];
  const double sum_y = y[0] + 2.0 * y[1];
  const double sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];

  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);

  const double expected_jacobian_y[14] = {
    0.0, 0.0,
    1.0, 2.0,
    0.0, 0.0,
    sum_x, 2.0 * sum_x,
    sum_z, 2.0 * sum_z,
    0.0, 0.0,
    sum_x * sum_z, 2.0 * sum_x * sum_z
  };
  for (int i = 0; i < 14; ++i) {
    EXPECT_EQ(expected_jacobian_y[i], jacobian_vect[1][i]);
  }
}

TEST(DynamicAutodiffCostFunctionTest,
     ThreeParameterJacobianWithSecondParameterBlockConstant) {
  vector<double> x(1, 0.0);
  vector<double> y(2);
  y[0] = 1.0;
  y[1] = 3.0;
  vector<double> z(3);
  z[0] = 2.0;
  z[1] = 4.0;
  z[2] = 6.0;

  DynamicAutoDiffCostFunction<MyThreeParameterCostFunctor, 3> cost_function(
    new MyThreeParameterCostFunctor());
  cost_function.AddParameterBlock(x.size());
  cost_function.AddParameterBlock(y.size());
  cost_function.AddParameterBlock(z.size());
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(3);
  parameter_blocks[0] = &x[0];
  parameter_blocks[1] = &y[0];
  parameter_blocks[2] = &z[0];

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(3);
  jacobian_vect[0].resize(7 * x.size(), -100000);
  jacobian_vect[1].resize(7 * y.size(), -100000);
  jacobian_vect[2].resize(7 * z.size(), -100000);

  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[2].data());

  // Test jacobian computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x[0];
  const double sum_y = y[0] + 2.0 * y[1];
  const double sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];

  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);

  const double expected_jacobian_x[7] = {
    1.0,
    0.0,
    0.0,
    sum_y,
    0.0,
    sum_z,
    sum_y * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_x[i], jacobian_vect[0][i]);
  }

  const double expected_jacobian_z[21] = {
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 3.0, 6.0,
    0.0, 0.0, 0.0,
    sum_y, 3.0 * sum_y, 6.0 * sum_y,
    sum_x, 3.0 * sum_x, 6.0 * sum_x,
    sum_x * sum_y, 3.0 * sum_x * sum_y, 6.0 * sum_x * sum_y
  };
  for (int i = 0; i < 21; ++i) {
    EXPECT_EQ(expected_jacobian_z[i], jacobian_vect[2][i]);
  }
}

// Takes 6 parameter blocks all of size 1:
//     x0, y0, y1, z0, z1, z2
// Same 7 residuals as MyThreeParameterCostFunctor.
// Naming convention for tests is (V)ariable and (C)onstant.
class MySixParameterCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* x0 = parameters[0];
    const T* y0 = parameters[1];
    const T* y1 = parameters[2];
    const T* z0 = parameters[3];
    const T* z1 = parameters[4];
    const T* z2 = parameters[5];

    T sum_x = x0[0];
    T sum_y = y0[0] + 2.0 * y1[0];
    T sum_z = z0[0] + 3.0 * z1[0] + 6.0 * z2[0];

    residuals[0] = sum_x;
    residuals[1] = sum_y;
    residuals[2] = sum_z;
    residuals[3] = sum_x * sum_y;
    residuals[4] = sum_y * sum_z;
    residuals[5] = sum_x * sum_z;
    residuals[6] = sum_x * sum_y * sum_z;
    return true;
  }
};

TEST(DynamicAutodiffCostFunctionTest, TestSixParameterResiduals) {
  double x0 = 0.0, y0 = 1.0, y1 = 3.0, z0 = 2.0, z1 = 4.0, z2 = 6.0;

  DynamicAutoDiffCostFunction<MySixParameterCostFunctor, 3> cost_function(
    new MySixParameterCostFunctor());
  for (int i = 0; i < 6; ++i) {
    cost_function.AddParameterBlock(1);
  }
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(6);
  parameter_blocks[0] = &x0;
  parameter_blocks[1] = &y0;
  parameter_blocks[2] = &y1;
  parameter_blocks[3] = &z0;
  parameter_blocks[4] = &z1;
  parameter_blocks[5] = &z2;

  // Test residual computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     NULL));

  const double sum_x = x0;
  const double sum_y = y0 + 2.0 * y1;
  const double sum_z = z0 + 3.0 * z1 + 6.0 * z2;
  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);
}

TEST(DynamicAutodiffCostFunctionTest, TestSixParameterJacobian) {
  double x0 = 0.0, y0 = 1.0, y1 = 3.0, z0 = 2.0, z1 = 4.0, z2 = 6.0;

  DynamicAutoDiffCostFunction<MySixParameterCostFunctor, 3> cost_function(
    new MySixParameterCostFunctor());
  for (int i = 0; i < 6; ++i) {
    cost_function.AddParameterBlock(1);
  }
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(6);
  parameter_blocks[0] = &x0;
  parameter_blocks[1] = &y0;
  parameter_blocks[2] = &y1;
  parameter_blocks[3] = &z0;
  parameter_blocks[4] = &z1;
  parameter_blocks[5] = &z2;

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(6);
  for (int i = 0; i < 6; ++i) {
    jacobian_vect[i].resize(7, -100000);
  };

  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(jacobian_vect[1].data());
  jacobian.push_back(jacobian_vect[2].data());
  jacobian.push_back(jacobian_vect[3].data());
  jacobian.push_back(jacobian_vect[4].data());
  jacobian.push_back(jacobian_vect[5].data());

  // Test residual computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x0;
  const double sum_y = y0 + 2.0 * y1;
  const double sum_z = z0 + 3.0 * z1 + 6.0 * z2;
  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);
 
  // Test jacobian computation.
  const double expected_jacobian_x0[7] = {
    1.0,
    0.0,
    0.0,
    sum_y,
    0.0,
    sum_z,
    sum_y * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_x0[i], jacobian_vect[0][i]);
  }

  const double expected_jacobian_y0[7] = {
    0.0,
    1.0,
    0.0,
    sum_x,
    sum_z,
    0.0,
    sum_x * sum_z,
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_y0[i], jacobian_vect[1][i]);
  }

  const double expected_jacobian_y1[7] = {
    0.0,
    2.0,
    0.0,
    2.0 * sum_x,
    2.0 * sum_z,
    0.0,
    2.0 * sum_x * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_y1[i], jacobian_vect[2][i]);
  }

  const double expected_jacobian_z0[7] = {
    0.0,
    0.0,
    1.0,
    0.0,
    sum_y,
    sum_x,
    sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z0[i], jacobian_vect[3][i]);
  }

  const double expected_jacobian_z1[7] = {
    0.0,
    0.0,
    3.0,
    0.0,
    3.0 * sum_y,
    3.0 * sum_x,
    3.0 * sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z1[i], jacobian_vect[4][i]);
  }

  const double expected_jacobian_z2[7] = {
    0.0,
    0.0,
    6.0,
    0.0,
    6.0 * sum_y,
    6.0 * sum_x,
    6.0 * sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z2[i], jacobian_vect[5][i]);
  }
}

TEST(DynamicAutodiffCostFunctionTest, TestSixParameterJacobianVVCVVC) {
  double x0 = 0.0, y0 = 1.0, y1 = 3.0, z0 = 2.0, z1 = 4.0, z2 = 6.0;

  DynamicAutoDiffCostFunction<MySixParameterCostFunctor, 3> cost_function(
    new MySixParameterCostFunctor());
  for (int i = 0; i < 6; ++i) {
    cost_function.AddParameterBlock(1);
  }
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(6);
  parameter_blocks[0] = &x0;
  parameter_blocks[1] = &y0;
  parameter_blocks[2] = &y1;
  parameter_blocks[3] = &z0;
  parameter_blocks[4] = &z1;
  parameter_blocks[5] = &z2;

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(6);
  for (int i = 0; i < 6; ++i) {
    jacobian_vect[i].resize(7, -100000);
  };

  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(jacobian_vect[1].data());
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[3].data());
  jacobian.push_back(jacobian_vect[4].data());
  jacobian.push_back(NULL);

  // Test residual computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x0;
  const double sum_y = y0 + 2.0 * y1;
  const double sum_z = z0 + 3.0 * z1 + 6.0 * z2;
  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);
 
  // Test jacobian computation.
  const double expected_jacobian_x0[7] = {
    1.0,
    0.0,
    0.0,
    sum_y,
    0.0,
    sum_z,
    sum_y * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_x0[i], jacobian_vect[0][i]);
  }

  const double expected_jacobian_y0[7] = {
    0.0,
    1.0,
    0.0,
    sum_x,
    sum_z,
    0.0,
    sum_x * sum_z,
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_y0[i], jacobian_vect[1][i]);
  }

  const double expected_jacobian_z0[7] = {
    0.0,
    0.0,
    1.0,
    0.0,
    sum_y,
    sum_x,
    sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z0[i], jacobian_vect[3][i]);
  }

  const double expected_jacobian_z1[7] = {
    0.0,
    0.0,
    3.0,
    0.0,
    3.0 * sum_y,
    3.0 * sum_x,
    3.0 * sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z1[i], jacobian_vect[4][i]);
  }
}

TEST(DynamicAutodiffCostFunctionTest, TestSixParameterJacobianVCCVCV) {
  double x0 = 0.0, y0 = 1.0, y1 = 3.0, z0 = 2.0, z1 = 4.0, z2 = 6.0;

  DynamicAutoDiffCostFunction<MySixParameterCostFunctor, 3> cost_function(
    new MySixParameterCostFunctor());
  for (int i = 0; i < 6; ++i) {
    cost_function.AddParameterBlock(1);
  }
  cost_function.SetNumResiduals(7);

  // Prepare the residuals.
  vector<double> residuals(7, -100000);

  // Prepare the parameters.
  vector<double*> parameter_blocks(6);
  parameter_blocks[0] = &x0;
  parameter_blocks[1] = &y0;
  parameter_blocks[2] = &y1;
  parameter_blocks[3] = &z0;
  parameter_blocks[4] = &z1;
  parameter_blocks[5] = &z2;

  // Prepare the jacobian.
  vector<vector<double> > jacobian_vect(6);
  for (int i = 0; i < 6; ++i) {
    jacobian_vect[i].resize(7, -100000);
  };

  vector<double*> jacobian;
  jacobian.push_back(jacobian_vect[0].data());
  jacobian.push_back(NULL);
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[3].data());
  jacobian.push_back(NULL);
  jacobian.push_back(jacobian_vect[5].data());

  // Test residual computation.
  EXPECT_TRUE(cost_function.Evaluate(parameter_blocks.data(),
                                     residuals.data(),
                                     jacobian.data()));

  const double sum_x = x0;
  const double sum_y = y0 + 2.0 * y1;
  const double sum_z = z0 + 3.0 * z1 + 6.0 * z2;
  EXPECT_EQ(residuals[0], sum_x);
  EXPECT_EQ(residuals[1], sum_y);
  EXPECT_EQ(residuals[2], sum_z);
  EXPECT_EQ(residuals[3], sum_x * sum_y);
  EXPECT_EQ(residuals[4], sum_y * sum_z);
  EXPECT_EQ(residuals[5], sum_x * sum_z);
  EXPECT_EQ(residuals[6], sum_x * sum_y * sum_z);
 
  // Test jacobian computation.
  const double expected_jacobian_x0[7] = {
    1.0,
    0.0,
    0.0,
    sum_y,
    0.0,
    sum_z,
    sum_y * sum_z
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_x0[i], jacobian_vect[0][i]);
  }

  const double expected_jacobian_z0[7] = {
    0.0,
    0.0,
    1.0,
    0.0,
    sum_y,
    sum_x,
    sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z0[i], jacobian_vect[3][i]);
  }

  const double expected_jacobian_z2[7] = {
    0.0,
    0.0,
    6.0,
    0.0,
    6.0 * sum_y,
    6.0 * sum_x,
    6.0 * sum_x * sum_y
  };
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(expected_jacobian_z2[i], jacobian_vect[5][i]);
  }
}

}  // namespace internal
}  // namespace ceres
