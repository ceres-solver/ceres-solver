// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/weighted_cost_function.h"

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/sized_cost_function.h"
#include "gtest/gtest.h"

namespace ceres {


class MockCostFunction : public SizedCostFunction<2, 3, 4> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    residuals[0] = 1.0;
    residuals[1] = 2.0;

    if (jacobians == NULL) {
      return true;
    }

    if (jacobians[0] != NULL) {
      MatrixRef jacobian(jacobians[0], 2, 3);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          jacobian(i, j) = i + j;
        }
      }
    }

    if (jacobians[1] != NULL) {
      MatrixRef jacobian(jacobians[1], 2, 4);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
          jacobian(i, j) = i + j;
        }
      }
    }

    return true;
  }
};

TEST(WeightedCostFunction, SameSizedResiduals) {
  MockCostFunction cost_function;
  double* parameters[2] = {NULL, NULL};

  Vector unweighted_residuals(2);
  Matrix unweighted_jacobian_block1(2, 3);
  Matrix unweighted_jacobian_block2(2, 4);
  double* unweighted_jacobians[2] = {unweighted_jacobian_block1.data(),
                                     unweighted_jacobian_block2.data()};
  EXPECT_TRUE(cost_function.Evaluate(parameters,
                                     unweighted_residuals.data(),
                                     unweighted_jacobians));
  Matrix weight_matrix(2, 2);
  weight_matrix.setRandom();

  Vector weighted_residuals(2);
  Matrix weighted_jacobian_block1(2, 3);
  Matrix weighted_jacobian_block2(2, 4);
  double* weighted_jacobians[2] = {weighted_jacobian_block1.data(),
                                   weighted_jacobian_block2.data()};

  WeightedCostFunction weighted_cost_function(weight_matrix.data(),
                                              weight_matrix.rows(),
                                              weight_matrix.cols(),
                                              new MockCostFunction);

  EXPECT_TRUE(weighted_cost_function.Evaluate(parameters,
                                              weighted_residuals.data(),
                                              weighted_jacobians));

  const double kTolerance = 1e-14;

  EXPECT_NEAR((weight_matrix * unweighted_residuals -
               weighted_residuals).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_residuals
      << "\n"
      << weighted_residuals;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block1 -
               weighted_jacobian_block1).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block1
      << "\n"
      << weighted_jacobian_block1;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block2 -
               weighted_jacobian_block2).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block2
      << "\n"
      << weighted_jacobian_block2;

  weighted_jacobians[0] = NULL;
  EXPECT_TRUE(weighted_cost_function.Evaluate(parameters,
                                              weighted_residuals.data(),
                                              weighted_jacobians));

  EXPECT_NEAR((weight_matrix * unweighted_residuals -
               weighted_residuals).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_residuals
      << "\n"
      << weighted_residuals;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block2 -
               weighted_jacobian_block2).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block2
      << "\n"
      << weighted_jacobian_block2;
}

TEST(WeightedCostFunction, DifferentSizedResiduals) {
  MockCostFunction cost_function;
  double* parameters[2] = {NULL, NULL};

  Vector unweighted_residuals(2);
  Matrix unweighted_jacobian_block1(2, 3);
  Matrix unweighted_jacobian_block2(2, 4);
  double* unweighted_jacobians[2] = {unweighted_jacobian_block1.data(),
                                     unweighted_jacobian_block2.data()};
  EXPECT_TRUE(cost_function.Evaluate(parameters,
                                     unweighted_residuals.data(),
                                     unweighted_jacobians));
  Matrix weight_matrix(1, 2);
  weight_matrix.setRandom();

  Vector weighted_residuals(1);
  Matrix weighted_jacobian_block1(1, 3);
  Matrix weighted_jacobian_block2(1, 4);
  double* weighted_jacobians[2] = {weighted_jacobian_block1.data(),
                                   weighted_jacobian_block2.data()};

  WeightedCostFunction weighted_cost_function(weight_matrix.data(),
                                              weight_matrix.rows(),
                                              weight_matrix.cols(),
                                              new MockCostFunction);

  EXPECT_TRUE(weighted_cost_function.Evaluate(parameters,
                                              weighted_residuals.data(),
                                              weighted_jacobians));

  const double kTolerance = 1e-14;

  EXPECT_NEAR((weight_matrix * unweighted_residuals -
               weighted_residuals).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_residuals
      << "\n"
      << weighted_residuals;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block1 -
               weighted_jacobian_block1).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block1
      << "\n"
      << weighted_jacobian_block1;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block2 -
               weighted_jacobian_block2).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block2
      << "\n"
      << weighted_jacobian_block2;

  weighted_jacobians[0] = NULL;
  EXPECT_TRUE(weighted_cost_function.Evaluate(parameters,
                                              weighted_residuals.data(),
                                              weighted_jacobians));

  EXPECT_NEAR((weight_matrix * unweighted_residuals -
               weighted_residuals).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_residuals
      << "\n"
      << weighted_residuals;

  EXPECT_NEAR((weight_matrix * unweighted_jacobian_block2 -
               weighted_jacobian_block2).norm(),
              0.0,
              kTolerance)
      << "\n"
      << weight_matrix * unweighted_jacobian_block2
      << "\n"
      << weighted_jacobian_block2;
}

}  // namespace ceres
