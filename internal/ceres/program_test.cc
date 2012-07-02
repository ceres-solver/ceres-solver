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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include <glog/logging.h>
#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/problem_impl.h"
#include "gtest/gtest.h"
#include "ceres/solver_impl.h"

namespace ceres {
namespace internal {

// Simple cost function used for testing Program::Evaluate.
template <int kNumResiduals, int kNumParameterBlocks >
class QuadraticCostFunction : public CostFunction {
 public:
  QuadraticCostFunction() {
    CHECK_GT(kNumResiduals, 0);
    CHECK_GT(kNumParameterBlocks, 0);
    set_num_residuals(kNumResiduals);
    for (int i = 0; i < kNumParameterBlocks; ++i) {
      mutable_parameter_block_sizes()->push_back(kNumResiduals);
    }
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < kNumResiduals; ++i) {
      residuals[i] = i;
      for (int j = 0; j < kNumParameterBlocks; ++j) {
        residuals[i] -= (j + 1.0) * parameters[j][i] * parameters[j][i];
      }
    }

    if (jacobians == NULL) {
      return true;
    }

    for (int j = 0; j < kNumParameterBlocks; ++j) {
      if (jacobians[j] != NULL) {
        MatrixRef(jacobians[j], kNumResiduals, kNumResiduals) =
            (-2.0 * (j + 1.0) * ConstVectorRef(parameters[j], kNumResiduals)).asDiagonal();
      }
    }

    return true;
  }
};

// Convert a CRSMatrix to a dense Eigen matrix.
void CRSToDenseMatrix(const CRSMatrix& input, Matrix* output) {
  Matrix& m = *CHECK_NOTNULL(output);
  m.resize(input.num_rows, input.num_cols);
  m.setZero();
  for (int row = 0; row < input.num_rows; ++row) {
    for (int j = input.rows[row]; j < input.rows[row + 1]; ++j) {
      const int col = input.cols[j];
      m(row, col) = input.values[j];
    }
  }
}

void EvaluateResidualsAndJacobians(double const * const parameter_block1,
                                   double const * const parameter_block2,
                                   const CostFunction* cost_function,
                                   const int num_residuals,
                                   const int parameter_block_size,
                                   const int row,
                                   const int col1,
                                   const int col2,
                                   Vector* residuals,
                                   Matrix* jacobian) {
  double const* parameters[2];
  parameters[0] = parameter_block1;
  parameters[1] = parameter_block2;
  double* jacobian_buffer[2];
  jacobian_buffer[0] = new double[num_residuals * parameter_block_size];
  jacobian_buffer[1] = new double[num_residuals * parameter_block_size];
  cost_function->Evaluate(parameters, residuals->data() + row, jacobian_buffer);

  jacobian->block(row, col1, num_residuals, parameter_block_size) =
      MatrixRef(jacobian_buffer[0], num_residuals, parameter_block_size);
  jacobian->block(row, col2, num_residuals, parameter_block_size) =
      MatrixRef(jacobian_buffer[1], num_residuals, parameter_block_size);

  delete []jacobian_buffer[0];
  delete []jacobian_buffer[1];
}

#define COMPARE_ACTUAL_AND_EXPECTED(object) \
  EXPECT_NEAR((actual_ ## object - expected_ ## object).norm(), 0.0, 1e-14);

TEST(Program, EvaluateMultipleResidualBlocks) {
  ProblemImpl problem;
  double parameters[6];
  for (int i = 0; i < 6; ++i) {
    parameters[i] = static_cast<double>(i);
  }

  CostFunction* cost_function = new QuadraticCostFunction<2, 2>;
  problem.AddResidualBlock(cost_function, NULL, parameters, parameters + 2);
  problem.AddResidualBlock(cost_function, NULL, parameters + 2, parameters + 4);
  problem.AddResidualBlock(cost_function, NULL, parameters + 4, parameters);

  Vector expected_initial_residuals(6);
  expected_initial_residuals.setZero();
  Matrix expected_initial_jacobian(6,6);
  expected_initial_jacobian.setZero();

  EvaluateResidualsAndJacobians(parameters,
                                parameters + 2,
                                cost_function,
                                2, 2, 0, 0, 2,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  EvaluateResidualsAndJacobians(parameters + 2,
                                parameters + 4,
                                cost_function,
                                2, 2, 2, 2, 4,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  EvaluateResidualsAndJacobians(parameters + 4,
                                parameters,
                                cost_function,
                                2, 2, 4, 4, 0,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  Solver::Options options;
  Solver::Summary summary;
  options.return_initial_residuals = true;
  options.return_final_residuals = true;
  options.return_initial_jacobian = true;
  options.return_final_jacobian = true;
  SolverImpl::Solve(options, &problem,  &summary);

  Vector expected_final_residuals(6);
  expected_final_residuals.setZero();
  Matrix expected_final_jacobian(6,6);
  expected_final_jacobian.setZero();

  EvaluateResidualsAndJacobians(parameters,
                                parameters + 2,
                                cost_function,
                                2, 2, 0, 0, 2,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EvaluateResidualsAndJacobians(parameters + 2,
                                parameters + 4,
                                cost_function,
                                2, 2, 2, 2, 4,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EvaluateResidualsAndJacobians(parameters + 4,
                                parameters,
                                cost_function,
                                2, 2, 4, 4, 0,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EXPECT_EQ(summary.initial_residuals.size(), 6);
  VectorRef actual_initial_residuals(&summary.initial_residuals[0], 6);
  COMPARE_ACTUAL_AND_EXPECTED(initial_residuals);

  EXPECT_EQ(summary.initial_jacobian.num_rows, 6);
  EXPECT_EQ(summary.initial_jacobian.num_cols, 6);
  EXPECT_EQ(summary.initial_jacobian.values.size(), 24);
  Matrix actual_initial_jacobian(6,6);
  CRSToDenseMatrix(summary.initial_jacobian, &actual_initial_jacobian);
  COMPARE_ACTUAL_AND_EXPECTED(initial_jacobian);

  EXPECT_EQ(summary.final_residuals.size(), 6);
  VectorRef actual_final_residuals(&summary.final_residuals[0], 6);
  COMPARE_ACTUAL_AND_EXPECTED(initial_residuals);

  EXPECT_EQ(summary.final_jacobian.num_rows, 6);
  EXPECT_EQ(summary.final_jacobian.num_cols, 6);
  EXPECT_EQ(summary.final_jacobian.values.size(), 24);
  Matrix actual_final_jacobian(6,6);
  CRSToDenseMatrix(summary.final_jacobian, &actual_final_jacobian);
  COMPARE_ACTUAL_AND_EXPECTED(final_jacobian);
}

TEST(Program, EvaluateMultipleResidualBlocksWithConstantParameterBlocks) {
  ProblemImpl problem;
  double parameters[6];
  for (int i = 0; i < 6; ++i) {
    parameters[i] = static_cast<double>(i);
  }

  CostFunction* cost_function = new QuadraticCostFunction<2, 2>;
  problem.AddResidualBlock(cost_function, NULL, parameters, parameters + 2);
  problem.AddResidualBlock(cost_function, NULL, parameters + 2, parameters + 4);
  problem.AddResidualBlock(cost_function, NULL, parameters + 4, parameters);

  // Set the second parameter block constant.
  problem.SetParameterBlockConstant(parameters + 2);

  Vector expected_initial_residuals(6);
  expected_initial_residuals.setZero();
  Matrix expected_initial_jacobian(6,6);
  expected_initial_jacobian.setZero();

  EvaluateResidualsAndJacobians(parameters,
                                parameters + 2,
                                cost_function,
                                2, 2, 0, 0, 2,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  EvaluateResidualsAndJacobians(parameters + 2,
                                parameters + 4,
                                cost_function,
                                2, 2, 2, 2, 4,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  EvaluateResidualsAndJacobians(parameters + 4,
                                parameters,
                                cost_function,
                                2, 2, 4, 4, 0,
                                &expected_initial_residuals,
                                &expected_initial_jacobian);

  Solver::Options options;
  Solver::Summary summary;
  options.return_initial_residuals = true;
  options.return_final_residuals = true;
  options.return_initial_jacobian = true;
  options.return_final_jacobian = true;
  SolverImpl::Solve(options, &problem,  &summary);

  Vector expected_final_residuals(6);
  expected_final_residuals.setZero();
  Matrix expected_final_jacobian(6,6);
  expected_final_jacobian.setZero();

  EvaluateResidualsAndJacobians(parameters,
                                parameters + 2,
                                cost_function,
                                2, 2, 0, 0, 2,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EvaluateResidualsAndJacobians(parameters + 2,
                                parameters + 4,
                                cost_function,
                                2, 2, 2, 2, 4,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EvaluateResidualsAndJacobians(parameters + 4,
                                parameters,
                                cost_function,
                                2, 2, 4, 4, 0,
                                &expected_final_residuals,
                                &expected_final_jacobian);

  EXPECT_EQ(summary.initial_residuals.size(), 6);
  VectorRef actual_initial_residuals(&summary.initial_residuals[0], 6);
  COMPARE_ACTUAL_AND_EXPECTED(initial_residuals);

  EXPECT_EQ(summary.initial_jacobian.num_rows, 6);
  EXPECT_EQ(summary.initial_jacobian.num_cols, 6);
  EXPECT_EQ(summary.initial_jacobian.values.size(), 16);
  expected_initial_jacobian.block(0,2,6,2).setZero();
  Matrix actual_initial_jacobian(6,6);
  CRSToDenseMatrix(summary.initial_jacobian, &actual_initial_jacobian);
  COMPARE_ACTUAL_AND_EXPECTED(initial_jacobian);

  EXPECT_EQ(summary.final_residuals.size(), 6);
  VectorRef actual_final_residuals(&summary.final_residuals[0], 6);
  COMPARE_ACTUAL_AND_EXPECTED(final_residuals);

  EXPECT_EQ(summary.final_jacobian.num_rows, 6);
  EXPECT_EQ(summary.final_jacobian.num_cols, 6);
  EXPECT_EQ(summary.final_jacobian.values.size(), 16);
  expected_final_jacobian.block(0,2,6,2).setZero();
  Matrix actual_final_jacobian(6,6);
  CRSToDenseMatrix(summary.final_jacobian, &actual_final_jacobian);
  COMPARE_ACTUAL_AND_EXPECTED(final_jacobian);
}



}  // namespace internal
}  // namespace ceres
