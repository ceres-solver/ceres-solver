// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#include "ceres/program.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ceres/internal/integer_sequence_algorithm.h"
#include "ceres/problem_impl.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/triplet_sparse_matrix.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// A cost function that simply returns its argument.
class UnaryIdentityCostFunction : public SizedCostFunction<1, 1> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    residuals[0] = parameters[0][0];
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = 1.0;
    }
    return true;
  }
};

// Templated base class for the CostFunction signatures.
template <int kNumResiduals, int... Ns>
class MockCostFunctionBase : public SizedCostFunction<kNumResiduals, Ns...> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    constexpr int kNumParameters = (Ns + ... + 0);

    for (int i = 0; i < kNumResiduals; ++i) {
      residuals[i] = kNumResiduals + kNumParameters;
    }
    return true;
  }
};

class UnaryCostFunction : public MockCostFunctionBase<2, 1> {};
class BinaryCostFunction : public MockCostFunctionBase<2, 1, 1> {};
class TernaryCostFunction : public MockCostFunctionBase<2, 1, 1, 1> {};

TEST(Program, RemoveFixedBlocksNothingConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.AddResidualBlock(new TernaryCostFunction(), nullptr, &x, &y, &z);

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));

  EXPECT_EQ(reduced_program->NumParameterBlocks(), 3);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 3);
  EXPECT_EQ(removed_parameter_blocks.size(), 0);
  EXPECT_EQ(fixed_cost, 0.0);
}

TEST(Program, RemoveFixedBlocksAllParameterBlocksConstant) {
  ProblemImpl problem;
  double x = 1.0;

  problem.AddParameterBlock(&x, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.SetParameterBlockConstant(&x);

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));

  EXPECT_EQ(reduced_program->NumParameterBlocks(), 0);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 0);
  EXPECT_EQ(removed_parameter_blocks.size(), 1);
  EXPECT_EQ(removed_parameter_blocks[0], &x);
  EXPECT_EQ(fixed_cost, 9.0);
}

TEST(Program, RemoveFixedBlocksNoResidualBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));
  EXPECT_EQ(reduced_program->NumParameterBlocks(), 0);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 0);
  EXPECT_EQ(removed_parameter_blocks.size(), 3);
  EXPECT_EQ(fixed_cost, 0.0);
}

TEST(Program, RemoveFixedBlocksOneParameterBlockConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.SetParameterBlockConstant(&x);

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));
  EXPECT_EQ(reduced_program->NumParameterBlocks(), 1);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 1);
}

TEST(Program, RemoveFixedBlocksNumEliminateBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), nullptr, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.SetParameterBlockConstant(&x);

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));
  EXPECT_EQ(reduced_program->NumParameterBlocks(), 2);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 2);
}

TEST(Program, RemoveFixedBlocksFixedCost) {
  ProblemImpl problem;
  double x = 1.23;
  double y = 4.56;
  double z = 7.89;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryIdentityCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), nullptr, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.SetParameterBlockConstant(&x);

  ResidualBlock* expected_removed_block =
      problem.program().residual_blocks()[0];
  std::unique_ptr<double[]> scratch(
      new double[expected_removed_block->NumScratchDoublesForEvaluate()]);
  double expected_fixed_cost;
  expected_removed_block->Evaluate(
      true, &expected_fixed_cost, nullptr, nullptr, scratch.get());

  std::vector<double*> removed_parameter_blocks;
  double fixed_cost = 0.0;
  std::string message;
  std::unique_ptr<Program> reduced_program(
      problem.program().CreateReducedProgram(
          &removed_parameter_blocks, &fixed_cost, &message));

  EXPECT_EQ(reduced_program->NumParameterBlocks(), 2);
  EXPECT_EQ(reduced_program->NumResidualBlocks(), 2);
  EXPECT_DOUBLE_EQ(fixed_cost, expected_fixed_cost);
}

class BlockJacobianTest : public ::testing::TestWithParam<int> {};

TEST_P(BlockJacobianTest, CreateJacobianBlockSparsityTranspose) {
  ProblemImpl problem;
  double x[2];
  double y[3];
  double z;

  problem.AddParameterBlock(x, 2);
  problem.AddParameterBlock(y, 3);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new MockCostFunctionBase<2, 2>(), nullptr, x);
  problem.AddResidualBlock(new MockCostFunctionBase<3, 1, 2>(), nullptr, &z, x);
  problem.AddResidualBlock(new MockCostFunctionBase<4, 1, 3>(), nullptr, &z, y);
  problem.AddResidualBlock(new MockCostFunctionBase<5, 1, 3>(), nullptr, &z, y);
  problem.AddResidualBlock(new MockCostFunctionBase<1, 2, 1>(), nullptr, x, &z);
  problem.AddResidualBlock(new MockCostFunctionBase<2, 1, 3>(), nullptr, &z, y);
  problem.AddResidualBlock(new MockCostFunctionBase<2, 2, 1>(), nullptr, x, &z);
  problem.AddResidualBlock(new MockCostFunctionBase<1, 3>(), nullptr, y);

  TripletSparseMatrix expected_block_sparse_jacobian(3, 8, 14);
  {
    int* rows = expected_block_sparse_jacobian.mutable_rows();
    int* cols = expected_block_sparse_jacobian.mutable_cols();
    double* values = expected_block_sparse_jacobian.mutable_values();
    rows[0] = 0;
    cols[0] = 0;

    rows[1] = 2;
    cols[1] = 1;
    rows[2] = 0;
    cols[2] = 1;

    rows[3] = 2;
    cols[3] = 2;
    rows[4] = 1;
    cols[4] = 2;

    rows[5] = 2;
    cols[5] = 3;
    rows[6] = 1;
    cols[6] = 3;

    rows[7] = 0;
    cols[7] = 4;
    rows[8] = 2;
    cols[8] = 4;

    rows[9] = 2;
    cols[9] = 5;
    rows[10] = 1;
    cols[10] = 5;

    rows[11] = 0;
    cols[11] = 6;
    rows[12] = 2;
    cols[12] = 6;

    rows[13] = 1;
    cols[13] = 7;
    std::fill(values, values + 14, 1.0);
    expected_block_sparse_jacobian.set_num_nonzeros(14);
  }

  Program* program = problem.mutable_program();
  program->SetParameterOffsetsAndIndex();

  const int start_row_block = GetParam();
  std::unique_ptr<TripletSparseMatrix> actual_block_sparse_jacobian(
      program->CreateJacobianBlockSparsityTranspose(start_row_block));

  Matrix expected_full_dense_jacobian;
  expected_block_sparse_jacobian.ToDenseMatrix(&expected_full_dense_jacobian);
  Matrix expected_dense_jacobian =
      expected_full_dense_jacobian.rightCols(8 - start_row_block);

  Matrix actual_dense_jacobian;
  actual_block_sparse_jacobian->ToDenseMatrix(&actual_dense_jacobian);
  EXPECT_EQ(expected_dense_jacobian.rows(), actual_dense_jacobian.rows());
  EXPECT_EQ(expected_dense_jacobian.cols(), actual_dense_jacobian.cols());
  EXPECT_EQ((expected_dense_jacobian - actual_dense_jacobian).norm(), 0.0);
}

INSTANTIATE_TEST_SUITE_P(AllColumns, BlockJacobianTest, ::testing::Range(0, 7));

template <int kNumResiduals, int kNumParameterBlocks>
class NumParameterBlocksCostFunction : public CostFunction {
 public:
  NumParameterBlocksCostFunction() {
    set_num_residuals(kNumResiduals);
    for (int i = 0; i < kNumParameterBlocks; ++i) {
      mutable_parameter_block_sizes()->push_back(1);
    }
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    return true;
  }
};

TEST(Program, ReallocationInCreateJacobianBlockSparsityTranspose) {
  // CreateJacobianBlockSparsityTranspose starts with a conservative
  // estimate of the size of the sparsity pattern. This test ensures
  // that when those estimates are violated, the reallocation/resizing
  // logic works correctly.

  ProblemImpl problem;
  double x[20];

  std::vector<double*> parameter_blocks;
  for (int i = 0; i < 20; ++i) {
    problem.AddParameterBlock(x + i, 1);
    parameter_blocks.push_back(x + i);
  }

  problem.AddResidualBlock(new NumParameterBlocksCostFunction<1, 20>(),
                           nullptr,
                           parameter_blocks.data(),
                           static_cast<int>(parameter_blocks.size()));

  TripletSparseMatrix expected_block_sparse_jacobian(20, 1, 20);
  {
    int* rows = expected_block_sparse_jacobian.mutable_rows();
    int* cols = expected_block_sparse_jacobian.mutable_cols();
    for (int i = 0; i < 20; ++i) {
      rows[i] = i;
      cols[i] = 0;
    }

    double* values = expected_block_sparse_jacobian.mutable_values();
    std::fill(values, values + 20, 1.0);
    expected_block_sparse_jacobian.set_num_nonzeros(20);
  }

  Program* program = problem.mutable_program();
  program->SetParameterOffsetsAndIndex();

  std::unique_ptr<TripletSparseMatrix> actual_block_sparse_jacobian(
      program->CreateJacobianBlockSparsityTranspose());

  Matrix expected_dense_jacobian;
  expected_block_sparse_jacobian.ToDenseMatrix(&expected_dense_jacobian);

  Matrix actual_dense_jacobian;
  actual_block_sparse_jacobian->ToDenseMatrix(&actual_dense_jacobian);
  EXPECT_EQ((expected_dense_jacobian - actual_dense_jacobian).norm(), 0.0);
}

TEST(Program, ProblemHasNanParameterBlocks) {
  ProblemImpl problem;
  double x[2];
  x[0] = 1.0;
  x[1] = std::numeric_limits<double>::quiet_NaN();
  problem.AddResidualBlock(new MockCostFunctionBase<1, 2>(), nullptr, x);
  std::string error;
  EXPECT_FALSE(problem.program().ParameterBlocksAreFinite(&error));
  EXPECT_NE(error.find("has at least one invalid value"), std::string::npos)
      << error;
}

TEST(Program, InfeasibleParameterBlock) {
  ProblemImpl problem;
  double x[] = {0.0, 0.0};
  problem.AddResidualBlock(new MockCostFunctionBase<1, 2>(), nullptr, x);
  problem.SetParameterLowerBound(x, 0, 2.0);
  problem.SetParameterUpperBound(x, 0, 1.0);
  std::string error;
  EXPECT_FALSE(problem.program().IsFeasible(&error));
  EXPECT_NE(error.find("infeasible bound"), std::string::npos) << error;
}

TEST(Program, InfeasibleConstantParameterBlock) {
  ProblemImpl problem;
  double x[] = {0.0, 0.0};
  problem.AddResidualBlock(new MockCostFunctionBase<1, 2>(), nullptr, x);
  problem.SetParameterLowerBound(x, 0, 1.0);
  problem.SetParameterUpperBound(x, 0, 2.0);
  problem.SetParameterBlockConstant(x);
  std::string error;
  EXPECT_FALSE(problem.program().IsFeasible(&error));
  EXPECT_NE(error.find("infeasible value"), std::string::npos) << error;
}

}  // namespace internal
}  // namespace ceres
