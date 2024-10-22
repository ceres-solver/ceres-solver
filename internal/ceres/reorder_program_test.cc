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

#include "ceres/reorder_program.h"

#include <random>
#include <vector>

#include "ceres/internal/config.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Templated base class for the CostFunction signatures.
template <int kNumResiduals, int... Ns>
class MockCostFunctionBase : public SizedCostFunction<kNumResiduals, Ns...> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const final {
    // Do nothing. This is never called.
    return true;
  }
};

class UnaryCostFunction : public MockCostFunctionBase<2, 1> {};
class BinaryCostFunction : public MockCostFunctionBase<2, 1, 1> {};
class TernaryCostFunction : public MockCostFunctionBase<2, 1, 1, 1> {};

TEST(_, ReorderResidualBlockNormalFunction) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &z, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &y);

  auto linear_solver_ordering = std::make_shared<ParameterBlockOrdering>();
  linear_solver_ordering->AddElementToGroup(&x, 0);
  linear_solver_ordering->AddElementToGroup(&y, 0);
  linear_solver_ordering->AddElementToGroup(&z, 1);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering = linear_solver_ordering;

  const std::vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();

  std::vector<ResidualBlock*> expected_residual_blocks;

  // This is a bit fragile, but it serves the purpose. We know the
  // bucketing algorithm that the reordering function uses, so we
  // expect the order for residual blocks for each e_block to be
  // filled in reverse.
  expected_residual_blocks.push_back(residual_blocks[4]);
  expected_residual_blocks.push_back(residual_blocks[1]);
  expected_residual_blocks.push_back(residual_blocks[0]);
  expected_residual_blocks.push_back(residual_blocks[5]);
  expected_residual_blocks.push_back(residual_blocks[2]);
  expected_residual_blocks.push_back(residual_blocks[3]);

  Program* program = problem.mutable_program();
  program->SetParameterOffsetsAndIndex();

  std::string message;
  EXPECT_TRUE(LexicographicallyOrderResidualBlocks(
      2, problem.mutable_program(), &message));
  EXPECT_EQ(residual_blocks.size(), expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(residual_blocks[i], expected_residual_blocks[i]);
  }
}

TEST(_, ApplyOrderingOrderingTooSmall) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x, 0);
  linear_solver_ordering.AddElementToGroup(&y, 1);

  Program program(problem.program());
  std::string message;
  EXPECT_FALSE(ApplyOrdering(
      problem.parameter_map(), linear_solver_ordering, &program, &message));
}

TEST(_, ApplyOrderingNormal) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x, 0);
  linear_solver_ordering.AddElementToGroup(&y, 2);
  linear_solver_ordering.AddElementToGroup(&z, 1);

  Program* program = problem.mutable_program();
  std::string message;

  EXPECT_TRUE(ApplyOrdering(
      problem.parameter_map(), linear_solver_ordering, program, &message));
  const std::vector<ParameterBlock*>& parameter_blocks =
      program->parameter_blocks();

  EXPECT_EQ(parameter_blocks.size(), 3);
  EXPECT_EQ(parameter_blocks[0]->user_state(), &x);
  EXPECT_EQ(parameter_blocks[1]->user_state(), &z);
  EXPECT_EQ(parameter_blocks[2]->user_state(), &y);
}

#ifndef CERES_NO_SUITESPARSE
class ReorderProgramForSparseCholeskyUsingSuiteSparseTest
    : public ::testing::Test {
 protected:
  void SetUp() override {
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &x_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &z_, &x_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &z_, &y_);
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &z_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &x_, &y_);
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &y_);
  }

  void ComputeAndValidateOrdering(
      const ParameterBlockOrdering& linear_solver_ordering) {
    Program* program = problem_.mutable_program();
    std::vector<ParameterBlock*> unordered_parameter_blocks =
        program->parameter_blocks();

    std::string error;
    EXPECT_TRUE(ReorderProgramForSparseCholesky(ceres::SUITE_SPARSE,
                                                ceres::AMD,
                                                linear_solver_ordering,
                                                0, /* use all rows */
                                                1, /* single threaded */
                                                program,
                                                &error));
    const std::vector<ParameterBlock*>& ordered_parameter_blocks =
        program->parameter_blocks();
    EXPECT_EQ(ordered_parameter_blocks.size(),
              unordered_parameter_blocks.size());

    EXPECT_THAT(unordered_parameter_blocks,
                ::testing::UnorderedElementsAreArray(ordered_parameter_blocks));
  }

  ProblemImpl problem_;
  double x_;
  double y_;
  double z_;
};

TEST_F(ReorderProgramForSparseCholeskyUsingSuiteSparseTest,
       EverythingInGroupZero) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 0);
  linear_solver_ordering.AddElementToGroup(&z_, 0);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_F(ReorderProgramForSparseCholeskyUsingSuiteSparseTest, ContiguousGroups) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 1);
  linear_solver_ordering.AddElementToGroup(&z_, 2);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_F(ReorderProgramForSparseCholeskyUsingSuiteSparseTest, GroupsWithGaps) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 2);
  linear_solver_ordering.AddElementToGroup(&z_, 2);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_F(ReorderProgramForSparseCholeskyUsingSuiteSparseTest,
       NonContiguousStartingAtTwo) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 2);
  linear_solver_ordering.AddElementToGroup(&y_, 4);
  linear_solver_ordering.AddElementToGroup(&z_, 4);

  ComputeAndValidateOrdering(linear_solver_ordering);
}
#endif  // CERES_NO_SUITESPARSE

TEST(_, ReorderResidualBlocksbyPartition) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &z, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &x, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &y);

  std::vector<ResidualBlockId> residual_block_ids;
  problem.GetResidualBlocks(&residual_block_ids);
  std::vector<ResidualBlock*> residual_blocks =
      problem.program().residual_blocks();
  auto rng = std::mt19937{};
  for (int i = 1; i < 6; ++i) {
    std::shuffle(
        std::begin(residual_block_ids), std::end(residual_block_ids), rng);
    std::unordered_set<ResidualBlockId> bottom(residual_block_ids.begin(),
                                               residual_block_ids.begin() + i);
    const int start_bottom =
        ReorderResidualBlocksByPartition(bottom, problem.mutable_program());
    std::vector<ResidualBlock*> actual_residual_blocks =
        problem.program().residual_blocks();
    EXPECT_THAT(actual_residual_blocks,
                testing::UnorderedElementsAreArray(residual_blocks));
    EXPECT_EQ(start_bottom, residual_blocks.size() - i);
    for (int j = start_bottom; j < residual_blocks.size(); ++j) {
      EXPECT_THAT(bottom, ::testing::Contains(actual_residual_blocks[j]));
    }
  }
}

}  // namespace internal
}  // namespace ceres
