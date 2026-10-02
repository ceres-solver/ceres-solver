// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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

#include "ceres/coordinate_descent_minimizer.h"

#include <algorithm>
#include <string>
#include <vector>

#include "ceres/minimizer.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "gtest/gtest.h"

namespace ceres::internal {
namespace {

class UnaryResidual final : public SizedCostFunction<1, 1> {
 public:
  explicit UnaryResidual(double target) : target_(target) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    residuals[0] = parameters[0][0] - target_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = 1.0;
    }
    return true;
  }

 private:
  const double target_;
};

class BinaryResidual final : public SizedCostFunction<1, 1, 1> {
 public:
  explicit BinaryResidual(double target) : target_(target) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    residuals[0] = parameters[0][0] + parameters[1][0] - target_;
    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        jacobians[0][0] = 1.0;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = 1.0;
      }
    }
    return true;
  }

 private:
  const double target_;
};

struct BlockMetadata {
  ParameterBlock* block;
  int index;
  int state_offset;
  int delta_offset;
};

std::vector<BlockMetadata> RecordMetadata(ProblemImpl* problem) {
  problem->mutable_program()->SetParameterOffsetsAndIndex();
  std::vector<BlockMetadata> metadata;
  for (ParameterBlock* block : problem->program().parameter_blocks()) {
    EXPECT_FALSE(block->IsConstant());
    metadata.push_back(
        {block, block->index(), block->state_offset(), block->delta_offset()});
  }
  return metadata;
}

void ExpectRestoredMetadata(const std::vector<BlockMetadata>& metadata,
                            const std::vector<double>& state) {
  for (const auto& entry : metadata) {
    EXPECT_EQ(entry.index, entry.block->index());
    EXPECT_EQ(entry.state_offset, entry.block->state_offset());
    EXPECT_EQ(entry.delta_offset, entry.block->delta_offset());
    EXPECT_FALSE(entry.block->IsConstant());
    EXPECT_EQ(state.data() + entry.state_offset, entry.block->state());
  }
}

void MinimizeAndCheck(ProblemImpl* problem,
                      const ParameterBlockOrdering& ordering,
                      std::vector<double>* state,
                      int num_threads) {
  const auto metadata = RecordMetadata(problem);
  const std::vector<double> initial_state = *state;
  ASSERT_EQ(problem->program().NumParameters(), state->size());
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem->program(), ordering, &error))
      << error;
  CoordinateDescentMinimizer minimizer(problem->context());
  ASSERT_TRUE(minimizer.Init(
      problem->program(), problem->parameter_map(), ordering, &error))
      << error;
  Minimizer::Options options;
  options.num_threads = num_threads;
  problem->context()->EnsureMinimumThreads(num_threads);
  Solver::Summary summary;
  minimizer.Minimize(options, state->data(), &summary);
  ExpectRestoredMetadata(metadata, *state);
  const std::vector<double> first_result = *state;
  std::copy(initial_state.begin(), initial_state.end(), state->begin());
  minimizer.Minimize(options, state->data(), &summary);
  ExpectRestoredMetadata(metadata, *state);
  for (int i = 0; i < state->size(); ++i) {
    EXPECT_NEAR(first_result[i], (*state)[i], 1e-6);
  }
}

TEST(CoordinateDescentMinimizer, IndependentBlocksWithOneAndTwoThreads) {
  for (int num_threads : {1, 2}) {
    double x = 0.0;
    double y = 0.0;
    ProblemImpl problem;
    problem.AddResidualBlock(new UnaryResidual(2.0), nullptr, &x);
    problem.AddResidualBlock(new UnaryResidual(-3.0), nullptr, &y);
    ParameterBlockOrdering ordering;
    ordering.AddElementToGroup(&x, 0);
    ordering.AddElementToGroup(&y, 0);
    std::vector<double> state = {0.0, 0.0};

    MinimizeAndCheck(&problem, ordering, &state, num_threads);
    EXPECT_NEAR(2.0, state[0], 1e-6);
    EXPECT_NEAR(-3.0, state[1], 1e-6);
  }
}

TEST(CoordinateDescentMinimizer, CoupledBlocksFollowGroupOrder) {
  for (bool x_first : {true, false}) {
    double x = 0.0;
    double y = 0.0;
    ProblemImpl problem;
    problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
    problem.AddResidualBlock(new UnaryResidual(1.0), nullptr, &y);
    ParameterBlockOrdering ordering;
    ordering.AddElementToGroup(&x, x_first ? 0 : 1);
    ordering.AddElementToGroup(&y, x_first ? 1 : 0);
    std::vector<double> state = {0.0, 0.0};

    MinimizeAndCheck(&problem, ordering, &state, 1);
    EXPECT_NEAR(x_first ? 3.0 : 1.0, state[0], 5e-4);
    EXPECT_NEAR(x_first ? 0.5 : 2.0, state[1], 5e-4);
  }
}

TEST(CoordinateDescentMinimizer, ExcludedBlockRemainsUnchanged) {
  double x = 0.0;
  double z = 1.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(5.0), nullptr, &x, &z);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  std::vector<double> state = {0.0, 1.0};

  MinimizeAndCheck(&problem, ordering, &state, 1);
  EXPECT_NEAR(4.0, state[0], 1e-6);
  EXPECT_DOUBLE_EQ(1.0, state[1]);
}

TEST(CoordinateDescentMinimizer, ValidatesIndependentGroups) {
  double x = 0.0;
  double y = 0.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  std::string message;

  EXPECT_FALSE(CoordinateDescentMinimizer::IsOrderingValid(
      problem.program(), ordering, &message));
  EXPECT_FALSE(message.empty());
  ordering.AddElementToGroup(&y, 1);
  message.clear();
  EXPECT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem.program(), ordering, &message));
}

TEST(CoordinateDescentMinimizer, AutomaticOrderingIncludesIndependentBlocks) {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
  problem.AddResidualBlock(new UnaryResidual(1.0), nullptr, &z);
  const auto ordering =
      CoordinateDescentMinimizer::CreateOrdering(problem.program());
  std::string message;

  ASSERT_NE(nullptr, ordering);
  EXPECT_TRUE(ordering->IsMember(&x));
  EXPECT_TRUE(ordering->IsMember(&y));
  EXPECT_TRUE(ordering->IsMember(&z));
  int num_ordered_blocks = 0;
  for (const auto& group : ordering->group_to_elements()) {
    num_ordered_blocks += group.second.size();
  }
  EXPECT_EQ(3, num_ordered_blocks);
  EXPECT_NE(ordering->GroupId(&x), ordering->GroupId(&y));
  EXPECT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem.program(), *ordering, &message))
      << message;
}

}  // namespace
}  // namespace ceres::internal
