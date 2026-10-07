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
#include <memory>
#include <string>
#include <vector>

#include "ceres/minimizer.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {
namespace {

constexpr double kStateTolerance = 1e-6;
// The default inner LM stop can leave endpoint error below 3e-4.
constexpr double kCoupledOrderingTolerance = 5e-4;

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
  int index;
  int state_offset;
  int delta_offset;
  bool is_constant;
};

MATCHER_P2(RestoresBlockMetadata,
           expected,
           state_data,
           "restores parameter block metadata") {
  if (arg->index() != expected.index) {
    *result_listener << "index is " << arg->index() << ", expected "
                     << expected.index;
    return false;
  }
  if (arg->state_offset() != expected.state_offset) {
    *result_listener << "state offset is " << arg->state_offset()
                     << ", expected " << expected.state_offset;
    return false;
  }
  if (arg->delta_offset() != expected.delta_offset) {
    *result_listener << "delta offset is " << arg->delta_offset()
                     << ", expected " << expected.delta_offset;
    return false;
  }
  if (arg->IsConstant() != expected.is_constant) {
    *result_listener << "constant state is " << arg->IsConstant()
                     << ", expected " << expected.is_constant;
    return false;
  }
  if (arg->state() != state_data + expected.state_offset) {
    *result_listener << "state pointer does not match offset "
                     << expected.state_offset;
    return false;
  }
  return true;
}

class CoordinateDescentMinimizerTest : public ::testing::Test {
 protected:
  std::vector<BlockMetadata> RecordMetadata() {
    problem_.mutable_program()->SetParameterOffsetsAndIndex();
    std::vector<BlockMetadata> metadata;
    for (ParameterBlock* block : problem_.program().parameter_blocks()) {
      metadata.push_back({block->index(),
                          block->state_offset(),
                          block->delta_offset(),
                          block->IsConstant()});
    }
    return metadata;
  }

  std::unique_ptr<CoordinateDescentMinimizer> CreateMinimizer(
      const ParameterBlockOrdering& ordering, std::string* error) {
    auto minimizer =
        std::make_unique<CoordinateDescentMinimizer>(problem_.context());
    if (!minimizer->Init(
            problem_.program(), problem_.parameter_map(), ordering, error)) {
      return nullptr;
    }
    return minimizer;
  }

  void Minimize(CoordinateDescentMinimizer* minimizer, int num_threads) {
    Minimizer::Options options;
    options.num_threads = num_threads;
    problem_.context()->EnsureMinimumThreads(num_threads);
    Solver::Summary summary;
    minimizer->Minimize(options, state_.data(), &summary);
  }

  double x_ = 0.0;
  double y_ = 0.0;
  double z_ = 0.0;
  std::vector<double> state_;
  ProblemImpl problem_;
};

TEST(CoordinateDescentMinimizerOrdering, RejectsCoupledParametersInSameGroup) {
  double x = 0.0;
  double y = 0.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);
  std::string error;

  EXPECT_FALSE(CoordinateDescentMinimizer::IsOrderingValid(
      problem.program(), ordering, &error));
  EXPECT_FALSE(error.empty());
}

TEST(CoordinateDescentMinimizerOrdering,
     AcceptsCoupledParametersInSeparateGroups) {
  double x = 0.0;
  double y = 0.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 1);
  std::string error;

  EXPECT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem.program(), ordering, &error))
      << error;
}

TEST(CoordinateDescentMinimizerOrdering,
     AutomaticOrderingSeparatesCoupledParameters) {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  ProblemImpl problem;
  problem.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x, &y);
  problem.AddResidualBlock(new UnaryResidual(1.0), nullptr, &z);
  const auto ordering =
      CoordinateDescentMinimizer::CreateOrdering(problem.program());
  std::string error;

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
      problem.program(), *ordering, &error))
      << error;
}

TEST_F(CoordinateDescentMinimizerTest,
       MinimizesIndependentBlocksWithOneAndTwoRequestedThreads) {
  problem_.AddResidualBlock(new UnaryResidual(2.0), nullptr, &x_);
  problem_.AddResidualBlock(new UnaryResidual(-3.0), nullptr, &y_);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x_, 0);
  ordering.AddElementToGroup(&y_, 0);
  state_ = {0.0, 0.0};
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem_.program(), ordering, &error))
      << error;
  auto minimizer = CreateMinimizer(ordering, &error);
  ASSERT_NE(nullptr, minimizer) << error;
  const auto metadata = RecordMetadata();
  ASSERT_EQ(problem_.program().NumParameters(), state_.size());

  // Minimize requests up to one thread per problem in an independent group.
  for (int num_threads : {1, 2}) {
    state_ = {0.0, 0.0};
    Minimize(minimizer.get(), num_threads);
    EXPECT_THAT(problem_.program().parameter_blocks(),
                ::testing::ElementsAre(
                    RestoresBlockMetadata(metadata[0], state_.data()),
                    RestoresBlockMetadata(metadata[1], state_.data())));
    EXPECT_NEAR(2.0, state_[0], kStateTolerance);
    EXPECT_NEAR(-3.0, state_[1], kStateTolerance);
  }
}

TEST_F(CoordinateDescentMinimizerTest, UsesUpdatedXWhenMinimizingY) {
  problem_.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x_, &y_);
  problem_.AddResidualBlock(new UnaryResidual(1.0), nullptr, &y_);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x_, 0);
  ordering.AddElementToGroup(&y_, 1);
  state_ = {0.0, 0.0};
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem_.program(), ordering, &error))
      << error;
  auto minimizer = CreateMinimizer(ordering, &error);
  ASSERT_NE(nullptr, minimizer) << error;
  const auto metadata = RecordMetadata();
  ASSERT_EQ(problem_.program().NumParameters(), state_.size());

  Minimize(minimizer.get(), 1);
  EXPECT_THAT(problem_.program().parameter_blocks(),
              ::testing::ElementsAre(
                  RestoresBlockMetadata(metadata[0], state_.data()),
                  RestoresBlockMetadata(metadata[1], state_.data())));
  // X reaches 3 first. The single Y sweep then minimizes both residuals.
  EXPECT_NEAR(3.0, state_[0], kCoupledOrderingTolerance);
  EXPECT_NEAR(0.5, state_[1], kCoupledOrderingTolerance);
}

TEST_F(CoordinateDescentMinimizerTest, UsesUpdatedYWhenMinimizingX) {
  problem_.AddResidualBlock(new BinaryResidual(3.0), nullptr, &x_, &y_);
  problem_.AddResidualBlock(new UnaryResidual(1.0), nullptr, &y_);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x_, 1);
  ordering.AddElementToGroup(&y_, 0);
  state_ = {0.0, 0.0};
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem_.program(), ordering, &error))
      << error;
  auto minimizer = CreateMinimizer(ordering, &error);
  ASSERT_NE(nullptr, minimizer) << error;
  const auto metadata = RecordMetadata();
  ASSERT_EQ(problem_.program().NumParameters(), state_.size());

  Minimize(minimizer.get(), 1);
  EXPECT_THAT(problem_.program().parameter_blocks(),
              ::testing::ElementsAre(
                  RestoresBlockMetadata(metadata[0], state_.data()),
                  RestoresBlockMetadata(metadata[1], state_.data())));
  // Y reaches 2 first. The single X sweep then minimizes the binary residual.
  EXPECT_NEAR(1.0, state_[0], kCoupledOrderingTolerance);
  EXPECT_NEAR(2.0, state_[1], kCoupledOrderingTolerance);
}

TEST_F(CoordinateDescentMinimizerTest,
       RepeatedMinimizationRestoresMetadataAfterStateReset) {
  problem_.AddResidualBlock(new UnaryResidual(2.0), nullptr, &x_);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x_, 0);
  const std::vector<double> initial_state = {0.0};
  state_ = initial_state;
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem_.program(), ordering, &error))
      << error;
  auto minimizer = CreateMinimizer(ordering, &error);
  ASSERT_NE(nullptr, minimizer) << error;
  const auto metadata = RecordMetadata();
  ASSERT_EQ(problem_.program().NumParameters(), state_.size());

  Minimize(minimizer.get(), 1);
  EXPECT_THAT(problem_.program().parameter_blocks(),
              ::testing::ElementsAre(
                  RestoresBlockMetadata(metadata[0], state_.data())));
  EXPECT_NEAR(2.0, state_[0], kStateTolerance);

  state_ = initial_state;
  Minimize(minimizer.get(), 1);
  EXPECT_THAT(problem_.program().parameter_blocks(),
              ::testing::ElementsAre(
                  RestoresBlockMetadata(metadata[0], state_.data())));
  EXPECT_NEAR(2.0, state_[0], kStateTolerance);
}

TEST_F(CoordinateDescentMinimizerTest, KeepsExcludedParameterBlockFixed) {
  z_ = 1.0;
  problem_.AddResidualBlock(new BinaryResidual(5.0), nullptr, &x_, &z_);
  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x_, 0);
  state_ = {0.0, 1.0};
  std::string error;
  ASSERT_TRUE(CoordinateDescentMinimizer::IsOrderingValid(
      problem_.program(), ordering, &error))
      << error;
  auto minimizer = CreateMinimizer(ordering, &error);
  ASSERT_NE(nullptr, minimizer) << error;
  const auto metadata = RecordMetadata();
  ASSERT_EQ(problem_.program().NumParameters(), state_.size());

  Minimize(minimizer.get(), 1);
  EXPECT_THAT(problem_.program().parameter_blocks(),
              ::testing::ElementsAre(
                  RestoresBlockMetadata(metadata[0], state_.data()),
                  RestoresBlockMetadata(metadata[1], state_.data())));
  EXPECT_NEAR(4.0, state_[0], kStateTolerance);
  EXPECT_DOUBLE_EQ(1.0, state_[1]);
}

}  // namespace
}  // namespace ceres::internal
