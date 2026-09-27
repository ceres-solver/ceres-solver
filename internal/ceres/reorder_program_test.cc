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
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/reorder_program.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ceres/internal/config.h"
#include "ceres/mkl_ordering.h"
#include "ceres/mkl_sparse_matrix.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "ceres/types.h"
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

// Test that ApplyOrdering preserves the original program order within each
// group. This is essential for deterministic behavior - without preserving
// the original order, the ordering would depend on pointer addresses which
// can vary between runs due to ASLR or different memory allocation patterns.
TEST(_, ApplyOrderingPreservesOrderWithinGroups) {
  // Use heap-allocated parameter blocks to ensure pointer addresses are
  // not in any predictable order (simulating what happens in real usage).
  std::vector<std::unique_ptr<double[]>> params;
  std::vector<double*> param_ptrs;
  const int kNumParams = 10;

  for (int i = 0; i < kNumParams; ++i) {
    params.push_back(std::make_unique<double[]>(3));
    param_ptrs.push_back(params.back().get());
  }

  // Shuffle the parameter pointers to generate non-deterministic addresses
  // in case the allocator happens to return addresses in a specific order.
  // We add them to the problem in a specific order, which
  // should be preserved within each group after ApplyOrdering.
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::shuffle(param_ptrs.begin(), param_ptrs.end(), rng);

  ProblemImpl problem;

  // Add parameter blocks in the order of their index (0, 1, 2, ..., 9)
  // but use the original param_ptrs which may have any address ordering.
  for (int i = 0; i < kNumParams; ++i) {
    problem.AddParameterBlock(param_ptrs[i], 3);
  }

  // Assign parameters to groups:
  // Group 0: params 0, 2, 4, 6, 8 (evens)
  // Group 1: params 1, 3, 5, 7, 9 (odds)
  ParameterBlockOrdering linear_solver_ordering;
  for (int i = 0; i < kNumParams; ++i) {
    linear_solver_ordering.AddElementToGroup(param_ptrs[i], i % 2);
  }

  Program* program = problem.mutable_program();
  std::string message;

  EXPECT_TRUE(ApplyOrdering(
      problem.parameter_map(), linear_solver_ordering, program, &message));

  const std::vector<ParameterBlock*>& parameter_blocks =
      program->parameter_blocks();

  EXPECT_EQ(parameter_blocks.size(), kNumParams);

  // Check that group 0 elements (evens) come first and
  // maintain their original relative order.
  EXPECT_EQ(parameter_blocks[0]->user_state(), param_ptrs[0]);
  EXPECT_EQ(parameter_blocks[1]->user_state(), param_ptrs[2]);
  EXPECT_EQ(parameter_blocks[2]->user_state(), param_ptrs[4]);
  EXPECT_EQ(parameter_blocks[3]->user_state(), param_ptrs[6]);
  EXPECT_EQ(parameter_blocks[4]->user_state(), param_ptrs[8]);

  // Check that group 1 elements (odds) come second and
  // maintain their original relative order.
  EXPECT_EQ(parameter_blocks[5]->user_state(), param_ptrs[1]);
  EXPECT_EQ(parameter_blocks[6]->user_state(), param_ptrs[3]);
  EXPECT_EQ(parameter_blocks[7]->user_state(), param_ptrs[5]);
  EXPECT_EQ(parameter_blocks[8]->user_state(), param_ptrs[7]);
  EXPECT_EQ(parameter_blocks[9]->user_state(), param_ptrs[9]);
}

#if !defined(CERES_NO_SUITESPARSE) || !defined(CERES_NO_MKL)
class ReorderProgramForSparseCholeskyTest
    : public ::testing::TestWithParam<SparseLinearAlgebraLibraryType> {
 protected:
  void SetUp() override {
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &x_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &z_, &x_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &z_, &y_);
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &z_);
    problem_.AddResidualBlock(new BinaryCostFunction(), nullptr, &x_, &y_);
    problem_.AddResidualBlock(new UnaryCostFunction(), nullptr, &y_);
  }

  // Verifies that the reordered parameter blocks are a permutation of the
  // original ones in which lower numbered groups come first.
  void ComputeAndValidateOrdering(
      const ParameterBlockOrdering& linear_solver_ordering) {
    Program* program = problem_.mutable_program();
    std::vector<ParameterBlock*> unordered_parameter_blocks =
        program->parameter_blocks();

    std::string error;
    EXPECT_TRUE(ReorderProgramForSparseCholesky(GetParam(),
                                                ceres::AMD,
                                                linear_solver_ordering,
                                                0, /* use all rows */
                                                1,
                                                program,
                                                &error))
        << error;
    const std::vector<ParameterBlock*>& ordered_parameter_blocks =
        program->parameter_blocks();
    EXPECT_THAT(unordered_parameter_blocks,
                ::testing::UnorderedElementsAreArray(ordered_parameter_blocks));

    std::vector<int> groups;
    for (ParameterBlock* parameter_block : ordered_parameter_blocks) {
      groups.push_back(linear_solver_ordering.GroupId(
          parameter_block->mutable_user_state()));
    }
    EXPECT_TRUE(std::is_sorted(groups.begin(), groups.end()))
        << ::testing::PrintToString(groups);
  }

  ProblemImpl problem_;
  double x_;
  double y_;
  double z_;
};

TEST_P(ReorderProgramForSparseCholeskyTest, EverythingInGroupZero) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 0);
  linear_solver_ordering.AddElementToGroup(&z_, 0);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_P(ReorderProgramForSparseCholeskyTest, ContiguousGroups) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 1);
  linear_solver_ordering.AddElementToGroup(&z_, 2);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_P(ReorderProgramForSparseCholeskyTest, GroupsWithGaps) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 0);
  linear_solver_ordering.AddElementToGroup(&y_, 2);
  linear_solver_ordering.AddElementToGroup(&z_, 2);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

TEST_P(ReorderProgramForSparseCholeskyTest, NonContiguousStartingAtTwo) {
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 2);
  linear_solver_ordering.AddElementToGroup(&y_, 4);
  linear_solver_ordering.AddElementToGroup(&z_, 4);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

// w is absent from the problem but keeps the number of elements equal to the
// number of parameter blocks. z, which the ordering lacks, comes first.
TEST_P(ReorderProgramForSparseCholeskyTest, BlockNotInProgram) {
  double w = 0.0;
  ParameterBlockOrdering linear_solver_ordering;
  linear_solver_ordering.AddElementToGroup(&x_, 1);
  linear_solver_ordering.AddElementToGroup(&y_, 0);
  linear_solver_ordering.AddElementToGroup(&w, 0);

  ComputeAndValidateOrdering(linear_solver_ordering);
}

#ifndef CERES_NO_SUITESPARSE
INSTANTIATE_TEST_SUITE_P(SuiteSparse,
                         ReorderProgramForSparseCholeskyTest,
                         ::testing::Values(SUITE_SPARSE));
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_MKL
INSTANTIATE_TEST_SUITE_P(MklSparse,
                         ReorderProgramForSparseCholeskyTest,
                         ::testing::Values(MKL_SPARSE));
#endif  // CERES_NO_MKL
#endif  // !defined(CERES_NO_SUITESPARSE) || !defined(CERES_NO_MKL)

#ifndef CERES_NO_MKL
// Creates a block structure matrix with unit values.
static void CreateMklMatrix(const int num_rows,
                            const int num_cols,
                            std::vector<MKL_INT> row_offsets,
                            std::vector<MKL_INT> columns,
                            MklCsrMatrix* matrix) {
  std::vector<double> values(columns.size(), 1.0);
  std::string error;
  ASSERT_TRUE(matrix->Create(num_rows,
                             num_cols,
                             std::move(row_offsets),
                             std::move(columns),
                             std::move(values),
                             &error))
      << error;
}

// Returns the user states of the parameter blocks of program in [begin, end).
static std::vector<const double*> UserStates(const Program& program,
                                             const int begin,
                                             const int end) {
  std::vector<const double*> user_states;
  for (int index = begin; index < end; ++index) {
    user_states.push_back(program.parameter_blocks()[index]->user_state());
  }
  return user_states;
}

static void ExpectMklSchurOrdering(const int num_schur_groups) {
  constexpr int kNumEliminationBlocks = 4;
  constexpr int kNumSchurBlocks = 4;
  constexpr int kNumResidualBlocks = 8;

  MklCsrMatrix e_matrix;
  MklCsrMatrix f_matrix;
  ASSERT_NO_FATAL_FAILURE(CreateMklMatrix(kNumResidualBlocks,
                                          kNumEliminationBlocks,
                                          {0, 1, 2, 3, 4, 5, 6, 7, 8},
                                          {0, 0, 1, 1, 2, 2, 3, 3},
                                          &e_matrix));
  ASSERT_NO_FATAL_FAILURE(CreateMklMatrix(kNumResidualBlocks,
                                          kNumSchurBlocks,
                                          {0, 1, 2, 3, 4, 5, 6, 7, 8},
                                          {0, 1, 1, 2, 2, 3, 3, 0},
                                          &f_matrix));
  std::vector<int> expected_schur_ordering(kNumSchurBlocks);
  std::string error;
  ASSERT_TRUE(MklComputeSchurOrdering(
      e_matrix, f_matrix, AMD, 1, expected_schur_ordering.data(), &error))
      << error;
  ASSERT_THAT(expected_schur_ordering,
              ::testing::Not(::testing::ElementsAre(0, 1, 2, 3)));

  ProblemImpl problem;
  double e[kNumEliminationBlocks];
  double f[kNumSchurBlocks];
  for (int i = 0; i < kNumEliminationBlocks; ++i) {
    problem.AddParameterBlock(&e[i], 1);
  }
  for (int i = 0; i < kNumSchurBlocks; ++i) {
    problem.AddParameterBlock(&f[i], 1);
  }
  for (int i = 0; i < kNumSchurBlocks; ++i) {
    const int next = (i + 1) % kNumSchurBlocks;
    problem.AddResidualBlock(new BinaryCostFunction(), nullptr, &e[i], &f[i]);
    problem.AddResidualBlock(
        new BinaryCostFunction(), nullptr, &e[i], &f[next]);
  }

  // Without Schur groups all blocks share group 0 and Ceres chooses the
  // elimination group itself. Otherwise the Schur blocks are distributed over
  // num_schur_groups consecutive groups in program order.
  auto schur_group = [num_schur_groups](const int i) {
    return num_schur_groups == 0 ? 0
                                 : 1 + i * num_schur_groups / kNumSchurBlocks;
  };
  ParameterBlockOrdering ordering;
  for (int i = 0; i < kNumEliminationBlocks; ++i) {
    ordering.AddElementToGroup(&e[i], 0);
  }
  for (int i = 0; i < kNumSchurBlocks; ++i) {
    ordering.AddElementToGroup(&f[i], schur_group(i));
  }

  // PARDISO has no constrained ordering, so Ceres keeps the relative order of
  // its fill reducing ordering within each group.
  std::vector<int> expected_ordering = expected_schur_ordering;
  std::stable_sort(expected_ordering.begin(),
                   expected_ordering.end(),
                   [&schur_group](const int lhs, const int rhs) {
                     return schur_group(lhs) < schur_group(rhs);
                   });
  if (num_schur_groups > 1) {
    ASSERT_NE(expected_ordering, expected_schur_ordering)
        << "The groups must constrain the PARDISO ordering for the test to "
           "be meaningful.";
  }

  Program* program = problem.mutable_program();
  ASSERT_TRUE(ReorderProgramForSchurTypeLinearSolver(SPARSE_SCHUR,
                                                     MKL_SPARSE,
                                                     AMD,
                                                     problem.parameter_map(),
                                                     1,
                                                     &ordering,
                                                     program,
                                                     &error))
      << error;

  std::vector<const double*> expected_schur_blocks;
  for (const int index : expected_ordering) {
    expected_schur_blocks.push_back(&f[index]);
  }
  EXPECT_THAT(UserStates(*program,
                         kNumEliminationBlocks,
                         kNumEliminationBlocks + kNumSchurBlocks),
              ::testing::ElementsAreArray(expected_schur_blocks));
}

class MklFillReducingOrderingTest
    : public ::testing::TestWithParam<LinearSolverOrderingType> {};

// Eliminating the center of a star first fills in the whole matrix, whereas
// eliminating it last causes no fill-in. The center sits in the middle of the
// program so that neither the identity nor its reversal moves it last.
TEST_P(MklFillReducingOrderingTest, EliminatesStarCenterLast) {
  constexpr int kNumParameterBlocks = 5;
  constexpr int kCenter = 2;
  ProblemImpl problem;
  double parameters[kNumParameterBlocks];
  for (double& parameter : parameters) {
    problem.AddParameterBlock(&parameter, 1);
  }
  ParameterBlockOrdering ordering;
  for (int i = 0; i < kNumParameterBlocks; ++i) {
    ordering.AddElementToGroup(&parameters[i], 0);
    if (i != kCenter) {
      problem.AddResidualBlock(new BinaryCostFunction(),
                               nullptr,
                               &parameters[kCenter],
                               &parameters[i]);
    }
  }

  Program* program = problem.mutable_program();
  std::string error;
  ASSERT_TRUE(ReorderProgramForSparseCholesky(
      MKL_SPARSE, GetParam(), ordering, 0, 1, program, &error))
      << error;
  EXPECT_EQ(program->parameter_blocks().back()->user_state(),
            &parameters[kCenter]);
}

INSTANTIATE_TEST_SUITE_P(
    MklSparse,
    MklFillReducingOrderingTest,
    ::testing::Values(AMD, NESDIS),
    [](const ::testing::TestParamInfo<LinearSolverOrderingType>& info) {
      return std::string(LinearSolverOrderingTypeToString(info.param));
    });

// The groups of the Schur blocks constrain the relative order PARDISO chooses
// for them.
class MklSchurGroupOrderingTest : public ::testing::TestWithParam<int> {};

TEST_P(MklSchurGroupOrderingTest, KeepsPardisoOrderWithinGroups) {
  ExpectMklSchurOrdering(GetParam());
}

INSTANTIATE_TEST_SUITE_P(MklSparse,
                         MklSchurGroupOrderingTest,
                         ::testing::Values(0, 2),
                         [](const ::testing::TestParamInfo<int>& info) {
                           return info.param == 0 ? "NoSchurGroups"
                                                  : "TwoSchurGroups";
                         });

// A bundle adjustment problem, whose points SPARSE_SCHUR eliminates using
// MKL_SPARSE.
class MklSchurOrderingTest : public ::testing::Test {
 protected:
  static constexpr int kNumCameras = 4;

  void SetUp() override {
    for (double& camera : cameras_) {
      problem_.AddParameterBlock(&camera, 1);
      ordering_.AddElementToGroup(&camera, 1);
    }
  }

  // Adds a point observed by the given cameras.
  void AddPoint(const std::vector<int>& observed_cameras) {
    double& point = points_.emplace_back(0.0);
    problem_.AddParameterBlock(&point, 1);
    ordering_.AddElementToGroup(&point, 0);
    for (const int camera : observed_cameras) {
      problem_.AddResidualBlock(
          new BinaryCostFunction(), nullptr, &point, &cameras_[camera]);
    }
  }

  // Adds a residual block that depends on two cameras but no point.
  void CoupleCameras(const int first, const int second) {
    problem_.AddResidualBlock(
        new BinaryCostFunction(), nullptr, &cameras_[first], &cameras_[second]);
  }

  void Reorder() {
    std::string error;
    ASSERT_TRUE(
        ReorderProgramForSchurTypeLinearSolver(SPARSE_SCHUR,
                                               MKL_SPARSE,
                                               AMD,
                                               problem_.parameter_map(),
                                               1,
                                               &ordering_,
                                               problem_.mutable_program(),
                                               &error))
        << error;
  }

  const std::vector<ParameterBlock*>& parameter_blocks() const {
    return problem_.program().parameter_blocks();
  }

  ProblemImpl problem_;
  double cameras_[kNumCameras];
  // A deque keeps the addresses of the points stable as points are added.
  std::deque<double> points_;
  ParameterBlockOrdering ordering_;
};

// Camera 1 shares a point with every other camera. Only the E'F part of the
// Schur complement couples it with them, which makes it the center of a star
// that must be eliminated last.
TEST_F(MklSchurOrderingTest, EliminatesCameraSharingPointsWithAllCamerasLast) {
  constexpr int kCenter = 1;
  for (int camera = 0; camera < kNumCameras; ++camera) {
    if (camera != kCenter) {
      AddPoint({kCenter, camera});
    }
  }
  ASSERT_NO_FATAL_FAILURE(Reorder());
  EXPECT_EQ(parameter_blocks().back()->user_state(), &cameras_[kCenter]);
}

// Here, only the F'F part of the Schur complement couples camera 1 with the
// other cameras.
TEST_F(MklSchurOrderingTest, EliminatesCameraCoupledWithAllCamerasLast) {
  constexpr int kCenter = 1;
  for (int camera = 0; camera < kNumCameras; ++camera) {
    AddPoint({camera});
    if (camera != kCenter) {
      CoupleCameras(kCenter, camera);
    }
  }
  ASSERT_NO_FATAL_FAILURE(Reorder());
  EXPECT_EQ(parameter_blocks().back()->user_state(), &cameras_[kCenter]);
}

// Points are eliminated in the order of the first camera they observe, so that
// consecutive points update nearby blocks of the Schur complement.
TEST_F(MklSchurOrderingTest, SortsPointsByFirstCamera) {
  // No camera ordering sorts the points in their original order because the
  // first and the fifth point observe the same camera and the points in
  // between observe the other cameras.
  const std::vector<std::vector<int>> observed_cameras{
      {3}, {2}, {1}, {0}, {3}, {0, 3}};
  for (const std::vector<int>& cameras : observed_cameras) {
    AddPoint(cameras);
  }
  ASSERT_NO_FATAL_FAILURE(Reorder());

  const int num_points = static_cast<int>(points_.size());
  std::vector<int> camera_positions(kNumCameras);
  for (int camera = 0; camera < kNumCameras; ++camera) {
    for (int position = num_points; position < num_points + kNumCameras;
         ++position) {
      if (parameter_blocks()[position]->user_state() == &cameras_[camera]) {
        camera_positions[camera] = position;
      }
    }
  }
  std::vector<int> first_camera_positions(num_points,
                                          std::numeric_limits<int>::max());
  for (int point = 0; point < num_points; ++point) {
    for (const int camera : observed_cameras[point]) {
      first_camera_positions[point] =
          std::min(first_camera_positions[point], camera_positions[camera]);
    }
  }

  std::vector<int> expected_points(num_points);
  std::iota(expected_points.begin(), expected_points.end(), 0);
  auto by_first_camera = [&first_camera_positions](const int lhs,
                                                   const int rhs) {
    return first_camera_positions[lhs] < first_camera_positions[rhs];
  };
  ASSERT_FALSE(std::is_sorted(
      expected_points.begin(), expected_points.end(), by_first_camera));
  std::stable_sort(
      expected_points.begin(), expected_points.end(), by_first_camera);
  std::vector<const double*> expected_point_blocks;
  for (const int point : expected_points) {
    expected_point_blocks.push_back(&points_[point]);
  }
  EXPECT_THAT(UserStates(problem_.program(), 0, num_points),
              ::testing::ElementsAreArray(expected_point_blocks));
}

// oneMKL rejects the Schur complement pattern without columns, which arises
// if every parameter block is eliminated.
TEST(MklSchurOrdering, AcceptsProblemWithoutSchurBlocks) {
  ProblemImpl problem;
  double x = 0.0;
  double y = 0.0;
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &x);
  problem.AddResidualBlock(new UnaryCostFunction(), nullptr, &y);

  ParameterBlockOrdering ordering;
  ordering.AddElementToGroup(&x, 0);
  ordering.AddElementToGroup(&y, 0);

  std::string error;
  EXPECT_TRUE(ReorderProgramForSchurTypeLinearSolver(SPARSE_SCHUR,
                                                     MKL_SPARSE,
                                                     AMD,
                                                     problem.parameter_map(),
                                                     1,
                                                     &ordering,
                                                     problem.mutable_program(),
                                                     &error))
      << error;
}

#endif  // CERES_NO_MKL

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
