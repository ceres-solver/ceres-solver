// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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

#include "ceres/reorder_program.h"

#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(_, ReorderResidualBlockNormalFunction) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);

  ParameterBlockOrdering* linear_solver_ordering = new ParameterBlockOrdering;
  linear_solver_ordering->AddElementToGroup(&x, 0);
  linear_solver_ordering->AddElementToGroup(&y, 0);
  linear_solver_ordering->AddElementToGroup(&z, 1);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering.reset(linear_solver_ordering);

  const vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();

  vector<ResidualBlock*> expected_residual_blocks;

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

  string message;
  EXPECT_TRUE(SolverImpl::LexicographicallyOrderResidualBlocks(
                  2,
                  problem.mutable_program(),
                  &message));
  EXPECT_EQ(residual_blocks.size(), expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(residual_blocks[i], expected_residual_blocks[i]);
  }
}

TEST(_, ReorderResidualBlockNormalFunctionWithFixedBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  // Set one parameter block constant.
  problem.SetParameterBlockConstant(&z);

  // Mark residuals for x's row block with "x" for readability.
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);       // 0 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);  // 1 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 2
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 3
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);  // 4 x
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);  // 5
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);  // 6 x
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);       // 7

  ParameterBlockOrdering* linear_solver_ordering = new ParameterBlockOrdering;
  linear_solver_ordering->AddElementToGroup(&x, 0);
  linear_solver_ordering->AddElementToGroup(&z, 0);
  linear_solver_ordering->AddElementToGroup(&y, 1);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering.reset(linear_solver_ordering);

  // Create the reduced program. This should remove the fixed block "z",
  // marking the index to -1 at the same time. x and y also get indices.
  string message;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options, &problem, NULL, &message));

  const vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();

  // This is a bit fragile, but it serves the purpose. We know the
  // bucketing algorithm that the reordering function uses, so we
  // expect the order for residual blocks for each e_block to be
  // filled in reverse.

  vector<ResidualBlock*> expected_residual_blocks;

  // Row block for residuals involving "x". These are marked "x" in the block
  // of code calling AddResidual() above.
  expected_residual_blocks.push_back(residual_blocks[6]);
  expected_residual_blocks.push_back(residual_blocks[4]);
  expected_residual_blocks.push_back(residual_blocks[1]);
  expected_residual_blocks.push_back(residual_blocks[0]);

  // Row block for residuals involving "y".
  expected_residual_blocks.push_back(residual_blocks[7]);
  expected_residual_blocks.push_back(residual_blocks[5]);
  expected_residual_blocks.push_back(residual_blocks[3]);
  expected_residual_blocks.push_back(residual_blocks[2]);

  EXPECT_EQ(reduced_program->residual_blocks().size(),
            expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(reduced_program->residual_blocks()[i],
              expected_residual_blocks[i]);
  }
}

TEST(_, AutomaticSchurReorderingRespectsConstantBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  // Set one parameter block constant.
  problem.SetParameterBlockConstant(&z);

  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &z, &y);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &z);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &y);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &z);

  ParameterBlockOrdering* linear_solver_ordering = new ParameterBlockOrdering;
  linear_solver_ordering->AddElementToGroup(&x, 0);
  linear_solver_ordering->AddElementToGroup(&z, 0);
  linear_solver_ordering->AddElementToGroup(&y, 0);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.linear_solver_ordering.reset(linear_solver_ordering);

  string message;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options, &problem, NULL, &message));

  const vector<ResidualBlock*>& residual_blocks =
      reduced_program->residual_blocks();
  const vector<ParameterBlock*>& parameter_blocks =
      reduced_program->parameter_blocks();

  const vector<ResidualBlock*>& original_residual_blocks =
      problem.program().residual_blocks();

  EXPECT_EQ(residual_blocks.size(), 8);
  EXPECT_EQ(reduced_program->parameter_blocks().size(), 2);

  // Verify that right parmeter block and the residual blocks have
  // been removed.
  for (int i = 0; i < 8; ++i) {
    EXPECT_NE(residual_blocks[i], original_residual_blocks.back());
  }
  for (int i = 0; i < 2; ++i) {
    EXPECT_NE(parameter_blocks[i]->mutable_user_state(), &z);
  }
}

TEST(_, ApplyUserOrderingOrderingTooSmall) {
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
  string message;
  EXPECT_FALSE(SolverImpl::ApplyUserOrdering(problem.parameter_map(),
                                             &linear_solver_ordering,
                                             &program,
                                             &message));
}

TEST(_, ApplyUserOrderingNormal) {
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
  string message;

  EXPECT_TRUE(SolverImpl::ApplyUserOrdering(problem.parameter_map(),
                                            &linear_solver_ordering,
                                            program,
                                            &message));
  const vector<ParameterBlock*>& parameter_blocks = program->parameter_blocks();

  EXPECT_EQ(parameter_blocks.size(), 3);
  EXPECT_EQ(parameter_blocks[0]->user_state(), &x);
  EXPECT_EQ(parameter_blocks[1]->user_state(), &z);
  EXPECT_EQ(parameter_blocks[2]->user_state(), &y);
}

}  // namespace internal
}  // namespace ceres
