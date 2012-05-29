// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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

#include "gtest/gtest.h"
#include "ceres/linear_solver.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/solver_impl.h"
#include "ceres/sized_cost_function.h"

namespace ceres {
namespace internal {

// Templated base class for the CostFunction signatures.
template <int kNumResiduals, int N0, int N1, int N2>
class MockCostFunctionBase : public
SizedCostFunction<kNumResiduals, N0, N1, N2> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    // Do nothing. This is never called.
    return true;
  }
};

class UnaryCostFunction : public MockCostFunctionBase<2, 1, 0, 0> {};
class BinaryCostFunction : public MockCostFunctionBase<2, 1, 1, 0> {};
class TernaryCostFunction : public MockCostFunctionBase<2, 1, 1, 1> {};

TEST(SolverImpl, RemoveFixedBlocksNothingConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);

  string error;
  {
    int num_eliminate_blocks = 0;
    Program program(*problem.mutable_program());
    EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                         &num_eliminate_blocks,
                                                         &error));
    EXPECT_EQ(program.NumParameterBlocks(), 3);
    EXPECT_EQ(program.NumResidualBlocks(), 3);
    EXPECT_EQ(num_eliminate_blocks, 0);
  }

  // Check that num_eliminate_blocks is preserved, when it contains
  // all blocks.
  {
    int num_eliminate_blocks = 3;
    Program program(problem.program());
    EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                         &num_eliminate_blocks,
                                                         &error));
    EXPECT_EQ(program.NumParameterBlocks(), 3);
    EXPECT_EQ(program.NumResidualBlocks(), 3);
    EXPECT_EQ(num_eliminate_blocks, 3);
  }
}

TEST(SolverImpl, RemoveFixedBlocksAllParameterBlocksConstant) {
  ProblemImpl problem;
  double x;

  problem.AddParameterBlock(&x, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.SetParameterBlockConstant(&x);

  int num_eliminate_blocks = 0;
  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &num_eliminate_blocks,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 0);
  EXPECT_EQ(program.NumResidualBlocks(), 0);
  EXPECT_EQ(num_eliminate_blocks, 0);
}

TEST(SolverImpl, RemoveFixedBlocksNoResidualBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  int num_eliminate_blocks = 0;
  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &num_eliminate_blocks,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 0);
  EXPECT_EQ(program.NumResidualBlocks(), 0);
  EXPECT_EQ(num_eliminate_blocks, 0);
}

TEST(SolverImpl, RemoveFixedBlocksOneParameterBlockConstant) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.SetParameterBlockConstant(&x);

  int num_eliminate_blocks = 0;
  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &num_eliminate_blocks,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 1);
  EXPECT_EQ(program.NumResidualBlocks(), 1);
  EXPECT_EQ(num_eliminate_blocks, 0);
}

TEST(SolverImpl, RemoveFixedBlocksNumEliminateBlocks) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);
  problem.SetParameterBlockConstant(&x);

  int num_eliminate_blocks = 2;
  Program program(problem.program());
  string error;
  EXPECT_TRUE(SolverImpl::RemoveFixedBlocksFromProgram(&program,
                                                       &num_eliminate_blocks,
                                                       &error));
  EXPECT_EQ(program.NumParameterBlocks(), 2);
  EXPECT_EQ(program.NumResidualBlocks(), 2);
  EXPECT_EQ(num_eliminate_blocks, 1);
}

TEST(SolverImpl, ReorderResidualBlockNonSchurSolver) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);

  const vector<ResidualBlock*>& residual_blocks =
      problem.program().residual_blocks();
  vector<ResidualBlock*> current_residual_blocks(residual_blocks);

  Solver::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  string error;

  EXPECT_TRUE(SolverImpl::MaybeReorderResidualBlocks(options,
                                                     problem.mutable_program(),
                                                     &error));
  EXPECT_EQ(current_residual_blocks.size(), residual_blocks.size());
  for (int i = 0; i < current_residual_blocks.size(); ++i) {
    EXPECT_EQ(current_residual_blocks[i], residual_blocks[i]);
  }
}

TEST(SolverImpl, ReorderResidualBlockNumEliminateBlockDeathTest) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);
  problem.AddResidualBlock(new UnaryCostFunction(), NULL, &x);
  problem.AddResidualBlock(new TernaryCostFunction(), NULL, &x, &y, &z);
  problem.AddResidualBlock(new BinaryCostFunction(), NULL, &x, &y);

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.num_eliminate_blocks = 0;
  string error;
  EXPECT_DEATH(
      SolverImpl::MaybeReorderResidualBlocks(
          options, problem.mutable_program(), &error),
      "Congratulations");
}

TEST(SolverImpl, ReorderResidualBlockNormalFunction) {
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

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.num_eliminate_blocks = 2;

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

  string error;
  EXPECT_TRUE(SolverImpl::MaybeReorderResidualBlocks(options,
                                                     problem.mutable_program(),
                                                     &error));
  EXPECT_EQ(residual_blocks.size(), expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(residual_blocks[i], expected_residual_blocks[i]);
  }
}

TEST(SolverImpl, ReorderResidualBlockNormalFunctionWithFixedBlocks) {
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

  Solver::Options options;
  options.linear_solver_type = DENSE_SCHUR;
  options.num_eliminate_blocks = 2;

  // Create the reduced program. This should remove the fixed block "z",
  // marking the index to -1 at the same time. x and y also get indices.
  string error;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options, &problem, &error));

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

  EXPECT_TRUE(SolverImpl::MaybeReorderResidualBlocks(options,
                                                     reduced_program.get(),
                                                     &error));

  EXPECT_EQ(reduced_program->residual_blocks().size(),
            expected_residual_blocks.size());
  for (int i = 0; i < expected_residual_blocks.size(); ++i) {
    EXPECT_EQ(reduced_program->residual_blocks()[i],
              expected_residual_blocks[i]);
  }
}

TEST(SolverImpl, ApplyUserOrderingOrderingTooSmall) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  vector<double*> ordering;
  ordering.push_back(&x);
  ordering.push_back(&z);

  Program program(problem.program());
  string error;
  EXPECT_FALSE(SolverImpl::ApplyUserOrdering(problem,
                                             ordering,
                                             &program,
                                             &error));
}

TEST(SolverImpl, ApplyUserOrderingHasDuplicates) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  vector<double*> ordering;
  ordering.push_back(&x);
  ordering.push_back(&z);
  ordering.push_back(&z);

  Program program(problem.program());
  string error;
  EXPECT_FALSE(SolverImpl::ApplyUserOrdering(problem,
                                             ordering,
                                             &program,
                                             &error));
}


TEST(SolverImpl, ApplyUserOrderingNormal) {
  ProblemImpl problem;
  double x;
  double y;
  double z;

  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddParameterBlock(&z, 1);

  vector<double*> ordering;
  ordering.push_back(&x);
  ordering.push_back(&z);
  ordering.push_back(&y);

  Program* program = problem.mutable_program();
  string error;

  EXPECT_TRUE(SolverImpl::ApplyUserOrdering(problem,
                                            ordering,
                                            program,
                                            &error));
  const vector<ParameterBlock*>& parameter_blocks = program->parameter_blocks();

  EXPECT_EQ(parameter_blocks.size(), 3);
  EXPECT_EQ(parameter_blocks[0]->user_state(), &x);
  EXPECT_EQ(parameter_blocks[1]->user_state(), &z);
  EXPECT_EQ(parameter_blocks[2]->user_state(), &y);
}

#if defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
TEST(SolverImpl, CreateLinearSolverNoSuiteSparse) {
  Solver::Options options;
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  string error;
  EXPECT_FALSE(SolverImpl::CreateLinearSolver(&options, &error));
}
#endif  // defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)

TEST(SolverImpl, CreateLinearSolverNegativeMaxNumIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_max_num_iterations = -1;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverNegativeMinNumIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_min_num_iterations = -1;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverMaxLessThanMinIterations) {
  Solver::Options options;
  options.linear_solver_type = DENSE_QR;
  options.linear_solver_min_num_iterations = 10;
  options.linear_solver_max_num_iterations = 5;
  string error;
  EXPECT_EQ(SolverImpl::CreateLinearSolver(&options, &error),
            static_cast<LinearSolver*>(NULL));
}

TEST(SolverImpl, CreateLinearSolverZeroNumEliminateBlocks) {
  Solver::Options options;
  options.num_eliminate_blocks = 0;
  options.linear_solver_type = DENSE_SCHUR;
  string error;
  scoped_ptr<LinearSolver> solver(
      SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_TRUE(solver != NULL);

#if defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);
#else
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);
#endif  // defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
}

TEST(SolverImpl, CreateLinearSolverDenseSchurMultipleThreads) {
  Solver::Options options;
  options.num_eliminate_blocks = 1;
  options.linear_solver_type = DENSE_SCHUR;
  options.num_linear_solver_threads = 2;
  string error;
  scoped_ptr<LinearSolver> solver(
      SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_TRUE(solver != NULL);
  EXPECT_EQ(options.linear_solver_type, DENSE_SCHUR);
  EXPECT_EQ(options.num_linear_solver_threads, 1);
}

TEST(SolverImpl, CreateLinearSolverNormalOperation) {
  Solver::Options options;
  scoped_ptr<LinearSolver> solver;
  options.linear_solver_type = DENSE_QR;
  string error;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);
  EXPECT_TRUE(solver.get() != NULL);

#ifndef CERES_NO_SUITESPARSE
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = SUITE_SPARSE;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);
  EXPECT_TRUE(solver.get() != NULL);
#endif  // CERES_NO_SUITESPARSE

#ifndef CERES_NO_CXSPARSE
  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library = CX_SPARSE;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);
  EXPECT_TRUE(solver.get() != NULL);
#endif  // CERES_NO_CXSPARSE

  options.linear_solver_type = DENSE_SCHUR;
  options.num_eliminate_blocks = 2;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, DENSE_SCHUR);
  EXPECT_TRUE(solver.get() != NULL);

  options.linear_solver_type = SPARSE_SCHUR;
  options.num_eliminate_blocks = 2;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));

#if defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)
  EXPECT_TRUE(SolverImpl::CreateLinearSolver(&options, &error) == NULL);
#else
  EXPECT_TRUE(solver.get() != NULL);
  EXPECT_EQ(options.linear_solver_type, SPARSE_SCHUR);
#endif  // defined(CERES_NO_SUITESPARSE) && defined(CERES_NO_CXSPARSE)

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.num_eliminate_blocks = 2;
  solver.reset(SolverImpl::CreateLinearSolver(&options, &error));
  EXPECT_EQ(options.linear_solver_type, ITERATIVE_SCHUR);
  EXPECT_TRUE(solver.get() != NULL);
}

}  // namespace internal
}  // namespace ceres
