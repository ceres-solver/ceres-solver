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

#include "gtest/gtest.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/linear_solver.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/solver_impl.h"
#include "ceres/sized_cost_function.h"

namespace ceres {
namespace internal {

// A cost function that sipmply returns its argument.
class UnaryIdentityCostFunction : public SizedCostFunction<1, 1> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    residuals[0] = parameters[0][0];
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = 1.0;
    }
    return true;
  }
};

// Templated base class for the CostFunction signatures.
template <int kNumResiduals, int N0, int N1, int N2>
class MockCostFunctionBase : public
SizedCostFunction<kNumResiduals, N0, N1, N2> {
 public:
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < kNumResiduals; ++i) {
      residuals[i] = 0.0;
    }
    return true;
  }
};

class UnaryCostFunction : public MockCostFunctionBase<2, 1, 0, 0> {};
class BinaryCostFunction : public MockCostFunctionBase<2, 1, 1, 0> {};
class TernaryCostFunction : public MockCostFunctionBase<2, 1, 1, 1> {};

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
  double fixed_cost;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options,
                                       &problem,
                                       &fixed_cost,
                                       &message));

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

TEST(SolverImpl, AutomaticSchurReorderingRespectsConstantBlocks) {
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
  double fixed_cost;
  scoped_ptr<Program> reduced_program(
      SolverImpl::CreateReducedProgram(&options,
                                       &problem,
                                       &fixed_cost,
                                       &message));

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

TEST(SolverImpl, ApplyUserOrderingOrderingTooSmall) {
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

TEST(SolverImpl, ApplyUserOrderingNormal) {
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

// The parameters must be in separate blocks so that they can be individually
// set constant or not.
struct Quadratic4DCostFunction {
  template <typename T> bool operator()(const T* const x,
                                        const T* const y,
                                        const T* const z,
                                        const T* const w,
                                        T* residual) const {
    // A 4-dimension axis-aligned quadratic.
    residual[0] = T(10.0) - *x +
                  T(20.0) - *y +
                  T(30.0) - *z +
                  T(40.0) - *w;
    return true;
  }
};

TEST(SolverImpl, ConstantParameterBlocksDoNotChangeAndStateInvariantKept) {
  double x = 50.0;
  double y = 50.0;
  double z = 50.0;
  double w = 50.0;
  const double original_x = 50.0;
  const double original_y = 50.0;
  const double original_z = 50.0;
  const double original_w = 50.0;

  scoped_ptr<CostFunction> cost_function(
      new AutoDiffCostFunction<Quadratic4DCostFunction, 1, 1, 1, 1, 1>(
          new Quadratic4DCostFunction));

  Problem::Options problem_options;
  problem_options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;

  ProblemImpl problem(problem_options);
  problem.AddResidualBlock(cost_function.get(), NULL, &x, &y, &z, &w);
  problem.SetParameterBlockConstant(&x);
  problem.SetParameterBlockConstant(&w);

  Solver::Options options;
  options.linear_solver_type = DENSE_QR;

  Solver::Summary summary;
  SolverImpl::Solve(options, &problem, &summary);

  // Verify only the non-constant parameters were mutated.
  EXPECT_EQ(original_x, x);
  EXPECT_NE(original_y, y);
  EXPECT_NE(original_z, z);
  EXPECT_EQ(original_w, w);

  // Check that the parameter block state pointers are pointing back at the
  // user state, instead of inside a random temporary vector made by Solve().
  EXPECT_EQ(&x, problem.program().parameter_blocks()[0]->state());
  EXPECT_EQ(&y, problem.program().parameter_blocks()[1]->state());
  EXPECT_EQ(&z, problem.program().parameter_blocks()[2]->state());
  EXPECT_EQ(&w, problem.program().parameter_blocks()[3]->state());

  EXPECT_TRUE(problem.program().IsValid());
}

TEST(SolverImpl, AlternateLinearSolverForSchurTypeLinearSolver) {
  Solver::Options options;

  options.linear_solver_type = DENSE_QR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);

  options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_NORMAL_CHOLESKY);

  options.linear_solver_type = SPARSE_NORMAL_CHOLESKY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);

  options.linear_solver_type = CGNR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);

  options.linear_solver_type = DENSE_SCHUR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, DENSE_QR);

  options.linear_solver_type = SPARSE_SCHUR;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, SPARSE_NORMAL_CHOLESKY);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = IDENTITY;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, IDENTITY);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = SCHUR_JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = CLUSTER_JACOBI;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);

  options.linear_solver_type = ITERATIVE_SCHUR;
  options.preconditioner_type = CLUSTER_TRIDIAGONAL;
  SolverImpl::AlternateLinearSolverForSchurTypeLinearSolver(&options);
  EXPECT_EQ(options.linear_solver_type, CGNR);
  EXPECT_EQ(options.preconditioner_type, JACOBI);
}

}  // namespace internal
}  // namespace ceres
