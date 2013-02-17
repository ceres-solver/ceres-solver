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
//         keir@google.com (Keir Mierle)

#include "ceres/problem.h"
#include "ceres/problem_impl.h"

#include "gtest/gtest.h"
#include "ceres/cost_function.h"
#include "ceres/local_parameterization.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/program.h"
#include "ceres/sized_cost_function.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace internal {

// The following three classes are for the purposes of defining
// function signatures. They have dummy Evaluate functions.

// Trivial cost function that accepts a single argument.
class UnaryCostFunction : public CostFunction {
 public:
  UnaryCostFunction(int num_residuals, int16 parameter_block_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block_size);
  }
  virtual ~UnaryCostFunction() {}

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 1;
    }
    return true;
  }
};

// Trivial cost function that accepts two arguments.
class BinaryCostFunction: public CostFunction {
 public:
  BinaryCostFunction(int num_residuals,
                     int16 parameter_block1_size,
                     int16 parameter_block2_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 2;
    }
    return true;
  }
};

// Trivial cost function that accepts three arguments.
class TernaryCostFunction: public CostFunction {
 public:
  TernaryCostFunction(int num_residuals,
                      int16 parameter_block1_size,
                      int16 parameter_block2_size,
                      int16 parameter_block3_size) {
    set_num_residuals(num_residuals);
    mutable_parameter_block_sizes()->push_back(parameter_block1_size);
    mutable_parameter_block_sizes()->push_back(parameter_block2_size);
    mutable_parameter_block_sizes()->push_back(parameter_block3_size);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    for (int i = 0; i < num_residuals(); ++i) {
      residuals[i] = 3;
    }
    return true;
  }
};

TEST(Problem, AddResidualWithNullCostFunctionDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(NULL, NULL, x),
                            "'cost_function' Must be non NULL");
}

TEST(Problem, AddResidualWithIncorrectNumberOfParameterBlocksDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  // UnaryCostFunction takes only one parameter, but two are passed.
  EXPECT_DEATH_IF_SUPPORTED(
      problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x, y),
      "parameter_blocks.size()");
}

TEST(Problem, AddResidualWithDifferentSizesOnTheSameVariableDies) {
  double x[3];

  Problem problem;
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new UnaryCostFunction(
                                    2, 4 /* 4 != 3 */), NULL, x),
                            "different block sizes");
}

TEST(Problem, AddResidualWithDuplicateParametersDies) {
  double x[3], z[5];

  Problem problem;
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new BinaryCostFunction(2, 3, 3), NULL, x, x),
                            "Duplicate parameter blocks");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
                                new TernaryCostFunction(1, 5, 3, 5),
                                NULL, z, x, z),
                            "Duplicate parameter blocks");
}

TEST(Problem, AddResidualWithIncorrectSizesOfParameterBlockDies) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);
  problem.AddParameterBlock(z, 5);

  // The cost function expects the size of the second parameter, z, to be 4
  // instead of 5 as declared above. This is fatal.
  EXPECT_DEATH_IF_SUPPORTED(problem.AddResidualBlock(
      new BinaryCostFunction(2, 3, 4), NULL, x, z),
               "different block sizes");
}

TEST(Problem, AddResidualAddsDuplicatedParametersOnlyOnce) {
  double x[3], y[4], z[5];

  Problem problem;
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new UnaryCostFunction(2, 4), NULL, y);
  problem.AddResidualBlock(new UnaryCostFunction(2, 5), NULL, z);

  EXPECT_EQ(3, problem.NumParameterBlocks());
  EXPECT_EQ(12, problem.NumParameters());
}

TEST(Problem, AddParameterWithDifferentSizesOnTheSameVariableDies) {
  double x[3], y[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);

  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(x, 4),
                            "different block sizes");
}

static double *IntToPtr(int i) {
  return reinterpret_cast<double*>(sizeof(double) * i);  // NOLINT
}

TEST(Problem, AddParameterWithAliasedParametersDies) {
  // Layout is
  //
  //   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
  //                 [x] x  x  x  x          [y] y  y
  //         o==o==o                 o==o==o           o==o
  //               o--o--o     o--o--o     o--o  o--o--o
  //
  // Parameter block additions are tested as listed above; expected successful
  // ones marked with o==o and aliasing ones marked with o--o.

  Problem problem;
  problem.AddParameterBlock(IntToPtr(5),  5);  // x
  problem.AddParameterBlock(IntToPtr(13), 3);  // y

  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 2),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 3),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 4), 9),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr( 8), 3),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr(12), 2),
                            "Aliasing detected");
  EXPECT_DEATH_IF_SUPPORTED(problem.AddParameterBlock(IntToPtr(14), 3),
                            "Aliasing detected");

  // These ones should work.
  problem.AddParameterBlock(IntToPtr( 2), 3);
  problem.AddParameterBlock(IntToPtr(10), 3);
  problem.AddParameterBlock(IntToPtr(16), 2);

  ASSERT_EQ(5, problem.NumParameterBlocks());
}

TEST(Problem, AddParameterIgnoresDuplicateCalls) {
  double x[3], y[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  problem.AddParameterBlock(y, 4);

  // Creating parameter blocks multiple times is ignored.
  problem.AddParameterBlock(x, 3);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);

  // ... even repeatedly.
  problem.AddParameterBlock(x, 3);
  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);

  // More parameters are fine.
  problem.AddParameterBlock(y, 4);
  problem.AddResidualBlock(new UnaryCostFunction(2, 4), NULL, y);

  EXPECT_EQ(2, problem.NumParameterBlocks());
  EXPECT_EQ(7, problem.NumParameters());
}

TEST(Problem, AddingParametersAndResidualsResultsInExpectedProblem) {
  double x[3], y[4], z[5], w[4];

  Problem problem;
  problem.AddParameterBlock(x, 3);
  EXPECT_EQ(1, problem.NumParameterBlocks());
  EXPECT_EQ(3, problem.NumParameters());

  problem.AddParameterBlock(y, 4);
  EXPECT_EQ(2, problem.NumParameterBlocks());
  EXPECT_EQ(7, problem.NumParameters());

  problem.AddParameterBlock(z, 5);
  EXPECT_EQ(3,  problem.NumParameterBlocks());
  EXPECT_EQ(12, problem.NumParameters());

  // Add a parameter that has a local parameterization.
  w[0] = 1.0; w[1] = 0.0; w[2] = 0.0; w[3] = 0.0;
  problem.AddParameterBlock(w, 4, new QuaternionParameterization);
  EXPECT_EQ(4,  problem.NumParameterBlocks());
  EXPECT_EQ(16, problem.NumParameters());

  problem.AddResidualBlock(new UnaryCostFunction(2, 3), NULL, x);
  problem.AddResidualBlock(new BinaryCostFunction(6, 5, 4) , NULL, z, y);
  problem.AddResidualBlock(new BinaryCostFunction(3, 3, 5), NULL, x, z);
  problem.AddResidualBlock(new BinaryCostFunction(7, 5, 3), NULL, z, x);
  problem.AddResidualBlock(new TernaryCostFunction(1, 5, 3, 4), NULL, z, x, y);

  const int total_residuals = 2 + 6 + 3 + 7 + 1;
  EXPECT_EQ(problem.NumResidualBlocks(), 5);
  EXPECT_EQ(problem.NumResiduals(), total_residuals);
}

class DestructorCountingCostFunction : public SizedCostFunction<3, 4, 5> {
 public:
  explicit DestructorCountingCostFunction(int *num_destructions)
      : num_destructions_(num_destructions) {}

  virtual ~DestructorCountingCostFunction() {
    *num_destructions_ += 1;
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return true;
  }

 private:
  int* num_destructions_;
};

TEST(Problem, ReusedCostFunctionsAreOnlyDeletedOnce) {
  double y[4], z[5];
  int num_destructions = 0;

  // Add a cost function multiple times and check to make sure that
  // the destructor on the cost function is only called once.
  {
    Problem problem;
    problem.AddParameterBlock(y, 4);
    problem.AddParameterBlock(z, 5);

    CostFunction* cost = new DestructorCountingCostFunction(&num_destructions);
    problem.AddResidualBlock(cost, NULL, y, z);
    problem.AddResidualBlock(cost, NULL, y, z);
    problem.AddResidualBlock(cost, NULL, y, z);
    EXPECT_EQ(3, problem.NumResidualBlocks());
  }

  // Check that the destructor was called only once.
  CHECK_EQ(num_destructions, 1);
}

TEST(Problem, CostFunctionsAreDeletedEvenWithRemovals) {
  double y[4], z[5], w[4];
  int num_destructions = 0;
  {
    Problem problem;
    problem.AddParameterBlock(y, 4);
    problem.AddParameterBlock(z, 5);

    CostFunction* cost_yz =
        new DestructorCountingCostFunction(&num_destructions);
    CostFunction* cost_wz =
        new DestructorCountingCostFunction(&num_destructions);
    ResidualBlock* r_yz = problem.AddResidualBlock(cost_yz, NULL, y, z);
    ResidualBlock* r_wz = problem.AddResidualBlock(cost_wz, NULL, w, z);
    EXPECT_EQ(2, problem.NumResidualBlocks());

    // In the current implementation, the destructor shouldn't get run yet.
    problem.RemoveResidualBlock(r_yz);
    CHECK_EQ(num_destructions, 0);
    problem.RemoveResidualBlock(r_wz);
    CHECK_EQ(num_destructions, 0);

    EXPECT_EQ(0, problem.NumResidualBlocks());
  }
  CHECK_EQ(num_destructions, 2);
}

// Make the dynamic problem tests (e.g. for removing residual blocks)
// parameterized on whether the low-latency mode is enabled or not.
//
// This tests against ProblemImpl instead of Problem in order to inspect the
// state of the resulting Program; this is difficult with only the thin Problem
// interface.
struct DynamicProblem : public ::testing::TestWithParam<bool> {
  DynamicProblem() {
    Problem::Options options;
    options.enable_fast_parameter_block_removal = GetParam();
    problem.reset(new ProblemImpl(options));
  }

  ParameterBlock* GetParameterBlock(int block) {
    return problem->program().parameter_blocks()[block];
  }
  ResidualBlock* GetResidualBlock(int block) {
    return problem->program().residual_blocks()[block];
  }

  bool HasResidualBlock(ResidualBlock* residual_block) {
    return find(problem->program().residual_blocks().begin(),
                problem->program().residual_blocks().end(),
                residual_block) != problem->program().residual_blocks().end();
  }

  // The next block of functions until the end are only for testing the
  // residual block removals.
  void ExpectContainsOnly(double* values, ResidualBlock* residual_block) {
    ParameterBlock* parameter_block =
        FindOrDie(problem->parameter_map(), values);
    EXPECT_TRUE(ContainsKey(*(parameter_block->mutable_residual_blocks()),
                            residual_block));
  }

  void ExpectSize(double* values, int size) {
    ParameterBlock* parameter_block =
        FindOrDie(problem->parameter_map(), values);
    EXPECT_EQ(size, parameter_block->mutable_residual_blocks()->size());
  }

  // Degenerate case.
  void ExpectParameterBlockContains(double* values) {
    ExpectSize(values, 0);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1) {
    ExpectSize(values, 1);
    ExpectContainsOnly(values, r1);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2) {
    ExpectSize(values, 2);
    ExpectContainsOnly(values, r1);
    ExpectContainsOnly(values, r2);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2,
                                    ResidualBlock* r3) {
    ExpectSize(values, 3);
    ExpectContainsOnly(values, r1);
    ExpectContainsOnly(values, r2);
    ExpectContainsOnly(values, r3);
  }

  void ExpectParameterBlockContains(double* values,
                                    ResidualBlock* r1,
                                    ResidualBlock* r2,
                                    ResidualBlock* r3,
                                    ResidualBlock* r4) {
    ExpectSize(values, 4);
    ExpectContainsOnly(values, r1);
    ExpectContainsOnly(values, r2);
    ExpectContainsOnly(values, r3);
    ExpectContainsOnly(values, r4);
  }

  scoped_ptr<ProblemImpl> problem;
  double y[4], z[5], w[3];
};

TEST_P(DynamicProblem, RemoveParameterBlockWithNoResiduals) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // w is at the end, which might break the swapping logic so try adding and
  // removing it.
  problem->RemoveParameterBlock(w);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // Now remove z, which is in the middle, and add it back.
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());
  problem->AddParameterBlock(z, 5);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());
  EXPECT_EQ(z, GetParameterBlock(2)->user_state());

  // Now remove everything.
  // y
  problem->RemoveParameterBlock(y);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(z, GetParameterBlock(0)->user_state());
  EXPECT_EQ(w, GetParameterBlock(1)->user_state());

  // z
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(1, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(w, GetParameterBlock(0)->user_state());

  // w
  problem->RemoveParameterBlock(w);
  EXPECT_EQ(0, problem->NumParameterBlocks());
  EXPECT_EQ(0, problem->NumResidualBlocks());
}

TEST_P(DynamicProblem, RemoveParameterBlockWithResiduals) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  EXPECT_EQ(y, GetParameterBlock(0)->user_state());
  EXPECT_EQ(z, GetParameterBlock(1)->user_state());
  EXPECT_EQ(w, GetParameterBlock(2)->user_state());

  // Add all combinations of cost functions.
  CostFunction* cost_yzw = new TernaryCostFunction(1, 4, 5, 3);
  CostFunction* cost_yz  = new BinaryCostFunction (1, 4, 5);
  CostFunction* cost_yw  = new BinaryCostFunction (1, 4, 3);
  CostFunction* cost_zw  = new BinaryCostFunction (1, 5, 3);
  CostFunction* cost_y   = new UnaryCostFunction  (1, 4);
  CostFunction* cost_z   = new UnaryCostFunction  (1, 5);
  CostFunction* cost_w   = new UnaryCostFunction  (1, 3);

  ResidualBlock* r_yzw = problem->AddResidualBlock(cost_yzw, NULL, y, z, w);
  ResidualBlock* r_yz  = problem->AddResidualBlock(cost_yz,  NULL, y, z);
  ResidualBlock* r_yw  = problem->AddResidualBlock(cost_yw,  NULL, y, w);
  ResidualBlock* r_zw  = problem->AddResidualBlock(cost_zw,  NULL, z, w);
  ResidualBlock* r_y   = problem->AddResidualBlock(cost_y,   NULL, y);
  ResidualBlock* r_z   = problem->AddResidualBlock(cost_z,   NULL, z);
  ResidualBlock* r_w   = problem->AddResidualBlock(cost_w,   NULL, w);

  EXPECT_EQ(3, problem->NumParameterBlocks());
  EXPECT_EQ(7, problem->NumResidualBlocks());

  // Remove w, which should remove r_yzw, r_yw, r_zw, r_w.
  problem->RemoveParameterBlock(w);
  ASSERT_EQ(2, problem->NumParameterBlocks());
  ASSERT_EQ(3, problem->NumResidualBlocks());

  ASSERT_FALSE(HasResidualBlock(r_yzw));
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_FALSE(HasResidualBlock(r_yw ));
  ASSERT_FALSE(HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_FALSE(HasResidualBlock(r_w  ));

  // Remove z, which will remove almost everything else.
  problem->RemoveParameterBlock(z);
  ASSERT_EQ(1, problem->NumParameterBlocks());
  ASSERT_EQ(1, problem->NumResidualBlocks());

  ASSERT_FALSE(HasResidualBlock(r_yzw));
  ASSERT_FALSE(HasResidualBlock(r_yz ));
  ASSERT_FALSE(HasResidualBlock(r_yw ));
  ASSERT_FALSE(HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_FALSE(HasResidualBlock(r_z  ));
  ASSERT_FALSE(HasResidualBlock(r_w  ));

  // Remove y; all gone.
  problem->RemoveParameterBlock(y);
  EXPECT_EQ(0, problem->NumParameterBlocks());
  EXPECT_EQ(0, problem->NumResidualBlocks());
}

TEST_P(DynamicProblem, RemoveResidualBlock) {
  problem->AddParameterBlock(y, 4);
  problem->AddParameterBlock(z, 5);
  problem->AddParameterBlock(w, 3);

  // Add all combinations of cost functions.
  CostFunction* cost_yzw = new TernaryCostFunction(1, 4, 5, 3);
  CostFunction* cost_yz  = new BinaryCostFunction (1, 4, 5);
  CostFunction* cost_yw  = new BinaryCostFunction (1, 4, 3);
  CostFunction* cost_zw  = new BinaryCostFunction (1, 5, 3);
  CostFunction* cost_y   = new UnaryCostFunction  (1, 4);
  CostFunction* cost_z   = new UnaryCostFunction  (1, 5);
  CostFunction* cost_w   = new UnaryCostFunction  (1, 3);

  ResidualBlock* r_yzw = problem->AddResidualBlock(cost_yzw, NULL, y, z, w);
  ResidualBlock* r_yz  = problem->AddResidualBlock(cost_yz,  NULL, y, z);
  ResidualBlock* r_yw  = problem->AddResidualBlock(cost_yw,  NULL, y, w);
  ResidualBlock* r_zw  = problem->AddResidualBlock(cost_zw,  NULL, z, w);
  ResidualBlock* r_y   = problem->AddResidualBlock(cost_y,   NULL, y);
  ResidualBlock* r_z   = problem->AddResidualBlock(cost_z,   NULL, z);
  ResidualBlock* r_w   = problem->AddResidualBlock(cost_w,   NULL, w);

  if (GetParam()) {
    // In this test parameterization, there should be back-pointers from the
    // parameter blocks to the residual blocks.
    ExpectParameterBlockContains(y, r_yzw, r_yz, r_yw, r_y);
    ExpectParameterBlockContains(z, r_yzw, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_yzw, r_yw, r_zw, r_w);
  } else {
    // Otherwise, nothing.
    EXPECT_TRUE(GetParameterBlock(0)->mutable_residual_blocks() == NULL);
    EXPECT_TRUE(GetParameterBlock(1)->mutable_residual_blocks() == NULL);
    EXPECT_TRUE(GetParameterBlock(2)->mutable_residual_blocks() == NULL);
  }
  EXPECT_EQ(3, problem->NumParameterBlocks());
  EXPECT_EQ(7, problem->NumResidualBlocks());

  // Remove each residual and check the state after each removal.

  // Remove r_yzw.
  problem->RemoveResidualBlock(r_yzw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(6, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_yw, r_y);
    ExpectParameterBlockContains(z, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_yw, r_zw, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_yw ));
  ASSERT_TRUE (HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_yw.
  problem->RemoveResidualBlock(r_yw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(5, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_zw, r_z);
    ExpectParameterBlockContains(w, r_zw, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_zw ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_zw.
  problem->RemoveResidualBlock(r_zw);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(4, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_z);
    ExpectParameterBlockContains(w, r_w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));
  ASSERT_TRUE (HasResidualBlock(r_w  ));

  // Remove r_w.
  problem->RemoveResidualBlock(r_w);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(3, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_yz, r_y);
    ExpectParameterBlockContains(z, r_yz, r_z);
    ExpectParameterBlockContains(w);
  }
  ASSERT_TRUE (HasResidualBlock(r_yz ));
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));

  // Remove r_yz.
  problem->RemoveResidualBlock(r_yz);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(2, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y, r_y);
    ExpectParameterBlockContains(z, r_z);
    ExpectParameterBlockContains(w);
  }
  ASSERT_TRUE (HasResidualBlock(r_y  ));
  ASSERT_TRUE (HasResidualBlock(r_z  ));

  // Remove the last two.
  problem->RemoveResidualBlock(r_z);
  problem->RemoveResidualBlock(r_y);
  ASSERT_EQ(3, problem->NumParameterBlocks());
  ASSERT_EQ(0, problem->NumResidualBlocks());
  if (GetParam()) {
    ExpectParameterBlockContains(y);
    ExpectParameterBlockContains(z);
    ExpectParameterBlockContains(w);
  }
}

INSTANTIATE_TEST_CASE_P(OptionsInstantiation,
                        DynamicProblem,
                        ::testing::Values(true, false));

}  // namespace internal
}  // namespace ceres
