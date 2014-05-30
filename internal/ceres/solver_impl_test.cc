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

}  // namespace internal
}  // namespace ceres
