// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2019 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// A test comparing the result of AutoDiffCodeGen to AutoDiff.

#include "ceres/autodiff_codegen.h"
#include "ceres/autodiff_cost_function.h"

#include <memory>

#include "autodiff_codegen_functor.h"
#include "ceres/array_utils.h"
#include "ceres/cost_function.h"
#include "gtest/gtest.h"
namespace ceres {
namespace internal {

TEST(AutodiffCodeGen, SnavelyReprojectionError) {
  auto functor = new SnavelyReprojectionErrorGen(500, 0);
  CostFunction* cost_function =
      new AutoDiffCostFunction<SnavelyReprojectionErrorGen, 2, 9, 3>(functor);

  std::array<double, 9> parameter_block1 = {.1, .4, .2, 4, 5, 6, 7, 8, 9};
  std::array<double, 3> parameter_block2 = {1, 2, 3};

  std::array<double, 9 * 2> reference_jacobian1, test_jacobian1;
  std::array<double, 3 * 2> reference_jacobian2, test_jacobian2;
  std::array<double, 2> reference_residuals, test_residuals;

  double* parameters[] = {parameter_block1.data(), parameter_block2.data()};
  {
    double* jacobians[] = {reference_jacobian1.data(),
                           reference_jacobian2.data()};
    cost_function->Evaluate(parameters, reference_residuals.data(), jacobians);
  }
  {
    double* jacobians[] = {test_jacobian1.data(), test_jacobian2.data()};
    functor->Evaluate(parameters, test_residuals.data(), jacobians);
  }

  // The generated code may not be exactly equal to autodiff, because of
  // floating point reorderings and simplifications.
  for (int i = 0; i < reference_jacobian1.size(); ++i) {
    EXPECT_NEAR(reference_jacobian1[i], test_jacobian1[i], 1e-40);
  }

  for (int i = 0; i < reference_jacobian2.size(); ++i) {
    EXPECT_NEAR(reference_jacobian2[i], test_jacobian2[i], 1e-40);
  }

  for (int i = 0; i < reference_residuals.size(); ++i) {
    EXPECT_NEAR(reference_residuals[i], test_residuals[i], 1e-40);
  }

  delete cost_function;
}

}  // namespace internal
}  // namespace ceres
