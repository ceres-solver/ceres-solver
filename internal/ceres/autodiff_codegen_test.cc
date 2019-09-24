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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/autodiff_cost_function.h"
#include "ceres/autodiff_codegen.h"

#include <memory>

#include "gtest/gtest.h"
#include "ceres/cost_function.h"
#include "ceres/array_utils.h"

namespace ceres {
namespace internal {

class BinaryScalarCost {
 public:
  explicit BinaryScalarCost(double a): a_(a) {}
  template <typename T>
  bool operator()(const T* const x, const T* const y,
                  T* cost) const {
    cost[0] = x[0] * y[0] + x[1] * y[1]  - T(a_);
    return true;
  }
 private:
  double a_;
};


bool BinaryScalarCost_Evaluate(double const* const* parameters, double* residuals, double** jacobians)
{
  // This code is generated with ceres::AutoDiffCodeGen
  // See ceres/autodiff_codegen.h for more informations.
  const double v_25 = parameters[0][0];
  const double v_30 = parameters[0][1];
  const double v_35 = parameters[1][0];
  const double v_40 = parameters[1][1];
  const double v_50 = v_25 * v_35;
  const double v_68 = v_30 * v_40;
  const double v_86 = v_50 + v_68;
  const double v_91 = 1.000000;
  const double v_101 = v_86 - v_91;
  residuals[0] = v_101;
  jacobians[0][0] = v_35;
  jacobians[0][1] = v_40;
  jacobians[1][0] = v_25;
  jacobians[1][1] = v_30;
  return true;
}

TEST(AutodiffCostFunction, BilinearDifferentiationTest) {
  CostFunction* cost_function  =
    new AutoDiffCostFunction<BinaryScalarCost, 1, 2, 2>(
        new BinaryScalarCost(1.0));

#if 0
  // Use these lines to generate the code which is included above
  ceres::AutoDiffCodeGen<BinaryScalarCost, 1, 2,2> codeGen(
      new BinaryScalarCost(1.0));
  codeGen.Generate();
#endif

  double** parameters = new double*[2];
  parameters[0] = new double[2];
  parameters[1] = new double[2];

  parameters[0][0] = 1;
  parameters[0][1] = 2;

  parameters[1][0] = 3;
  parameters[1][1] = 4;

  // Result of ceres::autodiff
  double** jacobians = new double*[2];
  jacobians[0] = new double[2];
  jacobians[1] = new double[2];
  double residuals = 0.0;

  // Result of ceres::autodiffcodegen
  double** jacobians2 = new double*[2];
  jacobians2[0] = new double[2];
  jacobians2[1] = new double[2];
  double residuals2 = 0.0;


  cost_function->Evaluate(parameters, &residuals, jacobians);
  BinaryScalarCost_Evaluate(parameters, &residuals2, jacobians2);

  // compare results between 'normal' autodiff and the generated code
  EXPECT_EQ(residuals, residuals2);
  EXPECT_EQ(jacobians[0][0], jacobians2[0][0]);
  EXPECT_EQ(jacobians[0][1], jacobians2[0][1]);
  EXPECT_EQ(jacobians[1][0], jacobians2[1][0]);
  EXPECT_EQ(jacobians[1][1], jacobians2[1][1]);

  delete[] parameters[0];
  delete[] parameters[1];
  delete[] parameters;

  delete[] jacobians[0];
  delete[] jacobians[1];
  delete[] jacobians;

  delete[] jacobians2[0];
  delete[] jacobians2[1];
  delete[] jacobians2;

  delete cost_function;
}

struct TenParameterCost {
  template <typename T>
  bool operator()(const T* const x0,
                  const T* const x1,
                  const T* const x2,
                  const T* const x3,
                  const T* const x4,
                  const T* const x5,
                  const T* const x6,
                  const T* const x7,
                  const T* const x8,
                  const T* const x9,
                  T* cost) const {
    cost[0] = *x0 + *x1 + *x2 + *x3 + *x4 + *x5 + *x6 + *x7 + *x8 + *x9;
    return true;
  }
};

bool TenParameterCost_Evaluate(double const* const* parameters, double* residuals, double** jacobians)
{
  // This code is generated with ceres::AutoDiffCodeGen
  // See ceres/autodiff_codegen.h for more informations.
  const double v_121 = parameters[0][0];
  const double v_122 = 1.000000;
  const double v_132 = parameters[1][0];
  const double v_133 = 0.000000;
  const double v_134 = 1.000000;
  const double v_143 = parameters[2][0];
  const double v_146 = 1.000000;
  const double v_154 = parameters[3][0];
  const double v_158 = 1.000000;
  const double v_165 = parameters[4][0];
  const double v_170 = 1.000000;
  const double v_176 = parameters[5][0];
  const double v_182 = 1.000000;
  const double v_187 = parameters[6][0];
  const double v_194 = 1.000000;
  const double v_198 = parameters[7][0];
  const double v_206 = 1.000000;
  const double v_209 = parameters[8][0];
  const double v_218 = 1.000000;
  const double v_220 = parameters[9][0];
  const double v_230 = 1.000000;
  const double v_242 = v_121 + v_132;
  const double v_243 = v_122 + v_133;
  const double v_264 = v_242 + v_143;
  const double v_286 = v_264 + v_154;
  const double v_308 = v_286 + v_165;
  const double v_330 = v_308 + v_176;
  const double v_352 = v_330 + v_187;
  const double v_374 = v_352 + v_198;
  const double v_396 = v_374 + v_209;
  const double v_418 = v_396 + v_220;
  residuals[0] = v_418;
  jacobians[0][0] = v_243;
  jacobians[1][0] = v_134;
  jacobians[2][0] = v_146;
  jacobians[3][0] = v_158;
  jacobians[4][0] = v_170;
  jacobians[5][0] = v_182;
  jacobians[6][0] = v_194;
  jacobians[7][0] = v_206;
  jacobians[8][0] = v_218;
  jacobians[9][0] = v_230;
  return true;
}

TEST(AutodiffCostFunction, ManyParameterAutodiffInstantiates) {
  CostFunction* cost_function  =
      new AutoDiffCostFunction<
          TenParameterCost, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
              new TenParameterCost);

#if 0
  // Use these lines to generate the code which is included above
  ceres::AutoDiffCodeGen<TenParameterCost, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1> codeGen(
      new TenParameterCost);
  codeGen.Generate();
#endif

  double** parameters = new double*[10];
  double** jacobians = new double*[10];
  double** jacobians2 = new double*[10];
  for (int i = 0; i < 10; ++i) {
    parameters[i] = new double[1];
    parameters[i][0] = i;
    jacobians[i] = new double[1];
    jacobians2[i] = new double[1];
  }

  double residuals = 0.0;
  double residuals2 = 0.0;



  cost_function->Evaluate(parameters, &residuals, jacobians);
  TenParameterCost_Evaluate(parameters, &residuals2, jacobians2);

  EXPECT_EQ(residuals, residuals2);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(jacobians[i][0], jacobians2[i][0]);
  }

  for (int i = 0; i < 10; ++i) {
    delete[] jacobians[i];
     delete[] jacobians2[i];
    delete[] parameters[i];
  }
  delete[] jacobians;
  delete[] jacobians2;
  delete[] parameters;
  delete cost_function;
}


}  // namespace internal
}  // namespace ceres
