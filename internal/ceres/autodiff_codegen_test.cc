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

#include "ceres/array_utils.h"
#include "ceres/cost_function.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

class BinaryScalarCost {
 public:
  explicit BinaryScalarCost(double v) : localVariable(v) {}
  template <typename T>
  bool operator()(const T* const x, const T* const y, T* cost) const {
    T ex = CERES_EXTERNAL_CONSTANT(localVariable);
    cost[0] = ex * x[0] * y[0] + x[1] * y[1] - T(7) / x[0] * x[1];
    return true;
  }

  bool Evaluate_ResidualAndJacobian(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) {
    // This code is generated with ceres::AutoDiffCodeGen
    // See ceres/autodiff_codegen.h for more informations.
    const double v_0 = parameters[0][0];
    const double v_1 = 1.000000;
    const double v_5 = parameters[0][1];
    const double v_10 = parameters[1][0];
    const double v_15 = parameters[1][1];
    const double v_20 = localVariable;
    const double v_30 = v_20 * v_0;
    const double v_48 = v_30 * v_10;
    const double v_53 = v_20 * v_10;
    const double v_66 = v_5 * v_15;
    const double v_84 = v_48 + v_66;
    const double v_89 = 7.000000;
    const double v_100 = v_1 / v_0;
    const double v_101 = v_89 * v_100;
    const double v_106 = -(v_101);
    const double v_110 = v_106 * v_100;
    const double v_119 = v_101 * v_5;
    const double v_124 = v_110 * v_5;
    const double v_137 = v_84 - v_119;
    const double v_138 = v_53 - v_124;
    const double v_139 = v_15 - v_101;
    residuals[0] = v_137;
    jacobians[0][0] = v_138;
    jacobians[0][1] = v_139;
    jacobians[1][0] = v_30;
    jacobians[1][1] = v_5;
    return true;
  }

 private:
  double localVariable;
};

TEST(AutodiffCodeGen, Simple) {
  CostFunction* cost_function =
      new AutoDiffCostFunction<BinaryScalarCost, 1, 2, 2>(
          new BinaryScalarCost(4));

#if 0
  // Use these lines to generate the code which is included above
  ceres::AutoDiffCodeGen<BinaryScalarCost, 1, 2, 2> codeGen(
      new BinaryScalarCost(4));
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

  BinaryScalarCost functor(4);

  cost_function->Evaluate(parameters, &residuals, jacobians);
  functor.Evaluate_ResidualAndJacobian(parameters, &residuals2, jacobians2);

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

struct ComplicatedCost {
  template <typename T>
  bool operator()(const T* const _x0,
                  const T* const _x1,
                  const T* const _x2,
                  const T* const _x3,
                  T* cost) const {
    auto& x0 = *_x0;
    auto& x1 = *_x1;
    auto& x2 = *_x2;
    auto& x3 = *_x3;

    T t1 = x0 * x1 * x2 * x3;
    T t2 = sin(x0) * cos(x1) * exp(x2) / sqrt(x3);

    cost[0] = t1 * t2;
    cost[1] = t1 / t2;

    return true;
  }

  bool Evaluate_ResidualAndJacobian(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) {
    // This code is generated with ceres::AutoDiffCodeGen
    // See ceres/autodiff_codegen.h for more informations.
    const double v_0 = parameters[0][0];
    const double v_1 = 1.000000;
    const double v_5 = parameters[1][0];
    const double v_10 = parameters[2][0];
    const double v_15 = parameters[3][0];
    const double v_25 = v_0 * v_5;
    const double v_43 = v_25 * v_10;
    const double v_48 = v_5 * v_10;
    const double v_49 = v_0 * v_10;
    const double v_61 = v_43 * v_15;
    const double v_66 = v_48 * v_15;
    const double v_67 = v_49 * v_15;
    const double v_68 = v_25 * v_15;
    const double v_79 = sin(v_0);
    const double v_80 = cos(v_0);
    const double v_90 = cos(v_5);
    const double v_91 = sin(v_5);
    const double v_92 = -(v_91);
    const double v_102 = v_79 * v_90;
    const double v_104 = v_79 * v_92;
    const double v_107 = v_80 * v_90;
    const double v_120 = exp(v_10);
    const double v_130 = v_102 * v_120;
    const double v_135 = v_107 * v_120;
    const double v_136 = v_104 * v_120;
    const double v_148 = sqrt(v_15);
    const double v_149 = 2.000000;
    const double v_150 = v_149 * v_148;
    const double v_154 = v_1 / v_150;
    const double v_161 = v_1 / v_148;
    const double v_162 = v_130 * v_161;
    const double v_166 = v_162 * v_154;
    const double v_170 = -(v_166);
    const double v_171 = v_135 * v_161;
    const double v_172 = v_136 * v_161;
    const double v_174 = v_170 * v_161;
    const double v_180 = v_61 * v_162;
    const double v_181 = v_61 * v_171;
    const double v_182 = v_61 * v_172;
    const double v_184 = v_61 * v_174;
    const double v_185 = v_66 * v_162;
    const double v_186 = v_67 * v_162;
    const double v_187 = v_68 * v_162;
    const double v_188 = v_43 * v_162;
    const double v_189 = v_181 + v_185;
    const double v_190 = v_182 + v_186;
    const double v_191 = v_180 + v_187;
    const double v_192 = v_184 + v_188;
    const double v_199 = v_1 / v_162;
    const double v_200 = v_61 * v_199;
    const double v_201 = v_200 * v_171;
    const double v_202 = v_200 * v_172;
    const double v_203 = v_200 * v_162;
    const double v_204 = v_200 * v_174;
    const double v_205 = v_66 - v_201;
    const double v_206 = v_67 - v_202;
    const double v_207 = v_68 - v_203;
    const double v_208 = v_43 - v_204;
    const double v_209 = v_205 * v_199;
    const double v_210 = v_206 * v_199;
    const double v_211 = v_207 * v_199;
    const double v_212 = v_208 * v_199;
    residuals[0] = v_180;
    residuals[1] = v_200;
    jacobians[0][0] = v_189;
    jacobians[0][1] = v_209;
    jacobians[1][0] = v_190;
    jacobians[1][1] = v_210;
    jacobians[2][0] = v_191;
    jacobians[2][1] = v_211;
    jacobians[3][0] = v_192;
    jacobians[3][1] = v_212;
    return true;
  }
};

TEST(AutodiffCodeGen, Complex) {
  CostFunction* cost_function =
      new AutoDiffCostFunction<ComplicatedCost, 2, 1, 1, 1, 1>(
          new ComplicatedCost);

#if 0
  // Use these lines to generate the code which is included above
  ceres::AutoDiffCodeGen<ComplicatedCost, 2, 1, 1, 1, 1> codeGen(
      new ComplicatedCost);
  codeGen.Generate();
#endif

  const int numParams = 4;
  const int numResiduals = 2;

  double** parameters = new double*[numParams];
  double** jacobians = new double*[numParams];
  double** jacobians2 = new double*[numParams];
  for (int i = 0; i < numParams; ++i) {
    parameters[i] = new double[1];
    parameters[i][0] = double(i + 1) / numParams;
    jacobians[i] = new double[numResiduals];
    jacobians2[i] = new double[numResiduals];
  }

  double* residuals = new double[numResiduals];
  double* residuals2 = new double[numResiduals];

  ComplicatedCost functor;
  cost_function->Evaluate(parameters, residuals, jacobians);
  functor.Evaluate_ResidualAndJacobian(parameters, residuals2, jacobians2);

  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(residuals[i], residuals2[i], 1e-10);
  }

  for (int i = 0; i < numParams; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_NEAR(jacobians[i][j], jacobians2[i][j], 1e-10);
    }
  }

  for (int i = 0; i < numParams; ++i) {
    delete[] jacobians[i];
    delete[] jacobians2[i];
    delete[] parameters[i];
  }
  delete[] residuals;
  delete[] residuals2;
  delete[] jacobians;
  delete[] jacobians2;
  delete[] parameters;
  delete cost_function;
}

}  // namespace internal
}  // namespace ceres
