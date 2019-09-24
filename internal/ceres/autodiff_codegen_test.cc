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
    cost[0] = ex * x[0] * y[0] + x[1] * y[1] - T(7) / x[0];

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
    const double v_22 = v_20 * v_0;
    const double v_35 = v_22 * v_10;
    const double v_37 = v_20 * v_10;
    const double v_48 = v_5 * v_15;
    const double v_61 = v_35 + v_48;
    const double v_66 = 7.000000;
    const double v_69 = v_1 / v_0;
    const double v_70 = v_66 * v_69;
    const double v_72 = -(v_70);
    const double v_73 = v_72 * v_69;
    const double v_83 = v_61 - v_70;
    const double v_84 = v_37 - v_73;
    residuals[0] = v_83;
    jacobians[0][0] = v_84;
    jacobians[0][1] = v_15;
    jacobians[1][0] = v_22;
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
  codeGen.Generate2();
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
    const double v_20 = v_0 * v_5;
    const double v_33 = v_20 * v_10;
    const double v_35 = v_5 * v_10;
    const double v_38 = v_0 * v_10;
    const double v_46 = v_33 * v_15;
    const double v_48 = v_35 * v_15;
    const double v_51 = v_38 * v_15;
    const double v_54 = v_20 * v_15;
    const double v_59 = sin(v_0);
    const double v_60 = cos(v_0);
    const double v_65 = cos(v_5);
    const double v_66 = sin(v_5);
    const double v_67 = -(v_66);
    const double v_72 = v_59 * v_65;
    const double v_74 = v_60 * v_65;
    const double v_76 = v_59 * v_67;
    const double v_85 = exp(v_10);
    const double v_90 = v_72 * v_85;
    const double v_92 = v_74 * v_85;
    const double v_95 = v_76 * v_85;
    const double v_103 = sqrt(v_15);
    const double v_105 = 2.000000;
    const double v_106 = v_105 * v_103;
    const double v_107 = v_1 / v_106;
    const double v_113 = v_1 / v_103;
    const double v_114 = v_90 * v_113;
    const double v_117 = v_92 * v_113;
    const double v_120 = v_95 * v_113;
    const double v_124 = v_114 * v_107;
    const double v_125 = -(v_124);
    const double v_126 = v_125 * v_113;
    const double v_127 = v_46 * v_114;
    const double v_128 = v_46 * v_117;
    const double v_129 = v_48 * v_114;
    const double v_130 = v_128 + v_129;
    const double v_131 = v_46 * v_120;
    const double v_132 = v_51 * v_114;
    const double v_133 = v_131 + v_132;
    const double v_135 = v_54 * v_114;
    const double v_136 = v_127 + v_135;
    const double v_137 = v_46 * v_126;
    const double v_138 = v_33 * v_114;
    const double v_139 = v_137 + v_138;
    const double v_141 = v_1 / v_114;
    const double v_142 = v_46 * v_141;
    const double v_143 = v_142 * v_117;
    const double v_144 = v_48 - v_143;
    const double v_145 = v_144 * v_141;
    const double v_146 = v_142 * v_120;
    const double v_147 = v_51 - v_146;
    const double v_148 = v_147 * v_141;
    const double v_149 = v_142 * v_114;
    const double v_150 = v_54 - v_149;
    const double v_151 = v_150 * v_141;
    const double v_152 = v_142 * v_126;
    const double v_153 = v_33 - v_152;
    const double v_154 = v_153 * v_141;
    residuals[0] = v_127;
    residuals[1] = v_142;
    jacobians[0][0] = v_130;
    jacobians[0][1] = v_145;
    jacobians[1][0] = v_133;
    jacobians[1][1] = v_148;
    jacobians[2][0] = v_136;
    jacobians[2][1] = v_151;
    jacobians[3][0] = v_139;
    jacobians[3][1] = v_154;
    return true;
  }
};

TEST(AutodiffCodeGen, Complex) {
  CostFunction* cost_function =
      new AutoDiffCostFunction<ComplicatedCost, 2, 1, 1, 1, 1>(
          new ComplicatedCost);

#if 1
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
