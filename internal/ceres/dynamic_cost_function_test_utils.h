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
// Type-parameterized tests shared by the dynamic cost functions. A test
// instantiates them with a traits type such as
//
//   struct Traits {
//     template <typename Functor>
//     using CostFunction = DynamicAutoDiffCostFunction<Functor>;
//     // The tolerance of the Jacobian entries.
//     static constexpr double kTolerance = 0.0;
//   };
//
//   INSTANTIATE_TYPED_TEST_SUITE_P(Name,
//                                  DynamicCostFunctionTest,
//                                  ::testing::Types<Traits>);

#ifndef CERES_INTERNAL_DYNAMIC_COST_FUNCTION_TEST_UTILS_H_
#define CERES_INTERNAL_DYNAMIC_COST_FUNCTION_TEST_UTILS_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

// Takes 2 parameter blocks:
//     parameters[0] is size 10.
//     parameters[1] is size 5.
// Emits 21 residuals:
//     A: i - parameters[0][i], for i in [0,10)  -- this is 10 residuals
//     B: parameters[0][i] - i, for i in [0,10)  -- this is another 10.
//     C: sum(parameters[0][i]^2 - 8*parameters[0][i]) + sum(parameters[1][i])
class MyCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    using std::pow;
    const T* params0 = parameters[0];
    int r = 0;
    for (int i = 0; i < 10; ++i) {
      residuals[r++] = T(i) - params0[i];
      residuals[r++] = params0[i] - T(i);
    }

    T c_residual(0.0);
    for (int i = 0; i < 10; ++i) {
      c_residual += pow(params0[i], 2) - T(8) * params0[i];
    }

    const T* params1 = parameters[1];
    for (int i = 0; i < 5; ++i) {
      c_residual += params1[i];
    }
    residuals[r++] = c_residual;
    return true;
  }
};

// Takes 3 parameter blocks:
//     parameters[0] (x) is size 1.
//     parameters[1] (y) is size 2.
//     parameters[2] (z) is size 3.
// Emits 7 residuals:
//     A: x[0] (= sum_x)
//     B: y[0] + 2.0 * y[1] (= sum_y)
//     C: z[0] + 3.0 * z[1] + 6.0 * z[2] (= sum_z)
//     D: sum_x * sum_y
//     E: sum_y * sum_z
//     F: sum_x * sum_z
//     G: sum_x * sum_y * sum_z
class MyThreeParameterCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* x = parameters[0];
    const T* y = parameters[1];
    const T* z = parameters[2];

    T sum_x = x[0];
    T sum_y = y[0] + 2.0 * y[1];
    T sum_z = z[0] + 3.0 * z[1] + 6.0 * z[2];

    residuals[0] = sum_x;
    residuals[1] = sum_y;
    residuals[2] = sum_z;
    residuals[3] = sum_x * sum_y;
    residuals[4] = sum_y * sum_z;
    residuals[5] = sum_x * sum_z;
    residuals[6] = sum_x * sum_y * sum_z;
    return true;
  }
};

// Takes 6 parameter blocks all of size 1:
//     x0, y0, y1, z0, z1, z2
// Same 7 residuals as MyThreeParameterCostFunctor.
class MySixParameterCostFunctor {
 public:
  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* x0 = parameters[0];
    const T* y0 = parameters[1];
    const T* y1 = parameters[2];
    const T* z0 = parameters[3];
    const T* z1 = parameters[4];
    const T* z2 = parameters[5];

    T sum_x = x0[0];
    T sum_y = y0[0] + 2.0 * y1[0];
    T sum_z = z0[0] + 3.0 * z1[0] + 6.0 * z2[0];

    residuals[0] = sum_x;
    residuals[1] = sum_y;
    residuals[2] = sum_z;
    residuals[3] = sum_x * sum_y;
    residuals[4] = sum_y * sum_z;
    residuals[5] = sum_x * sum_z;
    residuals[6] = sum_x * sum_y * sum_z;
    return true;
  }
};

template <typename Traits>
class DynamicCostFunctionTest : public ::testing::Test {
 protected:
  // Fills the Jacobians with this value to detect entries left unwritten.
  static constexpr double kUnwrittenValue = -100000;

  template <typename Functor>
  static std::unique_ptr<CostFunction> CreateCostFunction(
      const std::vector<int>& parameter_block_sizes, int num_residuals) {
    auto cost_function =
        std::make_unique<typename Traits::template CostFunction<Functor>>(
            new Functor());
    for (const int parameter_block_size : parameter_block_sizes) {
      cost_function->AddParameterBlock(parameter_block_size);
    }
    cost_function->SetNumResiduals(num_residuals);
    return cost_function;
  }

  // Evaluates the cost function requesting the Jacobians of the parameter
  // blocks marked as variable only, and expects the residuals and these
  // Jacobians to match the expected values. The columns of expected_jacobian
  // correspond to the parameters of all parameter blocks.
  static void ExpectEvaluation(const CostFunction& cost_function,
                               const std::vector<double*>& parameter_blocks,
                               const std::vector<bool>& is_variable,
                               const std::vector<double>& expected_residuals,
                               const Matrix& expected_jacobian) {
    const std::vector<int32_t>& parameter_block_sizes =
        cost_function.parameter_block_sizes();
    const int num_residuals = cost_function.num_residuals();

    std::vector<std::vector<double>> jacobian_storage(parameter_blocks.size());
    std::vector<double*> jacobians(parameter_blocks.size(), nullptr);
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      if (is_variable[i]) {
        jacobian_storage[i].resize(num_residuals * parameter_block_sizes[i],
                                   kUnwrittenValue);
        jacobians[i] = jacobian_storage[i].data();
      }
    }

    // Request no Jacobians at all if every parameter block is constant.
    const bool has_variable_block =
        std::find(is_variable.begin(), is_variable.end(), true) !=
        is_variable.end();
    std::vector<double> residuals(num_residuals, kUnwrittenValue);
    ASSERT_TRUE(cost_function.Evaluate(
        parameter_blocks.data(),
        residuals.data(),
        has_variable_block ? jacobians.data() : nullptr));
    EXPECT_THAT(residuals, ::testing::ElementsAreArray(expected_residuals));

    int offset = 0;
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      if (is_variable[i]) {
        SCOPED_TRACE(absl::StrFormat("parameter block %d", i));
        const Matrix expected_block =
            expected_jacobian.middleCols(offset, parameter_block_sizes[i]);
        EXPECT_THAT(
            jacobian_storage[i],
            ::testing::Pointwise(::testing::DoubleNear(Traits::kTolerance),
                                 absl::MakeConstSpan(expected_block)));
      }
      offset += parameter_block_sizes[i];
    }
  }

  // Evaluates MyCostFunctor at parameters[0][i] = 2 * i and parameters[1] = 0.
  void ExpectMyCostFunctorEvaluation(const std::vector<bool>& is_variable) {
    std::vector<double> param_block_0(10);
    for (int i = 0; i < 10; ++i) {
      param_block_0[i] = 2 * i;
    }
    std::vector<double> param_block_1(5, 0.0);
    const std::vector<double*> parameter_blocks{param_block_0.data(),
                                                param_block_1.data()};
    auto cost_function = CreateCostFunction<MyCostFunctor>({10, 5}, 21);

    std::vector<double> expected_residuals(21);
    Matrix expected_jacobian = Matrix::Zero(21, 10 + 5);
    for (int p = 0; p < 10; ++p) {
      expected_residuals[2 * p] = -p;
      expected_residuals[2 * p + 1] = p;
      // "A" and "B" Jacobians.
      expected_jacobian(2 * p, p) = -1.0;
      expected_jacobian(2 * p + 1, p) = 1.0;
      // "C" Jacobian for the first parameter block.
      expected_jacobian(20, p) = 4 * p - 8;
    }
    expected_residuals[20] = 420;
    // "C" Jacobian for the second parameter block.
    expected_jacobian.row(20).tail(5).setOnes();

    ExpectEvaluation(*cost_function,
                     parameter_blocks,
                     is_variable,
                     expected_residuals,
                     expected_jacobian);
  }

  // The parameters and the expected values of MyThreeParameterCostFunctor
  // and MySixParameterCostFunctor.
  static constexpr double kX = 0.0;
  static constexpr double kY0 = 1.0;
  static constexpr double kY1 = 3.0;
  static constexpr double kZ0 = 2.0;
  static constexpr double kZ1 = 4.0;
  static constexpr double kZ2 = 6.0;

  static std::vector<double> ExpectedSumResiduals() {
    const double sum_x = kX;
    const double sum_y = kY0 + 2.0 * kY1;
    const double sum_z = kZ0 + 3.0 * kZ1 + 6.0 * kZ2;
    return {sum_x,
            sum_y,
            sum_z,
            sum_x * sum_y,
            sum_y * sum_z,
            sum_x * sum_z,
            sum_x * sum_y * sum_z};
  }

  // Returns the Jacobian with respect to x0, y0, y1, z0, z1, z2.
  static Matrix ExpectedSumJacobian() {
    const double sum_x = kX;
    const double sum_y = kY0 + 2.0 * kY1;
    const double sum_z = kZ0 + 3.0 * kZ1 + 6.0 * kZ2;
    Matrix jacobian(7, 6);
    // clang-format off
    jacobian <<
      1.0,           0.0,           0.0,                 0.0,           0.0,                 0.0,
      0.0,           1.0,           2.0,                 0.0,           0.0,                 0.0,
      0.0,           0.0,           0.0,                 1.0,           3.0,                 6.0,
      sum_y,         sum_x,         2.0 * sum_x,         0.0,           0.0,                 0.0,
      0.0,           sum_z,         2.0 * sum_z,         sum_y,         3.0 * sum_y,         6.0 * sum_y,
      sum_z,         0.0,           0.0,                 sum_x,         3.0 * sum_x,         6.0 * sum_x,
      sum_y * sum_z, sum_x * sum_z, 2.0 * sum_x * sum_z, sum_x * sum_y, 3.0 * sum_x * sum_y, 6.0 * sum_x * sum_y;
    // clang-format on
    return jacobian;
  }

  void ExpectThreeParameterEvaluation(const std::vector<bool>& is_variable) {
    double x[] = {kX};
    double y[] = {kY0, kY1};
    double z[] = {kZ0, kZ1, kZ2};
    auto cost_function =
        CreateCostFunction<MyThreeParameterCostFunctor>({1, 2, 3}, 7);
    ExpectEvaluation(*cost_function,
                     {x, y, z},
                     is_variable,
                     ExpectedSumResiduals(),
                     ExpectedSumJacobian());
  }

  void ExpectSixParameterEvaluation(const std::vector<bool>& is_variable) {
    double parameters[] = {kX, kY0, kY1, kZ0, kZ1, kZ2};
    std::vector<double*> parameter_blocks;
    for (double& parameter : parameters) {
      parameter_blocks.push_back(&parameter);
    }
    auto cost_function =
        CreateCostFunction<MySixParameterCostFunctor>({1, 1, 1, 1, 1, 1}, 7);
    ExpectEvaluation(*cost_function,
                     parameter_blocks,
                     is_variable,
                     ExpectedSumResiduals(),
                     ExpectedSumJacobian());
  }
};

TYPED_TEST_SUITE_P(DynamicCostFunctionTest);

TYPED_TEST_P(DynamicCostFunctionTest, TestResiduals) {
  std::vector<double> param_block_0(10, 0.0);
  std::vector<double> param_block_1(5, 0.0);
  const std::vector<double*> parameter_blocks{param_block_0.data(),
                                              param_block_1.data()};
  auto cost_function =
      this->template CreateCostFunction<MyCostFunctor>({10, 5}, 21);

  std::vector<double> residuals(21, this->kUnwrittenValue);
  ASSERT_TRUE(cost_function->Evaluate(
      parameter_blocks.data(), residuals.data(), nullptr));

  std::vector<double> expected_residuals(21, 0.0);
  for (int r = 0; r < 10; ++r) {
    expected_residuals[2 * r] = r;
    expected_residuals[2 * r + 1] = -r;
  }
  EXPECT_THAT(residuals, ::testing::ElementsAreArray(expected_residuals));
}

TYPED_TEST_P(DynamicCostFunctionTest, TestJacobian) {
  this->ExpectMyCostFunctorEvaluation({true, true});
}

TYPED_TEST_P(DynamicCostFunctionTest, JacobianWithFirstParameterBlockConstant) {
  this->ExpectMyCostFunctorEvaluation({false, true});
}

TYPED_TEST_P(DynamicCostFunctionTest,
             JacobianWithSecondParameterBlockConstant) {
  this->ExpectMyCostFunctorEvaluation({true, false});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestThreeParameterResiduals) {
  this->ExpectThreeParameterEvaluation({false, false, false});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestThreeParameterJacobian) {
  this->ExpectThreeParameterEvaluation({true, true, true});
}

TYPED_TEST_P(DynamicCostFunctionTest,
             ThreeParameterJacobianWithFirstAndLastParameterBlockConstant) {
  this->ExpectThreeParameterEvaluation({false, true, false});
}

TYPED_TEST_P(DynamicCostFunctionTest,
             ThreeParameterJacobianWithSecondParameterBlockConstant) {
  this->ExpectThreeParameterEvaluation({true, false, true});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestSixParameterResiduals) {
  this->ExpectSixParameterEvaluation(
      {false, false, false, false, false, false});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestSixParameterJacobian) {
  this->ExpectSixParameterEvaluation({true, true, true, true, true, true});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestSixParameterJacobianVVCVVC) {
  this->ExpectSixParameterEvaluation({true, true, false, true, true, false});
}

TYPED_TEST_P(DynamicCostFunctionTest, TestSixParameterJacobianVCCVCV) {
  this->ExpectSixParameterEvaluation({true, false, false, true, false, true});
}

REGISTER_TYPED_TEST_SUITE_P(
    DynamicCostFunctionTest,
    TestResiduals,
    TestJacobian,
    JacobianWithFirstParameterBlockConstant,
    JacobianWithSecondParameterBlockConstant,
    TestThreeParameterResiduals,
    TestThreeParameterJacobian,
    ThreeParameterJacobianWithFirstAndLastParameterBlockConstant,
    ThreeParameterJacobianWithSecondParameterBlockConstant,
    TestSixParameterResiduals,
    TestSixParameterJacobian,
    TestSixParameterJacobianVVCVVC,
    TestSixParameterJacobianVCCVCV);

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_DYNAMIC_COST_FUNCTION_TEST_UTILS_H_
