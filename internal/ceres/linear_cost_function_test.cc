// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/linear_cost_function.h"

#include <vector>

#include "ceres/internal/eigen.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using std::vector;

TEST(LinearCostFunction, JacobianTest) {
  constexpr int kStateDim = 6;
  constexpr int kResidualDim = 2;
  const vector<int> parameter_block_sizes = {1, 2, 3};
  const int num_parameter_blocks = parameter_block_sizes.size();

  Matrix jacobian_expected(kResidualDim, kStateDim);
  for (int i = 0; i < kStateDim * kResidualDim; i++) {
    jacobian_expected.data()[i] = i + 1.0;
  }

  Vector b(kResidualDim);
  b(0) = 2.0;
  b(1) = 3.0;

  Vector x(kStateDim);
  for (int i = 0; i < kStateDim; i++) {
    x(i) = i + 10.0;
  }
  const vector<const double *> parameters = {&x(0), &x(1), &x(3)};

  const Vector residual_expected = jacobian_expected * x + b;

  LinearCostFunction linear_cost_function(jacobian_expected, b,
                                          parameter_block_sizes);
  Vector residual_actual(kResidualDim);

  vector<Matrix> jacobians(num_parameter_blocks);
  double *jacobian_ptrs[num_parameter_blocks];

  for (int i = 0; i < num_parameter_blocks; i++) {
    jacobians[i].resize(kResidualDim, parameter_block_sizes[i]);
    jacobian_ptrs[i] = jacobians[i].data();
  }

  linear_cost_function.Evaluate(parameters.data(), residual_actual.data(),
                                jacobian_ptrs);

  Matrix jacobian_actual(kResidualDim, kStateDim);
  jacobian_actual << jacobians[0], jacobians[1], jacobians[2];

  for (int i = 0; i < kStateDim * kResidualDim; i++) {
    EXPECT_DOUBLE_EQ(jacobian_actual.data()[i], jacobian_expected.data()[i]);
  }
  for (int i = 0; i < kResidualDim; i++) {
    EXPECT_DOUBLE_EQ(residual_actual.data()[i], residual_expected.data()[i]);
  }

  EXPECT_EQ(linear_cost_function.num_residuals(), kResidualDim);
  EXPECT_EQ(linear_cost_function.parameter_block_sizes(),
            parameter_block_sizes);
}

}  // namespace internal
}  // namespace ceres
