// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: sergey.vfx@gmail.com (Sergey Sharybin)

#include "ceres/so3_local_parameterization.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/rotation.h"
#include "gtest/gtest.h"

#include <cmath>

namespace ceres {
namespace internal {

struct RotationError {
  RotationError(const double x_initial[3], const double Rx_expected[3])
      : x_initial(x_initial), Rx_expected(Rx_expected) {}

  template <typename T>
  bool operator()(const T* const R,  // Rotation 3x3 column-major.
                  T* residuals) const {
    T X[3];
    X[0] = T(x_initial[0]);
    X[1] = T(x_initial[1]);
    X[2] = T(x_initial[2]);

    T Rx_predicted[3];
    Rx_predicted[0] = R[0]*X[0] + R[3]*X[1] + R[6]*X[2];
    Rx_predicted[1] = R[1]*X[0] + R[4]*X[1] + R[7]*X[2];
    Rx_predicted[2] = R[2]*X[0] + R[5]*X[1] + R[8]*X[2];

    residuals[0] = T(Rx_expected[0]) - Rx_predicted[0];
    residuals[1] = T(Rx_expected[1]) - Rx_predicted[1];
    residuals[2] = T(Rx_expected[2]) - Rx_predicted[2];

    return true;
  }

  const double *x_initial;
  const double *Rx_expected;
};

void EstimateRotationMatrixTestHelper(const double* x1,
                                      const double* x2,
                                      const double* angle_axis) {
  const double kTolerance = 1e-10;

  // Final and expected coordinate of points, which
  // are an original coordinates x rotated by angle_axis
  double Rx1_expected[3], Rx2_expected[3];
  AngleAxisRotatePoint(angle_axis, x1, Rx1_expected);
  AngleAxisRotatePoint(angle_axis, x2, Rx2_expected);

  // Use rotation which is close enough to real one as an initial rotation
  double R_initial[3];
  R_initial[0] = angle_axis[0] + M_PI / 10.0;
  R_initial[1] = angle_axis[1] + M_PI / 10.0;
  R_initial[2] = angle_axis[2] + M_PI / 10.0;

  double R[9];
  AngleAxisToRotationMatrix(R_initial, R);

  // Find a rotation matrix
  Problem::Options problem_options;
  Problem problem(problem_options);

  problem.AddResidualBlock(new AutoDiffCostFunction<
      RotationError, 3, 9>(new RotationError(x1, Rx1_expected)),
      NULL,
      R);

  problem.AddResidualBlock(new AutoDiffCostFunction<
      RotationError, 3, 9>(new RotationError(x2, Rx2_expected)),
      NULL,
      R);

  AutoDiffRotationMatrixParameterization *rotation_parameterization =
    new AutoDiffRotationMatrixParameterization;
  problem.SetParameterization(R, rotation_parameterization);

  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.parameter_tolerance = 1e-14;
  options.function_tolerance = 1e-14;
  options.gradient_tolerance = 1e-14;
  options.max_num_iterations = 100;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  EXPECT_NE(summary.termination_type, NUMERICAL_FAILURE);

  LOG(INFO) << summary.FullReport();

  double angle_axis_predicted[3];
  RotationMatrixToAngleAxis(R, angle_axis_predicted);

  // Check matrix is actually a rotation matrix by
  // compositing angle-axis back into rotation matrix
  // back and compare with matrix returned by a solver.
  //
  // NOTE: R*R^T could fail because this method wouldn't
  // catch cases when scale is composited into R matrix.
  double R_recomposited[9];
  AngleAxisToRotationMatrix(angle_axis_predicted, R_recomposited);
  for (int i = 0; i < 9; ++i) {
    EXPECT_TRUE(IsFinite(R_recomposited[i]));
    EXPECT_NEAR(R[i], R_recomposited[i], kTolerance)
        << "Rotation mismatch: i = " << i
        << "\n Expected \n" << R[i]
        << "\n Actual \n" << R_recomposited[i];
  }

  // Check predicted rotation matches actual one
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(IsFinite(angle_axis_predicted[i]));
    EXPECT_NEAR(angle_axis_predicted[i], angle_axis[i], kTolerance)
        << "Angle mismatch: i = " << i
        << "\n Expected = " << angle_axis[i]
        << "\n Actual = " << angle_axis_predicted[i];
  }
}

TEST(SO3LocalParameterization, EstimateRotationZeroTest) {
  double x1[3] = {100.0, 200.0, 400.0};
  double x2[3] = {700.0, 300.0, 200.0};

  double angle_axis[3] = {0.0, 0.0, 0.0};

  EstimateRotationMatrixTestHelper(x1, x2, angle_axis);
}

TEST(SO3LocalParameterization, EstimateRotationXAxis) {
  double x1[3] = {100.0, 200.0, 400.0};
  double x2[3] = {700.0, 300.0, 200.0};

  double angle_axis[3] = {M_PI / 3.0, 0.0, 0.0};

  EstimateRotationMatrixTestHelper(x1, x2, angle_axis);
}

TEST(SO3LocalParameterization, EstimateRotationYAxis) {
  double x1[3] = {100.0, 200.0, 400.0};
  double x2[3] = {700.0, 300.0, 200.0};

  double angle_axis[3] = {0.0, M_PI / 3.0, 0.0};

  EstimateRotationMatrixTestHelper(x1, x2, angle_axis);
}

TEST(SO3LocalParameterization, EstimateRotationZAxis) {
  double x1[3] = {100.0, 200.0, 400.0};
  double x2[3] = {700.0, 300.0, 200.0};

  double angle_axis[3] = {0.0, 0.0, M_PI / 3.0};

  EstimateRotationMatrixTestHelper(x1, x2, angle_axis);
}

TEST(SO3LocalParameterization, EstimateRotationAllAxis) {
  double x1[3] = {100.0, 200.0, 400.0};
  double x2[3] = {700.0, 300.0, 100.0};

  double angle_axis[3] = {M_PI / 2.0, M_PI / 3.0, M_PI / 4.0};

  EstimateRotationMatrixTestHelper(x1, x2, angle_axis);
}

void RotationMatrixPlusTestHelper(double* angle_axis,
                                  double* delta) {
  const double kTolerance = 1e-14;

  // Initial rotation as rotation matrix
  double R_mat[9];
  AngleAxisToRotationMatrix(angle_axis, R_mat);

  // Apply Parameterization::Plus on a matrix and delta
  AutoDiffRotationMatrixParameterization parameterization;

  double R_plus_delta_mat[9];
  parameterization.Plus(R_mat, delta, R_plus_delta_mat);

  // Expected rotation as angle-axis
  double R_plus_delta_angle_axis[3];
  R_plus_delta_angle_axis[0] = angle_axis[0] + delta[0];
  R_plus_delta_angle_axis[1] = angle_axis[1] + delta[1];
  R_plus_delta_angle_axis[2] = angle_axis[2] + delta[2];

  // Expected rotation as rotation matrix
  double R_plus_delta_expected[9];
  AngleAxisToRotationMatrix(R_plus_delta_angle_axis, R_plus_delta_expected);

  // Compare actual and expected rotation matrix
  for (int i = 0; i < 9; ++i) {
    EXPECT_TRUE(IsFinite(R_plus_delta_mat[i]));
    EXPECT_NEAR(R_plus_delta_expected[i], R_plus_delta_mat[i], kTolerance)
        << "Angle mismatch: i = " << i
        << "\n Expected = " << R_plus_delta_expected[i]
        << "\n Actual = " << R_plus_delta_mat[i];
  }
}

TEST(SO3LocalParameterization, RotationMatrixPlusZero) {
  double angle_axis[3] = {M_PI / 2.0, M_PI / 3.0, M_PI / 4.0};
  double delta[3] = {0.0, 0.0, 0.0};

  RotationMatrixPlusTestHelper(angle_axis, delta);
}

TEST(SO3LocalParameterization, RotationMatrixPlusSmallRotation) {
  double angle_axis[3] = {M_PI / 2.0, M_PI / 3.0, M_PI / 4.0};
  double delta[3] = {M_PI / 50.0, M_PI / 60.0, M_PI / 70.0};

  RotationMatrixPlusTestHelper(angle_axis, delta);
}

TEST(SO3LocalParameterization, RotationMatrixPlusFarFromZero) {
  double angle_axis[3] = {M_PI / 2.0, M_PI / 3.0, M_PI / 4.0};
  double delta[3] = {M_PI / 2, M_PI / 3, M_PI / 4};

  RotationMatrixPlusTestHelper(angle_axis, delta);
}

}  // namespace internal
}  // namespace ceres
