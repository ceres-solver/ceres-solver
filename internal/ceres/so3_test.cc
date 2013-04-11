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

#include "ceres/so3.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/problem.h"
#include "ceres/solver.h"
#include "ceres/rotation.h"
#include "gtest/gtest.h"

#include <cmath>

namespace ceres {
namespace internal {

// Helper function to test rotation matrix Plus functor,
// For given initial rotation angle_axis and delta it
// compares result of Plus functor and result of
//
//   AngleAxisToRotationMatrix(angle_axis + delta)
void RotationMatrixPlusTestHelper(double* angle_axis,
                                  double* delta) {
  const double kTolerance = 1e-14;

  // Initial rotation as rotation matrix
  double R_mat[9];
  AngleAxisToRotationMatrix(angle_axis, R_mat);

  // Apply Plus functor on a matrix and delta
  double R_plus_delta_mat[9];
  RotationMatrixPlus() (R_mat, delta, R_plus_delta_mat);

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
