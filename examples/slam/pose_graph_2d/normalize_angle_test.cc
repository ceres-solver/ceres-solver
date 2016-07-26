// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

#include "normalize_angle.h"

#include "gtest/gtest.h"

namespace {
using ceres::examples::pose_graph_2d::NormalizeAngle;

const double kTolerance = 1e-14;

// Tests zero will return 0.
TEST(NormalizeAngle, ZeroTest) {
  EXPECT_NEAR(0.0, NormalizeAngle(0.0), kTolerance);
}

// Tests a positive epsilon.
TEST(NormalizeAngle, PositiveEpsilonTest) {
  const double epsilon = 1e-5;
  EXPECT_NEAR(epsilon, NormalizeAngle(epsilon), kTolerance);
}

// Tests a negative epsilon.
TEST(NormalizeAngle, NegativeEpsilonTest) {
  const double epsilon = -1e-5;
  EXPECT_NEAR(epsilon, NormalizeAngle(epsilon), kTolerance);
}

// Tests that  0 < angle < pi will return angle.
TEST(NormalizeAngle, PositiveTest) {
  const double angle = 1.36;
  EXPECT_NEAR(angle, NormalizeAngle(angle), kTolerance);
}

// Tests that  -pi < angle < 0 will return angle.
TEST(NormalizeAngle, NegativeTest) {
  const double angle = -2.14;
  EXPECT_NEAR(angle, NormalizeAngle(angle), kTolerance);
}

// Tests that pi will wrap to -pi.
TEST(NormalizeAngle, PositivePiTest) {
  EXPECT_NEAR(-M_PI, NormalizeAngle(M_PI), kTolerance);
}

// Tests that -pi will not be modified.
TEST(NormalizeAngle, NegativePiTest) {
  const double angle = -M_PI;
  EXPECT_NEAR(angle, NormalizeAngle(angle), kTolerance);
}

// Tests that angle + 2 * pi will return angle.
TEST(NormalizeAngle, PositiveWrapTest) {
  const double angle = 1.23;
  EXPECT_NEAR(angle, NormalizeAngle(angle + 2.0 * M_PI), kTolerance);
}

// Tests that -angle - 2 * pi will return -angle.
TEST(NormalizeAngle, NegativeWrapTest) {
  const double angle = -0.23;
  EXPECT_NEAR(angle, NormalizeAngle(angle - 2.0 * M_PI), kTolerance);
}

}  // namespace
