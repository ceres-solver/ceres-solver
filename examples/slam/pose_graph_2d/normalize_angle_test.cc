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
