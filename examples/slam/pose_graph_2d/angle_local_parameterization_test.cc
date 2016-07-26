#include "angle_local_parameterization.h"

#include "ceres/autodiff_local_parameterization.h"
#include "gtest/gtest.h"
#include "normalize_angle.h"

namespace {
using ceres::examples::pose_graph_2d::NormalizeAngle;
using ceres::examples::pose_graph_2d::AngleLocalParameterization;

const double kTolerance = 1e-14;

// A templated version of the angle local parameterization to be used with
// automatic differentiation and Jet types to verify the Jacobian.
struct AngleLocalParameterizationPlus {
  template <typename Scalar>
  bool operator()(const Scalar* theta, const Scalar* delta_theta,
                  Scalar* theta_plus_delta) const {
    CHECK(theta != NULL);
    CHECK(delta_theta != NULL);
    CHECK(theta_plus_delta != NULL);

    *theta_plus_delta = NormalizeAngle(*theta + *delta_theta);

    return true;
  }
};

// Verifies the local parameterization maintains the angle in [-pi, pi) and also
// checks the Jacobian matches the autodifferentiation result.
void AngleLocalParameterizationPlusHelper(double theta, double delta_theta) {
  const double kTolerance = 1e-14;

  AngleLocalParameterization angle_local_parameterization;

  double theta_plus_delta = 0.0;
  angle_local_parameterization.Plus(&theta, &delta_theta, &theta_plus_delta);

  // Ensure the update maintains the range [-pi, pi);
  EXPECT_LT(theta_plus_delta, M_PI);
  EXPECT_GE(theta_plus_delta, -M_PI);

  // Autodiff jacobian at delta_x = 0.
  ceres::AutoDiffLocalParameterization<AngleLocalParameterizationPlus, 1, 1>
      autodiff_jacobian;

  double jacobian_autodiff;
  double jacobian_analytic;

  angle_local_parameterization.ComputeJacobian(&theta, &jacobian_analytic);
  autodiff_jacobian.ComputeJacobian(&theta, &jacobian_autodiff);

  EXPECT_TRUE(ceres::IsFinite(jacobian_analytic));
  EXPECT_NEAR(jacobian_analytic, jacobian_autodiff, kTolerance);
}

// Tests the parameterization with zero theta and delta theta.
TEST(AngleLocalParameterization, Zero) {
  double theta = 0.0;
  double delta_theta = 0.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with zero theta and an epsilon delta theta.
TEST(AngleLocalParameterization, ZeroWithEpsilonDelta) {
  double theta = 0.0;
  double delta_theta = 1e-10;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with zero theta and a large delta theta.
TEST(AngleLocalParameterization, ZeroWithLargeDelta) {
  double theta = 0.0;
  double delta_theta = 1.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large positive theta and zero delta theta.
TEST(AngleLocalParameterization, PositiveWithZeroDelta) {
  double theta = 1.1;
  double delta_theta = 0.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large positive theta and an epsilon delta
// theta.
TEST(AngleLocalParameterization, PositiveWithEpsilonDelta) {
  double theta = 1.23;
  double delta_theta = 1e-10;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large positive theta and a large positive
// delta theta.
TEST(AngleLocalParameterization, PositiveWithLargeDelta) {
  double theta = 1.5;
  double delta_theta = 1.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large positive theta and a large negative
// delta theta.
TEST(AngleLocalParameterization, PositiveWithLargeNegativeDelta) {
  double theta = 1.5;
  double delta_theta = -2.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large negative theta and a zero delta
// theta.
TEST(AngleLocalParameterization, NegativeWithZeroDelta) {
  double theta = -1.1;
  double delta_theta = 0.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large negative theta and an epsilon delta
// theta.
TEST(AngleLocalParameterization, NegativeWithEpsilonDelta) {
  double theta = -1.23;
  double delta_theta = -1e-10;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large negative theta and a large negative
// delta theta.
TEST(AngleLocalParameterization, NegativeWithLargeNegativeDelta) {
  double theta = -1.5;
  double delta_theta = -1.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large negative theta and a large negative
// delta theta.
TEST(AngleLocalParameterization, NegativeWithLargePositiveDelta) {
  double theta = -1.5;
  double delta_theta = 2.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

// Tests the parameterization with a large positive theta and a delta theta that
// is more than 360.
TEST(AngleLocalParameterization, PositiveWithLargerThan360DeltaTheta) {
  double theta = 1.5;
  double delta_theta = 20.0;

  AngleLocalParameterizationPlusHelper(theta, delta_theta);
}

}  // namespace
