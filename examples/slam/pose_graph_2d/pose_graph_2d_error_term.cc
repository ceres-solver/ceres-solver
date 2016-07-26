#include "pose_graph_2d_error_term.h"

#include <iostream>
#include <cmath>

#include "Eigen/Dense"
#include "normalize_angle.h"

namespace ceres {
namespace examples {
namespace pose_graph_2d {
namespace {
Eigen::Matrix2d RotationMatrix(double yaw_radians) {
  const double cos_yaw = std::cos(yaw_radians);
  const double sin_yaw = std::sin(yaw_radians);

  Eigen::Matrix2d rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}

// Computes the derivative of the rotation matrix w.r.t. to the angle.
Eigen::Matrix2d RotationMatrixDerivative(double yaw_radians) {
  const double cos_yaw = std::cos(yaw_radians);
  const double sin_yaw = std::sin(yaw_radians);

  Eigen::Matrix2d d_R_d_yaw;
  d_R_d_yaw << -sin_yaw, -cos_yaw, cos_yaw, -sin_yaw;
  return d_R_d_yaw;
}

}  // namespace

PoseGraph2dErrorTerm::PoseGraph2dErrorTerm(
    double A_x_B, double A_y_B, double A_yaw_B_radians,
    const Eigen::Matrix3d& sqrt_information)
    : A_p_B_(A_x_B, A_y_B),
      A_yaw_B_radians_(A_yaw_B_radians),
      sqrt_information_(sqrt_information) { }

bool PoseGraph2dErrorTerm::Evaluate(double const* const* parameters,
                                    double* residuals_ptr,
                                    double** jacobians) const {
  CHECK(residuals_ptr != NULL);
  CHECK(parameters != NULL);

  const int kNumParameters = 6;
  for (int i = 0; i < kNumParameters; ++i) {
    CHECK(parameters[i] != NULL) << "parameters[" << i << "] is null.";
  }

  const Eigen::Vector2d G_p_A(*parameters[0], *parameters[1]);
  const double& G_yaw_A = *parameters[2];
  const Eigen::Matrix2d A_R_G = RotationMatrix(G_yaw_A).transpose();

  const Eigen::Vector2d G_p_B(*parameters[3], *parameters[4]);
  const double& G_yaw_B = *parameters[5];

  Eigen::Matrix<double, NUM_RESIDUALS, 1> residuals;
  residuals.head<2>() = A_R_G * (G_p_B - G_p_A) - A_p_B_;
  residuals(2) = NormalizeAngle((G_yaw_B - G_yaw_A) - A_yaw_B_radians_);

  Eigen::Map<Eigen::Matrix<double, NUM_RESIDUALS, 1> > residuals_map(
      residuals_ptr);

  // Scale the residuals by the sqrt information matrix.
  residuals_map = sqrt_information_ * residuals;

  if (jacobians != NULL) {
    // Compute the derivative of the rotation matrix. We need the derivative of
    // the inverse of the rotation matrix, which is equalivant to the transpose
    // of the derivative.
    const Eigen::Matrix2d d_A_R_G_d_yaw =
        RotationMatrixDerivative(G_yaw_A).transpose();

    Eigen::Matrix<double, 3, 6> jacobian = Eigen::Matrix<double, 3, 6>::Zero();

    jacobian.block<2, 2>(0, 0) = -A_R_G;
    jacobian.block<2, 1>(0, 2) = d_A_R_G_d_yaw * (G_p_B - G_p_A);
    jacobian(2, 2) = -1;

    jacobian.block<2, 2>(0, 3) = A_R_G;
    jacobian(2, 5) = 1;

    // Scale the Jacobian by the sqrt information matrix.
    jacobian = sqrt_information_ * jacobian;

    // Jacobian wrt G_x_A.
    if (jacobians[0] != NULL) {
      jacobians[0][0] = jacobian(0, 0);
      jacobians[0][1] = jacobian(1, 0);
      jacobians[0][2] = jacobian(2, 0);
    }
    // Jacobian wrt G_y_A.
    if (jacobians[1] != NULL) {
      jacobians[1][0] = jacobian(0, 1);
      jacobians[1][1] = jacobian(1, 1);
      jacobians[1][2] = jacobian(2, 1);
    }
    // Jacobian wrt G_yaw_A.
    if (jacobians[2] != NULL) {
      jacobians[2][0] = jacobian(0, 2);
      jacobians[2][1] = jacobian(1, 2);
      jacobians[2][2] = jacobian(2, 2);
    }
    // Jacobian wrt G_x_B.
    if (jacobians[3] != NULL) {
      jacobians[3][0] = jacobian(0, 3);
      jacobians[3][1] = jacobian(1, 3);
      jacobians[3][2] = jacobian(2, 3);
    }
    // Jacobian wrt G_y_B.
    if (jacobians[4] != NULL) {
      jacobians[4][0] = jacobian(0, 4);
      jacobians[4][1] = jacobian(1, 4);
      jacobians[4][2] = jacobian(2, 4);
    }
    // Jacobian wrt G_yaw_B.
    if (jacobians[5] != NULL) {
      jacobians[5][0] = jacobian(0, 5);
      jacobians[5][1] = jacobian(1, 5);
      jacobians[5][2] = jacobian(2, 5);
    }
  }

  return true;
}
}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres
