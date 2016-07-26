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
//
// Cost function for a 2D pose graph formulation.

#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_

#include "Eigen/Core"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> RotationMatrix(Scalar yaw_radians) {
  const Scalar cos_yaw = ceres::cos(yaw_radians);
  const Scalar sin_yaw = ceres::sin(yaw_radians);

  Eigen::Matrix<Scalar, 2, 2> rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}

// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement.
//
// residual =  information^{1/2} * [  A_R_G * (G_p_B - G_p_A) - \hat{A_p_B}   ]
//                                 [ Normalize(yaw_B - yaw_A - \hat{A_yaw_B}) ]
//
// where A_R_G is the rotation matrix that rotates a vector represented in the
// global frame into frame A (it is the inverse of the rotation matrix built
// from the yaw angle of A, yaw_A), and Normalize(*) ensures the angles are in
// the range [-pi, pi).
class PoseGraph2dErrorTerm {
 public:
  PoseGraph2dErrorTerm(double A_x_B, double A_y_B, double A_yaw_B_radians,
                       const Eigen::Matrix3d& sqrt_information)
      : A_p_B_(A_x_B, A_y_B),
        A_yaw_B_radians_(A_yaw_B_radians),
        sqrt_information_(sqrt_information) {}

  template <typename Scalar>
  bool operator()(const Scalar* const G_x_A, const Scalar* const G_y_A,
                  const Scalar* const yaw_A, const Scalar* const G_x_B,
                  const Scalar* const G_y_B, const Scalar* const yaw_B,
                  Scalar* residuals_ptr) const {
    const Eigen::Matrix<Scalar, 2, 1> G_p_A(*G_x_A, *G_y_A);
    const Scalar& G_yaw_A = *yaw_A;

    const Eigen::Matrix<Scalar, 2, 2> A_R_G =
        RotationMatrix(G_yaw_A).transpose();

    const Eigen::Matrix<Scalar, 2, 1> G_p_B(*G_x_B, *G_y_B);
    const Scalar& G_yaw_B = *yaw_B;

    Eigen::Map<Eigen::Matrix<Scalar, 3, 1> > residuals_map(residuals_ptr);

    Eigen::Matrix<Scalar, 3, 1> residuals;
    residuals_map.template head<2>() =
        A_R_G * (G_p_B - G_p_A) - A_p_B_.cast<Scalar>();
    residuals_map(2) = ceres::examples::pose_graph_2d::NormalizeAngle(
        (G_yaw_B - G_yaw_A) - static_cast<Scalar>(A_yaw_B_radians_));

    // Scale the residuals by the square root information matrix.
    residuals_map = sqrt_information_.template cast<Scalar>() * residuals_map;

    return true;
  }

  static ceres::CostFunction* Create(double A_x_B, double A_y_B,
                                     double A_yaw_B_radians,
                                     const Eigen::Matrix3d& sqrt_information) {
    return (new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1,
                                            1, 1>(new PoseGraph2dErrorTerm(
        A_x_B, A_y_B, A_yaw_B_radians, sqrt_information)));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // The position of B relative to A in the A frame.
  Eigen::Vector2d A_p_B_;
  // The orientation of frame B relative to frame A.
  double A_yaw_B_radians_;
  // The inverse square root of the measurement covariance matrix.
  Eigen::Matrix3d sqrt_information_;
};

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
