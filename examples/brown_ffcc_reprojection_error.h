#pragma once

#include "ceres/autodiff_cost_function.h"
#include "ceres/rotation.h"

namespace ceres {
namespace examples {

// Templated pinhole camera model for used with Ceres with brown distortion.
// The camera pose is parameterized using 6 parameters: 3 for rotation, 3 for
// translation, and the camera intrinsics is parametrized using 4 parameters: 2
// for focal length and 2 for principal point.
struct BrownffccReprojectionError {
  BrownffccReprojectionError(const Eigen::Vector2d& observed)
      : observed(observed) {}

  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[3,4,5] are the translation.
    const T tp[3] = {
        point[0] - camera[3], point[1] - camera[4], point[2] - camera[5]};

    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    AngleAxisRotatePoint(camera, tp, p);

    // Compute the center of distortion.
    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second, fourth, and eighth order radial distortion.
    const T& k1 = intrinsics[4];
    const T& k2 = intrinsics[5];
    const T& k3 = intrinsics[6];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));

    // Compute final projected point position.
    const T& focal_x = intrinsics[0];
    const T& focal_y = intrinsics[1];
    const T& principalpoint_x = intrinsics[2];
    const T& principalpoint_y = intrinsics[3];
    const T predicted_x = focal_x * distortion * xp + principalpoint_x;
    const T predicted_y = focal_y * distortion * yp + principalpoint_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed.x();
    residuals[1] = predicted_y - observed.y();

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector2d& observed) {
    return (
        new ceres::AutoDiffCostFunction<BrownffccReprojectionError, 2, 4, 6, 3>(
            new BrownffccReprojectionError(observed)));
  }

  Eigen::Vector2d observed;
};

struct BrownffccReprojectionErrorWithQuaternions {
  // (u, v): the position of the observation with respect to the image
  // center point.
  BrownffccReprojectionErrorWithQuaternions(const Eigen::Vector2d& observed)
      : observed(observed) {}

  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[4,5,6] are the translation.
    const T tp[3] = {
        point[0] - camera[4], point[1] - camera[5], point[2] - camera[6]};

    // camera[0,1,2,3] is are the rotation of the camera as a quaternion.
    //
    // We use QuaternionRotatePoint as it does not assume that the
    // quaternion is normalized, since one of the ways to run the
    // bundle adjuster is to let Ceres optimize all 4 quaternion
    // parameters without using a Quaternion manifold.
    T p[3];
    QuaternionRotatePoint(camera, tp, p);

    // Compute the center of distortion.
    const T xp = p[0] / p[2];
    const T yp = p[1] / p[2];

    // Apply second, fourth, and eighth order radial distortion.
    const T& k1 = intrinsics[4];
    const T& k2 = intrinsics[5];
    const T& k3 = intrinsics[6];
    const T r2 = xp * xp + yp * yp;
    const T distortion = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));

    // Compute final projected point position.
    const T& focal_x = intrinsics[0];
    const T& focal_y = intrinsics[1];
    const T& principalpoint_x = intrinsics[2];
    const T& principalpoint_y = intrinsics[3];
    const T predicted_x = focal_x * distortion * xp + principalpoint_x;
    const T predicted_y = focal_y * distortion * yp + principalpoint_y;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed.x();
    residuals[1] = predicted_y - observed.y();

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector2d& observed) {
    return (new ceres::AutoDiffCostFunction<
            BrownffccReprojectionErrorWithQuaternions,
            2,
            4,
            7,
            3>(new BrownffccReprojectionErrorWithQuaternions(observed)));
  }

  Eigen::Vector2d observed;
};

}  // namespace examples
}  // namespace ceres
