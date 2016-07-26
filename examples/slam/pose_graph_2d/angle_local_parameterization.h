#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_

#include "ceres/local_parameterization.h"
#include "normalize_angle.h"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

// Defines a local parameterization for updating the angle to be constrained in
// [-pi to pi).
class AngleLocalParameterization : public ceres::LocalParameterization {
 public:
  AngleLocalParameterization() {}

  virtual ~AngleLocalParameterization() {}

  virtual bool Plus(const double* theta_radians,
                    const double* delta_theta_radians,
                    double* theta_radians_plus_delta) const {
    CHECK(theta_radians != NULL);
    CHECK(delta_theta_radians != NULL);
    CHECK(theta_radians_plus_delta != NULL);

    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  // Computes the Jacobian of Plus(theta, delta_theta) w.r.t. delta_theta at
  // delta_theta = 0. As a first approximation to get an intuition for the
  // Jacobian, the theta is already within the valid range of -pi to pi and the
  // delta_theta is small so the function NormalizeAngle(theta + delta_theta) =
  // theta + delta_theta.
  //
  // Consequently, the Jacobian
  //   d NormalizeAngle(theta + delta_theta) |
  //    -----------------------------------  |                    =  1.
  //          d delta_theta                  | delta_theta = 0
  virtual bool ComputeJacobian(const double* /* theta_radians */,
                               double* jacobian) const {
    CHECK(jacobian != NULL);
    jacobian[0] = 1;

    return true;
  }

  virtual int GlobalSize() const { return 1; }

  virtual int LocalSize() const { return 1; }
};

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_
