#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_

#include <cmath>

#include "ceres/ceres.h"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

// Normalizes the angle in radians between [-pi and pi).
template <typename Scalar>
inline Scalar NormalizeAngle(const Scalar& angle_radians) {
  // Use ceres::floor because it is specialized for double and Jet types.
  Scalar two_pi(2.0 * M_PI);
  return angle_radians -
      two_pi * ceres::floor((angle_radians + Scalar(M_PI)) / two_pi);
}

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_
