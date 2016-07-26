#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_

#include "Eigen/Core"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

struct Pose2d {
  double x;
  double y;
  double yaw_radians;
};

struct Constraint2d {
  int id_begin;
  int id_end;

  double x;
  double y;
  double yaw_radians;

  // The information matrix for the measurement uncertainty. The order of the
  // entries are x, y, and yaw.
  Eigen::Matrix3d information;
};

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_
