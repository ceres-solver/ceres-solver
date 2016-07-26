#include "read_g2o.h"

#include <iostream>
#include <fstream>

#include "Eigen/Core"
#include "glog/logging.h"
#include "normalize_angle.h"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

bool ReadG2oFile(const std::string& filename, std::map<int, Pose2d>* poses,
                 std::vector<Constraint2d>* constraints) {
  CHECK(poses != NULL);
  CHECK(constraints != NULL);

  poses->clear();
  constraints->clear();

  std::ifstream infile(filename.c_str());
  if (!infile) {
    return false;
  }

  std::string data_type;
  while (infile.good()) {
    // Read whether the type is a node or a constraint.
    infile >> data_type;
    if (data_type == "VERTEX_SE2") {
      int id;
      Pose2d pose;
      infile >> id >> pose.x >> pose.y >> pose.yaw_radians;
      // Normalize the angle between -pi to pi.
      pose.yaw_radians = NormalizeAngle(pose.yaw_radians);
      // Ensure we don't have duplicate poses.
      if (poses->find(id) != poses->end()) {
        std::cerr << "Duplicate vertex with ID: " << id << '\n';
        return false;
      }
      (*poses)[id] = pose;
    } else if (data_type == "EDGE_SE2") {
      Constraint2d constraint;

      // Read in the constraint data which is the x, y, yaw_radians and then the
      // upper triangular part of the information matrix.
      infile >> constraint.id_begin >> constraint.id_end >> constraint.x >>
          constraint.y >> constraint.yaw_radians >>
          constraint.information(0, 0) >> constraint.information(0, 1) >>
          constraint.information(0, 2) >> constraint.information(1, 1) >>
          constraint.information(1, 2) >> constraint.information(2, 2);

      // Set the lower triangular part of the information matrix.
      constraint.information(1, 0) = constraint.information(0, 1);
      constraint.information(2, 0) = constraint.information(0, 2);
      constraint.information(2, 1) = constraint.information(1, 2);

      // Normalize the angle between -pi to pi.
      constraint.yaw_radians = NormalizeAngle(constraint.yaw_radians);

      constraints->push_back(constraint);
    } else {
      std::cerr << "Unknown data type: " << data_type << '\n';
      return false;
    }

    // Clear any trailing whitespace from the file.
    infile >> std::ws;
  }

  return true;
}

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres
