#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_READ_G2O_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_READ_G2O_H_

#include <string>
#include <map>
#include <vector>

#include "types.h"

namespace ceres {
namespace examples {
namespace pose_graph_2d {

// Reads a file in the g2o filename format that describes a 2D pose graph
// problem. The g2o format consists of two entries, vertices and constraints. A
// vertex is defined as follows:
//
// VERTEX_SE2 ID x_meters y_meters yaw_radians
//
// A constraint is defined as follows:
//
// EDGE_SE2 ID_A ID_B A_x_B A_y_B A_yaw_B I_11 I_12 I_13 I_22 I_23 I_33
//
// where I_ij is the (i, j)-th entry of the information matrix for the
// measurement.
bool ReadG2oFile(const std::string& filename,
                 std::map<int, Pose2d>* poses,
                 std::vector<Constraint2d>* constraints);

}  // namespace pose_graph_2d
}  // namespace examples
}  // namespace ceres

#endif  // CERES_EXAMPLES_POSE_GRAPH_2D_READ_G2O_H_
