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

#include "read_g2o.h"

#include <iostream>
#include <fstream>

#include "Eigen/Core"
#include "glog/logging.h"
#include "normalize_angle.h"

namespace ceres {
namespace examples {
namespace {
// Reads a single pose from the input and inserts it into the map. Returns false
// if there is a duplicate entry.
bool ReadVertex(std::ifstream* infile, std::map<int, Pose2d>* poses) {
  int id;
  Pose2d pose;
  *infile >> id >> pose.x >> pose.y >> pose.yaw_radians;
  // Normalize the angle between -pi to pi.
  pose.yaw_radians = NormalizeAngle(pose.yaw_radians);
  // Ensure we don't have duplicate poses.
  if (poses->find(id) != poses->end()) {
    std::cerr << "Duplicate vertex with ID: " << id << '\n';
    return false;
  }
  (*poses)[id] = pose;
  return true;
}

// Reads the contraints between two vertices in the pose graph
void ReadConstraint(std::ifstream* infile,
                    std::vector<Constraint2d>* constraints) {
  Constraint2d constraint;

  // Read in the constraint data which is the x, y, yaw_radians and then the
  // upper triangular part of the information matrix.
  *infile >> constraint.id_begin >> constraint.id_end >> constraint.x >>
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
}
}

bool ReadG2oFile(const std::string& filename, std::map<int, Pose2d>* poses,
                 std::vector<Constraint2d>* constraints) {
  CHECK(poses != NULL);
  CHECK(constraints != NULL);

  poses->clear();
  constraints->clear();

  std::ifstream infile(filename.c_str());
  if (!infile) {
    std::cerr << "Error reading the file: " << filename << '\n';
    return false;
  }

  std::string data_type;
  while (infile.good()) {
    // Read whether the type is a node or a constraint.
    infile >> data_type;
    if (data_type == "VERTEX_SE2") {
      if (!ReadVertex(&infile, poses)) {
        return false;
      }
    } else if (data_type == "EDGE_SE2") {
      ReadConstraint(&infile, constraints);
    } else {
      std::cerr << "Unknown data type: " << data_type << '\n';
      return false;
    }

    // Clear any trailing whitespace from the file.
    infile >> std::ws;
  }

  return true;
}

}  // namespace examples
}  // namespace ceres
