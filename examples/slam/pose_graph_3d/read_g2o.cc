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

namespace ceres {
namespace examples {

bool ReadG2oFile(
    const std::string& filename,
    std::map<int, Pose3d, std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, Pose3d> > >* poses,
    std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >*
        constraints) {
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
    if (data_type == "VERTEX_SE3:QUAT") {
      int id;
      Pose3d pose;
      infile >> id >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >>
          pose.q.y() >> pose.q.z() >> pose.q.w();
      // Normalize the quaternion to account for precision loss due to
      // serialization.
      pose.q.normalize();

      // Ensure we don't have duplicate poses.
      if (poses->find(id) != poses->end()) {
        std::cerr << "Duplicate vertex with ID: " << id << '\n';
        return false;
      }
      (*poses)[id] = pose;
    } else if (data_type == "EDGE_SE3:QUAT") {
      Constraint3d constraint;

      // Read in the constraint data which is the x, y, z, q_x, q_y, q_z, q_w
      // and then the upper triangular part of the information matrix.
      Pose3d& t_be = constraint.t_be;
      infile >> constraint.id_begin >> constraint.id_end >> t_be.p.x() >>
          t_be.p.y() >> t_be.p.z() >> t_be.q.x() >> t_be.q.y() >> t_be.q.z() >>
          t_be.q.w();
      // Normalize the quaternion to account for precision loss due to
      // serialization.
      t_be.q.normalize();

      for (int i = 0; i < 6 && infile.good(); ++i) {
        for (int j = i; j < 6 && infile.good(); ++j) {
          infile >> constraint.information(i, j);
          if (i != j) {
            constraint.information(j, i) = constraint.information(i, j);
          }
        }
      }

      constraints->push_back(constraint);
    } else {
      std::cerr << "Unknown data type: " << data_type << '\n';
      return false;
    }

    // Clear any trailing whitespace from the line.
    infile >> std::ws;
  }

  return true;
}

}  // namespace examples
}  // namespace ceres
