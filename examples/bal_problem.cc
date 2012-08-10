// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "bal_problem.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <glog/logging.h>
#include "ceres/random.h"
#include "ceres/rotation.h"
#include "Eigen/Core"

namespace ceres {
namespace examples {

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    LOG(FATAL) << "Invalid UW data file.";
  }
}

BALProblem::BALProblem(const std::string filename, bool use_quaternions) {
  FILE* fptr = fopen(filename.c_str(), "r");

  if (!fptr) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);

  VLOG(1) << "Header: " << num_cameras_
          << " " << num_points_
          << " " << num_observations_;

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
  parameters_ = new double[num_parameters_];

  for (int i = 0; i < num_observations_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
    }
  }

  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }

  fclose(fptr);

  use_quaternions_ = use_quaternions;
  if (use_quaternions) {
    // Switch the angle-axis rotations to quaternions.
    num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
    double* quaternion_parameters = new double[num_parameters_];
    double* original_cursor = parameters_;
    double* quaternion_cursor = quaternion_parameters;
    for (int i = 0; i < num_cameras_; ++i) {
      AngleAxisToQuaternion(original_cursor, quaternion_cursor);
      quaternion_cursor += 4;
      original_cursor += 3;
      for (int j = 4; j < 10; ++j) {
       *quaternion_cursor++ = *original_cursor++;
      }
    }
    // Copy the rest of the points.
    for (int i = 0; i < 3 * num_points_; ++i) {
      *quaternion_cursor++ = *original_cursor++;
    }
    // Swap in the quaternion parameters.
    delete []parameters_;
    parameters_ = quaternion_parameters;
  }
}

void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);

  double* points = mutable_points();
  if (point_sigma > 0) {
    for (int i = 0; i < 3 * num_points_; ++i) {
      points[i] += point_sigma * RandNormal();
    }
  }

  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = mutable_cameras() + camera_block_size() * i;

    // Perturb the location of the camera rather than the translation
    // vector. This is makes the perturbation physically more sensible.
    if (translation_sigma > 0.0) {
      double center[3];
      Eigen::VectorXd angle_axis(3);

      if (use_quaternions_) {
        angle_axis = Eigen::Map<Eigen::VectorXd>(camera, 3);
      } else {
        QuaternionToAngleAxis(camera, angle_axis.data());
      }
      angle_axis *= -1.0;

      // Camera center is c = -R't, the negative sign does not matter.
      AngleAxisRotatePoint(angle_axis.data(),
                           camera + camera_block_size() - 6,
                           center);

      // Perturb center.
      for (int j = 0; j < 3; ++j) {
        center[j] += translation_sigma * RandNormal();
      }

      // t = -R * (- R' t + perturbation)
      AngleAxisRotatePoint(angle_axis.data(),
                           center,
                           camera + camera_block_size() - 6);
    }

    // First three coordinates of the camera rotation are shared
    // between the angle-axis and the quaternion representations.
    if (rotation_sigma > 0.0) {
      camera[0] += rotation_sigma * RandNormal();
      camera[1] += rotation_sigma * RandNormal();
      camera[2] += rotation_sigma * RandNormal();

      if (use_quaternions_) {
        camera[3] += rotation_sigma * RandNormal();
        Eigen::Map<Eigen::VectorXd>(camera, 4).normalize();
      }
    }

  }
}

BALProblem::~BALProblem() {
  delete []point_index_;
  delete []camera_index_;
  delete []observations_;
  delete []parameters_;
}

}  // namespace examples
}  // namespace ceres
