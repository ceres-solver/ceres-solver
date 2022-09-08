// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "bal_problem.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "brown_ffcc_reprojection_error.h"
#include "brown_ffcckkk_reprojection_error.h"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "random.h"
#include "snavely_reprojection_error.h"

namespace ceres::examples {
namespace {
using VectorRef = Eigen::Map<Eigen::VectorXd>;
using ConstVectorRef = Eigen::Map<const Eigen::VectorXd>;

template <typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    LOG(FATAL) << "Invalid UW data file.";
  }
}

void PerturbPoint3(const double sigma, double* point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += RandNormal() * sigma;
  }
}

double Median(std::vector<double>* data) {
  int n = data->size();
  auto mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

}  // namespace

BALProblem::BALProblem(const std::string& filename,
                       bool use_quaternions,
                       bool general_format)
    : use_quaternions_(use_quaternions) {
  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == nullptr) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  // This will die horribly on invalid files. Them's the breaks.
  if (general_format) {
    int _intrinsics_type;
    FscanfOrDie(fptr, "%d", &_intrinsics_type);
    FscanfOrDie(fptr, "%d", &num_intrinsics_);
    intrinsics_type_ = static_cast<IntrinsicsType>(_intrinsics_type);
  }
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);
  if (!general_format) {
    intrinsics_type_ = IntrinsicsType::SNAVELY;
    num_intrinsics_ = num_cameras_;
  }
  assert(num_cameras_ == num_intrinsics_ || num_intrinsics_ == 1);

  VLOG(1) << "Header: " << num_intrinsics_ << " intrinsics, " << num_cameras_
          << " cameras, " << num_points_ << " points, " << num_observations_
          << " observations";

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  intrinsics_ = new double[intrinsic_block_size() * num_intrinsics_];
  cameras_ = new double[camera_block_size() * num_cameras_];
  points_ = new double[point_block_size() * num_points_];

  for (int i = 0; i < num_observations_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
    }
  }

  if (intrinsics_type_ == IntrinsicsType::SNAVELY) {
    // intrinsics and cameras are interleaved
    assert(num_cameras_ == num_intrinsics_);
    for (int i = 0; i < num_cameras_; ++i) {
      for (int j = 0; j < 6; ++j) {
        FscanfOrDie(fptr, "%lf", cameras_ + i * 6 + j);
      }
      for (int j = 0; j < 3; ++j) {
        FscanfOrDie(fptr, "%lf", intrinsics_ + i * 3 + j);
      }
    }
  } else {
    // intrinsics and cameras are separate
    const int ibs = intrinsic_block_size();
    for (int i = 0; i < num_intrinsics_; ++i) {
      for (int j = 0; j < ibs; ++j) {
        FscanfOrDie(fptr, "%lf", intrinsics_ + i * ibs + j);
      }
    }
    for (int i = 0; i < num_cameras_; ++i) {
      for (int j = 0; j < 6; ++j) {
        FscanfOrDie(fptr, "%lf", cameras_ + i * 6 + j);
      }
    }
  }
  for (int i = 0; i < num_points_; ++i) {
    for (int j = 0; j < 3; ++j) {
      FscanfOrDie(fptr, "%lf", points_ + i * 3 + j);
    }
  }

  fclose(fptr);

  if (use_quaternions) {
    // Switch the angle-axis rotations to quaternions.
    double* original_cursor = cameras_ + 6 * num_cameras_;
    double* quaternion_cursor = cameras_ + 7 * num_cameras_;
    while (original_cursor > cameras_) {
      original_cursor -= 6;
      quaternion_cursor -= 7;
      const Eigen::Vector3d angleaxis = ConstVectorRef(original_cursor + 0, 3);
      const Eigen::Vector3d position = ConstVectorRef(original_cursor + 3, 3);
      AngleAxisToQuaternion(angleaxis.data(), quaternion_cursor);
      VectorRef(quaternion_cursor + 4, 3) = position;
    }
  }
}

// This function writes the problem to a file in the same format that
// is read by the constructor.
void BALProblem::WriteToFile(const std::string& filename) const {
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == nullptr) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  if (intrinsics_type_ == IntrinsicsType::SNAVELY) {
    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);
  } else {
    fprintf(fptr,
            "%d %d %d %d %d\n",
            static_cast<int>(intrinsics_type_),
            num_intrinsics_,
            num_cameras_,
            num_points_,
            num_observations_);
  }

  for (int i = 0; i < num_observations_; ++i) {
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", observations_[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if (intrinsics_type_ == IntrinsicsType::SNAVELY) {
    for (int i = 0; i < num_cameras(); ++i) {
      const double* intrinsic = this->intrinsic(i);
      for (int j = 0; j < intrinsic_block_size(); ++j) {
        fprintf(fptr, "%.16g\n", intrinsic[j]);
      }
      double* angleaxis;
      double _angleaxis[6];
      const double* camera = this->camera(i);
      if (use_quaternions_) {
        // Output in angle-axis format.
        QuaternionToAngleAxis(camera, _angleaxis);
        memcpy(_angleaxis + 3, camera + 4, 3 * sizeof(double));
        angleaxis = _angleaxis;
      } else {
        angleaxis = cameras_;
      }
      for (int j = 0; j < 6; ++j) {
        fprintf(fptr, "%.16g\n", angleaxis[j]);
      }
    }
  } else {
    for (int i = 0; i < num_intrinsics(); ++i) {
      const double* intrinsic = this->intrinsic(i);
      for (int j = 0; j < intrinsic_block_size(); ++j) {
        fprintf(fptr, "%.16g\n", intrinsic[j]);
      }
    }
    for (int i = 0; i < num_cameras(); ++i) {
      double* angleaxis;
      double _angleaxis[6];
      const double* camera = this->camera(i);
      if (use_quaternions_) {
        // Output in angle-axis format.
        QuaternionToAngleAxis(camera, _angleaxis);
        memcpy(_angleaxis + 3, camera + 4, 3 * sizeof(double));
        angleaxis = _angleaxis;
      } else {
        angleaxis = cameras_;
      }
      for (int j = 0; j < 6; ++j) {
        fprintf(fptr, "%.16g\n", angleaxis[j]);
      }
    }
  }
  for (int i = 0; i < num_points(); ++i) {
    const double* point = this->point(i);
    for (int j = 0; j < point_block_size(); ++j) {
      fprintf(fptr, "%.16g\n", point[j]);
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BALProblem::WriteToPLYFile(const std::string& filename) const {
  std::ofstream of(filename.c_str());

  of << "ply" << '\n'
     << "format ascii 1.0" << '\n'
     << "element vertex " << num_cameras_ + num_points_ << '\n'
     << "property float x" << '\n'
     << "property float y" << '\n'
     << "property float z" << '\n'
     << "property uchar red" << '\n'
     << "property uchar green" << '\n'
     << "property uchar blue" << '\n'
     << "end_header" << std::endl;

  // Export extrinsic data (i.e. camera centers) as green points.
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras(); ++i) {
    const double* camera = this->camera(i);
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    of << center[0] << ' ' << center[1] << ' ' << center[2] << " 0 255 0\n";
  }

  // Export the structure (i.e. 3D Points) as white points.
  for (int i = 0; i < num_points(); ++i) {
    const double* point = this->point(i);
    for (int j = 0; j < point_block_size(); ++j) {
      of << point[j] << ' ';
    }
    of << "255 255 255\n";
  }
  of.close();
}

void BALProblem::CameraToAngleAxisAndCenter(const double* camera,
                                            double* angle_axis,
                                            double* center) const {
  VectorRef angle_axis_ref(angle_axis, 3);
  if (use_quaternions_) {
    QuaternionToAngleAxis(camera, angle_axis);
  } else {
    angle_axis_ref = ConstVectorRef(camera, 3);
  }

  switch (intrinsics_type_) {
    case IntrinsicsType::SNAVELY: {
      // C = -R't
      const Eigen::VectorXd inverse_rotation = -angle_axis_ref;
      AngleAxisRotatePoint(
          inverse_rotation.data(), camera + (camera_block_size() - 3), center);
      VectorRef(center, 3) *= -1.0;
      break;
    }
    case IntrinsicsType::BROWN_FFCC:
    case IntrinsicsType::BROWN_FFCCKKK: {
      VectorRef(center, 3) =
          ConstVectorRef(camera + (camera_block_size() - 3), 3);
      break;
    }
    default:
      LOG(FATAL) << "Unknown intrinsics type";
  }
}

void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const {
  ConstVectorRef angle_axis_ref(angle_axis, 3);
  if (use_quaternions_) {
    AngleAxisToQuaternion(angle_axis, camera);
  } else {
    VectorRef(camera, 3) = angle_axis_ref;
  }

  switch (intrinsics_type_) {
    case IntrinsicsType::SNAVELY: {
      // t = -R * C
      AngleAxisRotatePoint(
          angle_axis, center, camera + (camera_block_size() - 3));
      VectorRef(camera + (camera_block_size() - 3), 3) *= -1.0;
      break;
    }
    case IntrinsicsType::BROWN_FFCC:
    case IntrinsicsType::BROWN_FFCCKKK: {
      VectorRef(camera + (camera_block_size() - 3), 3) =
          ConstVectorRef(center, 3);
      break;
    }
    default:
      LOG(FATAL) << "Unknown intrinsics type";
  }
}

void BALProblem::Normalize(const double median_scale) {
  if (median_scale <= 0) {
    return;
  }

  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < num_points_; ++j) {
      tmp[j] = points_[3 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points_ + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100.
  const double scale = median_scale / median_absolute_deviation;

  VLOG(2) << "median: " << median.transpose();
  VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
  VLOG(2) << "scale: " << scale;

  // X = scale * (X - median)
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points_ + 3 * i, 3);
    point = scale * (point - median);
  }

  Eigen::Vector3d angle_axis;
  Eigen::Vector3d center;
  for (int i = 0; i < num_cameras_; ++i) {
    double* const camera = this->mutable_camera(i);
    CameraToAngleAxisAndCenter(camera, angle_axis.data(), center.data());
    center = scale * (center - median);
    AngleAxisAndCenterToCamera(angle_axis.data(), center.data(), camera);
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
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(point_sigma, points + 3 * i);
    }
  }

  if (rotation_sigma > 0.0 || translation_sigma > 0.0) {
    for (int i = 0; i < num_cameras_; ++i) {
      double* const camera = this->mutable_camera(i);

      double angle_axis[3];
      double center[3];
      // Perturb in the rotation of the camera in the angle-axis
      // representation.
      CameraToAngleAxisAndCenter(camera, angle_axis, center);
      if (rotation_sigma > 0.0) {
        PerturbPoint3(rotation_sigma, angle_axis);
      }
      AngleAxisAndCenterToCamera(angle_axis, center, camera);

      if (translation_sigma > 0.0) {
        PerturbPoint3(translation_sigma, camera + camera_block_size() - 3);
      }
    }
  }
}

CostFunction* BALProblem::CreateReprojectionErrorCostFunction(int idx) const {
  // Each Residual block takes a point, a camera and an intrinsics as input and
  // outputs a 2 dimensional residual.
  const Eigen::Map<const Eigen::Vector2d> observation(observations_ + 2 * idx);
  switch (intrinsics_type_) {
    case IntrinsicsType::SNAVELY:
      return use_quaternions_
                 ? SnavelyReprojectionErrorWithQuaternions::Create(observation)
                 : SnavelyReprojectionError::Create(observation);
    case IntrinsicsType::BROWN_FFCC:
      return use_quaternions_
                 ? BrownffccReprojectionErrorWithQuaternions::Create(
                       observation)
                 : BrownffccReprojectionError::Create(observation);
    case IntrinsicsType::BROWN_FFCCKKK:
      return use_quaternions_
                 ? BrownffcckkkReprojectionErrorWithQuaternions::Create(
                       observation)
                 : BrownffcckkkReprojectionError::Create(observation);
    default:
      LOG(FATAL) << "Unknown intrinsics type";
      return nullptr;
  }
}

std::string BALProblem::PrintIntrinsics(int idx) const {
  assert(idx < num_cameras_);
  const double* intrinsic = this->intrinsic(idx);
  std::stringstream ss;
  const int ibs = intrinsic_block_size();
  for (int i = 0; i < ibs; ++i) {
    ss << intrinsic[i];
    if (i != ibs - 1) ss << ' ';
  }
  return ss.str();
}

BALProblem::~BALProblem() {
  delete[] point_index_;
  delete[] camera_index_;
  delete[] observations_;
  delete[] intrinsics_;
  delete[] cameras_;
  delete[] points_;
}

}  // namespace ceres::examples
