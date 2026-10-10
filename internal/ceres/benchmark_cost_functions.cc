// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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

#include "ceres/benchmark_cost_functions.h"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/rotation.h"

namespace ceres::internal {

std::array<double, 3> NormalizedVector3(double x, double y, double z) {
  std::array v{x, y, z};
  Eigen::Map<Eigen::Vector3d>(v.data()).normalize();
  return v;
}

std::array<double, 4> NormalizedQuaternion(double qx,
                                           double qy,
                                           double qz,
                                           double qw) {
  std::array q{qx, qy, qz, qw};
  Eigen::Map<Eigen::Quaterniond>(q.data()).normalize();
  return q;
}

std::array<double, 7> NormalizedPose(double qx,
                                     double qy,
                                     double qz,
                                     double qw,
                                     double tx,
                                     double ty,
                                     double tz) {
  std::array pose{qx, qy, qz, qw, tx, ty, tz};
  Eigen::Map<Eigen::Quaterniond>(pose.data()).normalize();
  return pose;
}

std::array<double, 7> PoseFromBALCamera(const double* bal_camera) {
  double q_wxyz[4];
  AngleAxisToQuaternion(bal_camera, q_wxyz);
  return {q_wxyz[1],
          q_wxyz[2],
          q_wxyz[3],
          q_wxyz[0],
          bal_camera[3],
          bal_camera[4],
          bal_camera[5]};
}

Rat43CostFunctor::Rat43CostFunctor(const double x, const double y)
    : x_(x), y_(y) {}

Rat43CostFunctor::ParameterBlocks Rat43CostFunctor::MakeParameters() {
  return ParameterBlocks({1., 2., 3., 4.});
}

SnavelyReprojectionError::SnavelyReprojectionError(double observed_x,
                                                   double observed_y)
    : observed_x(observed_x), observed_y(observed_y) {}

SnavelyReprojectionError::ParameterBlocks
SnavelyReprojectionError::MakeParameters() {
  return ParameterBlocks({1., 2., 3., 4., 5., 6., 7., 8., 9.}, {1., 2., 3.});
}

SnavelyReprojectionError::ParameterBlocks
SnavelyReprojectionError::MakeParametersFromBAL(const double* bal_camera,
                                                const double* bal_point) {
  std::array<double, 9> camera{};
  std::array<double, 3> point{};
  std::copy_n(bal_camera, 9, camera.data());
  std::copy_n(bal_point, 3, point.data());
  return ParameterBlocks(camera, point);
}

RelativePoseError::RelativePoseError(Eigen::Quaterniond q_i_j,
                                     Eigen::Vector3d t_i_j)
    : meas_q_i_j_(std::move(q_i_j)), meas_t_i_j_(std::move(t_i_j)) {}

RelativePoseError::ParameterBlocks RelativePoseError::MakeParameters() {
  return ParameterBlocks(NormalizedPose(1., 2., 3., 4., 5., 6., 7.),
                         NormalizedPose(1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1));
}

Brdf::ParameterBlocks Brdf::MakeParameters() {
  return ParameterBlocks({1., 2., 3., 4., 5., 6., 7., 8., 9., 10.},
                         {0.1, 0.2, 0.3},
                         NormalizedVector3(-0.1, 0.5, 0.2),
                         NormalizedVector3(0.5, -0.2, 0.9),
                         NormalizedVector3(-0.3, 0.4, -0.3),
                         NormalizedVector3(0.5, 0.7, -0.1),
                         NormalizedVector3(0.2, -0.2, -0.2));
}

ColmapOpenCVReprojectionError::ColmapOpenCVReprojectionError(double u, double v)
    : obs_x(u), obs_y(v) {}

ColmapOpenCVReprojectionError::ParameterBlocks
ColmapOpenCVReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {0.5, -0.3, 4.2},
      NormalizedPose(0.01, 0.02, -0.01, 0.9997, 0.1, -0.2, 0.5),
      {1000.0, 1000.0, 640.0, 360.0, -0.05, 0.01, 1e-4, -1e-4});
}

ColmapOpenCVReprojectionError::ParameterBlocks
ColmapOpenCVReprojectionError::MakeParametersFromBAL(const double* bal_camera,
                                                     const double* bal_point) {
  // BAL cameras look down -Z with focal length bal_camera[6] and radial
  // distortion (bal_camera[7], bal_camera[8]); negate focal length so
  // projection matches COLMAP's +Z camera convention.
  std::array intrinsics{-bal_camera[6],
                        -bal_camera[6],
                        0.0,
                        0.0,
                        bal_camera[7],
                        bal_camera[8],
                        1e-4,
                        1e-4};
  return ParameterBlocks({bal_point[0], bal_point[1], bal_point[2]},
                         PoseFromBALCamera(bal_camera),
                         intrinsics);
}

std::unique_ptr<Manifold> ColmapOpenCVReprojectionError::MakeManifold(
    int block_index) {
  if (block_index == 1) {
    return std::make_unique<
        ProductManifold<EigenQuaternionManifold, EuclideanManifold<3>>>();
  }
  return nullptr;
}

std::unique_ptr<LossFunction>
ColmapOpenCVReprojectionError::MakeLossFunction() {
  return std::make_unique<HuberLoss>(1.0);
}

ColmapFullOpenCVReprojectionError::ColmapFullOpenCVReprojectionError(double u,
                                                                     double v)
    : obs_x(u), obs_y(v) {}

ColmapFullOpenCVReprojectionError::ParameterBlocks
ColmapFullOpenCVReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {0.5, -0.3, 4.2},
      NormalizedPose(0.01, 0.02, -0.01, 0.9997, 0.1, -0.2, 0.5),
      {1000.0,
       1000.0,
       640.0,
       360.0,
       -0.05,
       0.01,
       1e-4,
       -1e-4,
       0.002,
       0.01,
       -0.005,
       0.001});
}

ColmapRigReprojectionError::ColmapRigReprojectionError(double u, double v)
    : obs_x(u), obs_y(v) {}

ColmapRigReprojectionError::ParameterBlocks
ColmapRigReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {0.5, -0.3, 4.2},
      NormalizedPose(0.01, -0.01, 0.02, 0.9997, 0.05, 0.0, 0.0),
      NormalizedPose(0.02, 0.01, -0.02, 0.9995, 0.1, -0.2, 0.5),
      {1000.0, 1000.0, 640.0, 360.0, -0.05, 0.01, 1e-4, -1e-4});
}

ColmapSampsonError::ColmapSampsonError(const Eigen::Vector2d& p1,
                                       const Eigen::Vector2d& p2)
    : point1_(p1), point2_(p2) {}

ColmapSampsonError::ParameterBlocks ColmapSampsonError::MakeParameters() {
  return ParameterBlocks(
      NormalizedPose(0.01, 0.02, -0.01, 0.9997, 0.8, 0.4, 0.4472));
}

ColmapFisheyeReprojectionError::ColmapFisheyeReprojectionError(double u,
                                                               double v)
    : obs_x(u), obs_y(v) {}

ColmapFisheyeReprojectionError::ParameterBlocks
ColmapFisheyeReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {0.5, -0.3, 4.2},
      NormalizedPose(0.01, 0.02, -0.01, 0.9997, 0.1, -0.2, 0.5),
      {800.0, 800.0, 640.0, 360.0, -0.02, 0.005, -0.001, 0.0002});
}

ColmapSphericalReprojectionError::ColmapSphericalReprojectionError(double u,
                                                                   double v)
    : obs_x(u), obs_y(v) {}

ColmapSphericalReprojectionError::ParameterBlocks
ColmapSphericalReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {0.5, -0.3, 4.2},
      NormalizedPose(0.01, 0.02, -0.01, 0.9997, 0.1, -0.2, 0.5));
}

LibmvBrownReprojectionError::LibmvBrownReprojectionError(double u, double v)
    : obs_x(u), obs_y(v) {}

LibmvBrownReprojectionError::ParameterBlocks
LibmvBrownReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {1000.0, 640.0, 360.0, -0.05, 0.01, -0.002, 0.0005, 1e-4, -1e-4},
      {0.02, -0.03, 0.01, 0.1, -0.2, 0.5},
      {0.5, -0.3, 4.2});
}

LibmvBrownReprojectionError::ParameterBlocks
LibmvBrownReprojectionError::MakeParametersFromBAL(const double* bal_camera,
                                                   const double* bal_point) {
  std::array intrinsics{-bal_camera[6],
                        0.0,
                        0.0,
                        bal_camera[7],
                        bal_camera[8],
                        0.0,
                        0.0,
                        1e-4,
                        1e-4};
  std::array<double, 6> pose{};
  std::copy_n(bal_camera, 6, pose.data());
  return ParameterBlocks(
      intrinsics, pose, {bal_point[0], bal_point[1], bal_point[2]});
}

std::unique_ptr<Manifold> LibmvBrownReprojectionError::MakeManifold(
    int block_index) {
  if (block_index == 0) {
    return std::make_unique<SubsetManifold>(9, std::vector<int>{1, 2, 5, 6});
  }
  return nullptr;
}

LibmvNukeInvertReprojectionError::LibmvNukeInvertReprojectionError(double u,
                                                                   double v)
    : obs_x(u), obs_y(v) {}

LibmvNukeInvertReprojectionError::ParameterBlocks
LibmvNukeInvertReprojectionError::MakeParameters() {
  return ParameterBlocks(
      {1000.0, 640.0, 360.0, -0.05, 0.01, 1e-4, -1e-4, 1.0},
      {0.02, -0.03, 0.01, 0.1, -0.2, 0.5},
      {0.5, -0.3, 4.2});
}

PoseGraph3dError::PoseGraph3dError(const Eigen::Vector3d& p_ab,
                                   const Eigen::Quaterniond& q_ab,
                                   const Eigen::Matrix<double, 6, 6>& sqrt_info)
    : p_ab_measured_(p_ab),
      q_ab_measured_(q_ab),
      sqrt_information_(sqrt_info) {}

PoseGraph3dError::ParameterBlocks PoseGraph3dError::MakeParameters() {
  return ParameterBlocks({1.0, 2.0, 3.0},
                         NormalizedQuaternion(0.01, 0.02, -0.01, 0.9997),
                         {2.1, 3.2, 4.3},
                         NormalizedQuaternion(0.03, -0.01, 0.02, 0.9993));
}

}  // namespace ceres::internal
