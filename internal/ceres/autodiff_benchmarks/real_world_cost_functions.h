// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
// Copyright (c) 2016, ETH Zurich and UNC Chapel Hill.
// Copyright (c) 2016-2026, The COLMAP Contributors.
// Copyright (c) 2011-2013, libmv authors.
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
// * Neither the name of Google Inc., ETH Zurich, UNC Chapel Hill, nor the names
//   of its contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
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
// SPDX-License-Identifier: BSD-3-Clause AND MIT

#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_REAL_WORLD_COST_FUNCTIONS_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_REAL_WORLD_COST_FUNCTIONS_H_

#include <cmath>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/rotation.h"

namespace ceres {

// 1. COLMAP ReprojErrorCostFunctor<OpenCVCameraModel>: <2, 3, 7, 8>
// point3D_in_world (3), cam_from_world [qx, qy, qz, qw, tx, ty, tz] (7),
// camera_params [fx, fy, cx, cy, k1, k2, p1, p2] (8).
struct ColmapOpenCVReprojectionError {
  ColmapOpenCVReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const point3D_in_world,
                         const T* const cam_from_world,
                         const T* const camera_params,
                         T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q(cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(point3D_in_world);
    const Eigen::Matrix<T, 3, 1> p_c = q * p_w + t;

    const T f1 = camera_params[0];
    const T f2 = camera_params[1];
    const T c1 = camera_params[2];
    const T c2 = camera_params[3];
    const T k1 = camera_params[4];
    const T k2 = camera_params[5];
    const T p1 = camera_params[6];
    const T p2 = camera_params[7];

    const T u = p_c[0] / p_c[2];
    const T v = p_c[1] / p_c[2];
    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    const T du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    const T dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);

    residuals[0] = f1 * (u + du) + c1 - T(obs_x);
    residuals[1] = f2 * (v + dv) + c2 - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 2. COLMAP ReprojErrorCostFunctor<FullOpenCVCameraModel>: <2, 3, 7, 12>
// Rational radial + tangential distortion:
// camera_params [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6] (12).
struct ColmapFullOpenCVReprojectionError {
  ColmapFullOpenCVReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const point3D_in_world,
                         const T* const cam_from_world,
                         const T* const camera_params,
                         T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q(cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(point3D_in_world);
    const Eigen::Matrix<T, 3, 1> p_c = q * p_w + t;

    const T f1 = camera_params[0];
    const T f2 = camera_params[1];
    const T c1 = camera_params[2];
    const T c2 = camera_params[3];
    const T k1 = camera_params[4];
    const T k2 = camera_params[5];
    const T p1 = camera_params[6];
    const T p2 = camera_params[7];
    const T k3 = camera_params[8];
    const T k4 = camera_params[9];
    const T k5 = camera_params[10];
    const T k6 = camera_params[11];

    const T u = p_c[0] / p_c[2];
    const T v = p_c[1] / p_c[2];
    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;
    const T radial = (T(1) + k1 * r2 + k2 * r4 + k3 * r6) /
                     (T(1) + k4 * r2 + k5 * r4 + k6 * r6);
    const T x = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    const T y = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);

    residuals[0] = f1 * x + c1 - T(obs_x);
    residuals[1] = f2 * y + c2 - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 3. COLMAP RigReprojErrorCostFunctor<OpenCVCameraModel>: <2, 3, 7, 7, 8>
// point3D_in_world (3), cam_from_rig (7), rig_from_world (7), camera_params
// (8).
struct ColmapRigReprojectionError {
  ColmapRigReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const point3D_in_world,
                         const T* const cam_from_rig,
                         const T* const rig_from_world,
                         const T* const camera_params,
                         T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q_cr(cam_from_rig);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cr(cam_from_rig + 4);
    const Eigen::Map<const Eigen::Quaternion<T>> q_rw(rig_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_rw(rig_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(point3D_in_world);

    const Eigen::Matrix<T, 3, 1> p_c = q_cr * (q_rw * p_w + t_rw) + t_cr;

    const T f1 = camera_params[0];
    const T f2 = camera_params[1];
    const T c1 = camera_params[2];
    const T c2 = camera_params[3];
    const T k1 = camera_params[4];
    const T k2 = camera_params[5];
    const T p1 = camera_params[6];
    const T p2 = camera_params[7];

    const T u = p_c[0] / p_c[2];
    const T v = p_c[1] / p_c[2];
    const T u2 = u * u;
    const T uv = u * v;
    const T v2 = v * v;
    const T r2 = u2 + v2;
    const T radial = k1 * r2 + k2 * r2 * r2;
    const T du = u * radial + T(2) * p1 * uv + p2 * (r2 + T(2) * u2);
    const T dv = v * radial + T(2) * p2 * uv + p1 * (r2 + T(2) * v2);

    residuals[0] = f1 * (u + du) + c1 - T(obs_x);
    residuals[1] = f2 * (v + dv) + c2 - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 4. COLMAP SampsonErrorCostFunctor: <1, 7>
// Essential matrix E = [t]_x R(q) and signed Sampson epipolar error.
struct ColmapSampsonError {
  ColmapSampsonError(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
      : point1_(p1), point2_(p2) {}

  template <typename T>
  inline bool operator()(const T* const cam2_from_cam1, T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        Eigen::Map<const Eigen::Quaternion<T>>(cam2_from_cam1)
            .toRotationMatrix();
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1[6], cam2_from_cam1[5], cam2_from_cam1[6], T(0),
        -cam2_from_cam1[4], -cam2_from_cam1[5], cam2_from_cam1[4], T(0);
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    const Eigen::Matrix<T, 3, 1> p1(T(point1_.x()), T(point1_.y()), T(1));
    const Eigen::Matrix<T, 3, 1> p2(T(point2_.x()), T(point2_.y()), T(1));
    const Eigen::Matrix<T, 3, 1> epipolar_line1 = E * p1;
    const T num = p2.dot(epipolar_line1);
    const Eigen::Matrix<T, 4, 1> denom(p2.dot(E.col(0)),
                                       p2.dot(E.col(1)),
                                       epipolar_line1.x(),
                                       epipolar_line1.y());
    residuals[0] = num / denom.norm();
    return true;
  }

  Eigen::Vector2d point1_;
  Eigen::Vector2d point2_;
};

// 5. COLMAP ReprojErrorCostFunctor<OpenCVFisheyeCameraModel>: <2, 3, 7, 8>
// Kannala-Brandt equidistant fisheye camera model:
// point3D_in_world (3), cam_from_world (7),
// camera_params [fx, fy, cx, cy, k1, k2, k3, k4] (8).
struct ColmapFisheyeReprojectionError {
  ColmapFisheyeReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const point3D_in_world,
                         const T* const cam_from_world,
                         const T* const camera_params,
                         T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q(cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(point3D_in_world);
    const Eigen::Matrix<T, 3, 1> p_c = q * p_w + t;

    const T f1 = camera_params[0];
    const T f2 = camera_params[1];
    const T c1 = camera_params[2];
    const T c2 = camera_params[3];
    const T k1 = camera_params[4];
    const T k2 = camera_params[5];
    const T k3 = camera_params[6];
    const T k4 = camera_params[7];

    const T u = p_c[0] / p_c[2];
    const T v = p_c[1] / p_c[2];
    const T r = hypot(u, v);
    T x = u;
    T y = v;
    if (r > T(1e-8)) {
      const T theta = atan(r);
      const T theta2 = theta * theta;
      const T theta4 = theta2 * theta2;
      const T theta6 = theta4 * theta2;
      const T theta8 = theta4 * theta4;
      const T thetad = theta * (T(1) + k1 * theta2 + k2 * theta4 + k3 * theta6 +
                                k4 * theta8);
      const T scale = thetad / r;
      x *= scale;
      y *= scale;
    }

    residuals[0] = f1 * x + c1 - T(obs_x);
    residuals[1] = f2 * y + c2 - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 6. COLMAP ReprojErrorCostFunctor<EquirectangularCameraModel>: <2, 3, 7>
// 360x180 spherical equirectangular projection using atan2 and asin.
struct ColmapSphericalReprojectionError {
  ColmapSphericalReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const point3D_in_world,
                         const T* const cam_from_world,
                         T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q(cam_from_world);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(cam_from_world + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_w(point3D_in_world);
    const Eigen::Matrix<T, 3, 1> p_c = q * p_w + t;

    const T norm = p_c.norm();
    const T lon = atan2(p_c[0], p_c[2]);
    const T lat = asin(p_c[1] / norm);

    constexpr double kWidth = 1920.0;
    constexpr double kHeight = 960.0;
    constexpr double kInvTwoPi = 1.0 / (2.0 * M_PI);
    constexpr double kInvPi = 1.0 / M_PI;
    residuals[0] = T(kWidth) * (T(0.5) + lon * T(kInvTwoPi)) - T(obs_x);
    residuals[1] = T(kHeight) * (T(0.5) + lat * T(kInvPi)) - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 7. Blender libmv Brown-Conrady ReprojectionErrorApplyIntrinsics: <2, 9, 6, 3>
// intrinsics [f, px, py, k1, k2, k3, k4, p1, p2] (9), R_t [rx, ry, rz, tx, ty,
// tz] (6), X (3).
struct LibmvBrownReprojectionError {
  LibmvBrownReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const intrinsics,
                         const T* const R_t,
                         const T* const X,
                         T* residuals) const {
    T p[3];
    ceres::AngleAxisRotatePoint(R_t, X, p);
    p[0] += R_t[3];
    p[1] += R_t[4];
    p[2] += R_t[5];

    const T x = p[0] / p[2];
    const T y = p[1] / p[2];

    const T focal = intrinsics[0];
    const T px = intrinsics[1];
    const T py = intrinsics[2];
    const T k1 = intrinsics[3];
    const T k2 = intrinsics[4];
    const T k3 = intrinsics[5];
    const T k4 = intrinsics[6];
    const T p1 = intrinsics[7];
    const T p2 = intrinsics[8];

    const T x2 = x * x;
    const T y2 = y * y;
    const T xy2 = T(2) * x * y;
    const T r2 = x2 + y2;
    const T r_coeff = T(1) + (((k4 * r2 + k3) * r2 + k2) * r2 + k1) * r2;
    const T tx = p1 * (r2 + T(2) * x2) + p2 * xy2;
    const T ty = p2 * (r2 + T(2) * y2) + p1 * xy2;

    residuals[0] = focal * (x * r_coeff + tx) + px - T(obs_x);
    residuals[1] = focal * (y * r_coeff + ty) + py - T(obs_y);
    return true;
  }

  double obs_x;
  double obs_y;
};

// 8. Blender libmv Nuke ReprojectionErrorInvertIntrinsics: <2, 8, 6, 3>
// intrinsics [f, px, py, k1, k2, p1, p2, aspect] (8), R_t (6), X (3).
struct LibmvNukeInvertReprojectionError {
  LibmvNukeInvertReprojectionError(double u, double v) : obs_x(u), obs_y(v) {}

  template <typename T>
  inline bool operator()(const T* const intrinsics,
                         const T* const R_t,
                         const T* const X,
                         T* residuals) const {
    const T focal = intrinsics[0];
    const T px = intrinsics[1];
    const T py = intrinsics[2];
    const T k1 = intrinsics[3];
    const T k2 = intrinsics[4];
    const T p1 = intrinsics[5];
    const T p2 = intrinsics[6];

    T p[3];
    ceres::AngleAxisRotatePoint(R_t, X, p);
    p[0] += R_t[3];
    p[1] += R_t[4];
    p[2] += R_t[5];

    const T xn = p[0] / p[2];
    const T yn = p[1] / p[2];
    const T predicted_x = focal * xn + px;
    const T predicted_y = focal * yn + py;

    constexpr double kHalfMaxImageSize = 960.0;
    const T xd = (T(obs_x) - px) / T(kHalfMaxImageSize);
    const T yd = (T(obs_y) - py) / T(kHalfMaxImageSize);
    const T xd2 = xd * xd;
    const T yd2 = yd * yd;
    const T rd2 = xd2 + yd2;
    const T rd4 = rd2 * rd2;
    const T xu = xd / (T(1) + k1 * rd2 + k2 * rd4 + p1 * yd2);
    const T yu = yd / (T(1) + k1 * rd2 + k2 * rd4 + p2 * xd2);

    const T obs_undistorted_x = xu * T(kHalfMaxImageSize) + px;
    const T obs_undistorted_y = yu * T(kHalfMaxImageSize) + py;

    residuals[0] = predicted_x - obs_undistorted_x;
    residuals[1] = predicted_y - obs_undistorted_y;
    return true;
  }

  double obs_x;
  double obs_y;
};

// 9. 3D Pose Graph SLAM Error Term (PoseGraph3dErrorTerm): <6, 3, 4, 3, 4>
struct PoseGraph3dError {
  PoseGraph3dError(const Eigen::Vector3d& p_ab,
                   const Eigen::Quaterniond& q_ab,
                   const Eigen::Matrix<double, 6, 6>& sqrt_info)
      : p_ab_measured_(p_ab),
        q_ab_measured_(q_ab),
        sqrt_information_(sqrt_info) {}

  template <typename T>
  inline bool operator()(const T* const p_a_ptr,
                         const T* const q_a_ptr,
                         const T* const p_b_ptr,
                         const T* const q_b_ptr,
                         T* residuals_ptr) const {
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    const Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    const Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    const Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    const Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;
    const Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);
    const Eigen::Quaternion<T> delta_q =
        q_ab_measured_.template cast<T>() * q_ab_estimated.conjugate();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - p_ab_measured_.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }

  Eigen::Vector3d p_ab_measured_;
  Eigen::Quaterniond q_ab_measured_;
  Eigen::Matrix<double, 6, 6> sqrt_information_;
};

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_REAL_WORLD_COST_FUNCTIONS_H_
