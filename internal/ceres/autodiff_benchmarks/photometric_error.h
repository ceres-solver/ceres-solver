// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
// Author: nikolaus@nikolaus-demmel.de (Nikolaus Demmel)
//
//
#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARK_PHOTOMETRIC_ERROR_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARK_PHOTOMETRIC_ERROR_H_

#include <Eigen/Dense>

#include "ceres/codegen/codegen_cost_function.h"
#include "ceres/cubic_interpolation.h"

namespace ceres {

// Photometric residual that computes the intensity difference for a patch
// between host and target frame. The point is parameterized with inverse
// distance relative to the host frame. The relative pose between host and
// target frame is computed from their respective absolute poses. Camera
// intrinsics are assumed constant, and thus host frame points are passed as
// (unprojected) bearing vectors.
template <int PATCH_SIZE_ = 8>
struct PhotometricError
    : public ceres::CodegenCostFunction<PATCH_SIZE_, 7, 7, 1> {
  static constexpr int PATCH_SIZE = PATCH_SIZE_;
  static constexpr int POSE_SIZE = 7;
  static constexpr int POINT_SIZE = 1;

  using Grid = Grid2D<uint8_t, 1>;
  using Interpolator = BiCubicInterpolator<Grid>;
  using Intrinsics = Eigen::Array<double, 6, 1>;

  template <typename T>
  using Patch = Eigen::Matrix<T, PATCH_SIZE, 1>;

  template <typename T>
  using PatchVectors = Eigen::Matrix<T, 3, PATCH_SIZE>;

  PhotometricError(const Patch<double>& intensities_host,
                   const PatchVectors<double>& bearings_host,
                   const Interpolator& image_target,
                   const Intrinsics& intrinsics)
      : intensities_host_(intensities_host),
        bearings_host_(bearings_host),
        image_target_(image_target),
        intrinsics_(intrinsics) {}

  template <typename T>
  bool project(Eigen::Matrix<T, 2, 1>& proj,
               const Eigen::Matrix<T, 3, 1>& p) const {
    // projection for extended unified camera model
    // see: https://arxiv.org/pdf/1807.08957.pdf

    const double& fx = intrinsics_[0];
    const double& fy = intrinsics_[1];
    const double& cx = intrinsics_[2];
    const double& cy = intrinsics_[3];
    const double& alpha = intrinsics_[4];
    const double& beta = intrinsics_[5];

    const auto rho2 = beta * (p.x() * p.x() + p.y() * p.y()) + p.z() * p.z();
    const auto rho = sqrt(rho2);

    // Check if valid
    const auto w = alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
    if (p.z() <= -w * rho + 1e-10) {
      throw int();
      return false;
    }

    const auto norm = alpha * rho + (1.0 - alpha) * p.z();
    const auto norm_inv = 1.0 / norm;

    const auto mx = p.x() * norm_inv;
    const auto my = p.y() * norm_inv;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    return true;
  }

  template <typename T>
  bool operator()(const T* const pose_host_ptr,
                  const T* const pose_target_ptr,
                  const T* const idist_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_w_h(pose_host_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_h(pose_host_ptr + 4);
    Eigen::Map<const Eigen::Quaternion<T>> q_w_t(pose_target_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_t(pose_target_ptr + 4);
    const T& idist = *idist_ptr;
    Eigen::Map<Patch<T>> residuals(residuals_ptr);

    // relative pose from host to target frame
    const Eigen::Quaternion<T> q_t_h = q_w_t.conjugate() * q_w_h;
    const Eigen::Matrix<T, 3, 3> R_t_h = q_t_h.toRotationMatrix();
    const Eigen::Matrix<T, 3, 1> t_t_h = q_w_t.conjugate() * (t_w_h - t_w_t);

    // transform points from host to target frame
    PatchVectors<T> p_target_scaled =
        (R_t_h * bearings_host_).colwise() + idist * t_t_h;

    // project points and interplate image
    Patch<T> intensities_target;
    for (int i = 0; i < p_target_scaled.cols(); ++i) {
      Eigen::Matrix<T, 2, 1> uv;
      if (!project(uv, Eigen::Matrix<T, 3, 1>(p_target_scaled.col(i)))) {
        residuals.setConstant(T(0.0));
        return true;
      }
      image_target_.Evaluate(uv[1], uv[0], &intensities_target[i]);
    }

    // residual is intensity difference between host and target frame
    residuals = intensities_target - intensities_host_;

    return true;
  }
#ifdef WITH_CODE_GENERATION
#include "benchmarks/photometricerror.h"
#else
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return false;
  }
#endif

 private:
  const Patch<double>& intensities_host_;
  const PatchVectors<double>& bearings_host_;
  const Interpolator& image_target_;
  const Intrinsics& intrinsics_;
};
}  // namespace ceres
#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARK_PHOTOMETRIC_ERROR_H_
