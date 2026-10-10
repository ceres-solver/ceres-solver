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
// Authors: darius.rueckert@fau.de (Darius Rueckert)
//          nikolaus@nikolaus-demmel.de (Nikolaus Demmel)
//          sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_BENCHMARK_COST_FUNCTIONS_H_
#define CERES_INTERNAL_BENCHMARK_COST_FUNCTIONS_H_

#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/constants.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

namespace ceres {

template <int kParameterBlockSize>
struct ConstantCostFunction {
  template <typename T>
  inline bool operator()(const T* const x, T* residuals) const {
    residuals[0] = T(5);
    return true;
  }
};

struct Linear1CostFunction {
  template <typename T>
  inline bool operator()(const T* const x, T* residuals) const {
    residuals[0] = x[0] + T(10);
    return true;
  }
};

struct Linear10CostFunction {
  template <typename T>
  inline bool operator()(const T* const x, T* residuals) const {
    for (int i = 0; i < 10; ++i) {
      residuals[i] = x[i] + T(i);
    }
    return true;
  }
};

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  inline bool operator()(const T* parameters, T* residuals) const {
    const T& b1 = parameters[0];
    const T& b2 = parameters[1];
    const T& b3 = parameters[2];
    const T& b4 = parameters[3];
    residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
    return true;
  }

 private:
  const double x_;
  const double y_;
};

struct SnavelyReprojectionError {
  static constexpr int kPointBlockIndex = 1;

  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  inline bool operator()(const T* const camera,
                         const T* const point,
                         T* residuals) const {
    T ox = T(observed_x);
    T oy = T(observed_y);

    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - ox;
    residuals[1] = predicted_y - oy;

    return true;
  }
  double observed_x;
  double observed_y;
};

// Photometric residual that computes the intensity difference for a patch
// between host and target frame. The point is parameterized with inverse
// distance relative to the host frame. The relative pose between host and
// target frame is computed from their respective absolute poses.
//
// The residual is similar to the one defined by Engel et al. [1]. Differences
// include:
//
// 1. Use of a camera model based on spherical projection, namely the enhanced
// unified camera model [2][3]. This is intended to bring some variability to
// the benchmark compared to the SnavelyReprojection that uses a
// polynomial-based distortion model.
//
// 2. To match the camera model, inverse distance parameterization is used for
// points instead of inverse depth [4].
//
// 3. For simplicity, camera intrinsics are assumed constant, and thus host
// frame points are passed as (unprojected) bearing vectors, which avoids the
// need for an 'unproject' function.
//
// 4. Some details of the residual in [1] are omitted for simplicity: The
// brightness transform parameters [a,b], the constant pre-weight w, and the
// per-pixel robust norm.
//
// [1] J. Engel, V. Koltun and D. Cremers, "Direct Sparse Odometry," in IEEE
// Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 3,
// pp. 611-625, 1 March 2018.
//
// [2] B. Khomutenko, G. Garcia and P. Martinet, "An Enhanced Unified Camera
// Model," in IEEE Robotics and Automation Letters, vol. 1, no. 1, pp. 137-144,
// Jan. 2016.
//
// [3] V. Usenko, N. Demmel and D. Cremers, "The Double Sphere Camera Model,"
// 2018 International Conference on 3D Vision (3DV), Verona, 2018, pp. 552-560.
//
// [4] H. Matsuki, L. von Stumberg, V. Usenko, J. Stückler and D. Cremers,
// "Omnidirectional DSO: Direct Sparse Odometry With Fisheye Cameras," in IEEE
// Robotics and Automation Letters, vol. 3, no. 4, pp. 3693-3700, Oct. 2018.
template <int PATCH_SIZE_ = 8>
struct PhotometricError {
  static constexpr int PATCH_SIZE = PATCH_SIZE_;
  static constexpr int POSE_SIZE = 7;
  static constexpr int POINT_SIZE = 1;

  using Grid = Grid2D<uint8_t, 1>;
  using Interpolator = BiCubicInterpolator<Grid>;
  using Intrinsics = Eigen::Array<double, 6, 1>;

  template <typename T>
  using Patch = Eigen::Array<T, PATCH_SIZE, 1>;

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
  inline bool Project(Eigen::Matrix<T, 2, 1>& proj,
                      const Eigen::Matrix<T, 3, 1>& p) const {
    const double& fx = intrinsics_[0];
    const double& fy = intrinsics_[1];
    const double& cx = intrinsics_[2];
    const double& cy = intrinsics_[3];
    const double& alpha = intrinsics_[4];
    const double& beta = intrinsics_[5];

    const T rho2 = beta * (p.x() * p.x() + p.y() * p.y()) + p.z() * p.z();
    const T rho = sqrt(rho2);

    // Check if 3D point is in domain of projection function.
    // See (8) and (17) in [3].
    constexpr double NUMERIC_EPSILON = 1e-10;
    const double w =
        alpha > 0.5 ? (1.0 - alpha) / alpha : alpha / (1.0 - alpha);
    if (p.z() <= -w * rho + NUMERIC_EPSILON) {
      return false;
    }

    const T norm = alpha * rho + (1.0 - alpha) * p.z();
    const T norm_inv = 1.0 / norm;

    const T mx = p.x() * norm_inv;
    const T my = p.y() * norm_inv;

    proj[0] = fx * mx + cx;
    proj[1] = fy * my + cy;

    return true;
  }

  template <typename T>
  inline bool operator()(const T* const pose_host_ptr,
                         const T* const pose_target_ptr,
                         const T* const idist_ptr,
                         T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_w_h(pose_host_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_h(pose_host_ptr + 4);
    Eigen::Map<const Eigen::Quaternion<T>> q_w_t(pose_target_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_t(pose_target_ptr + 4);
    const T& idist = *idist_ptr;
    Eigen::Map<Patch<T>> residuals(residuals_ptr);

    // Compute relative pose from host to target frame.
    const Eigen::Quaternion<T> q_t_h = q_w_t.conjugate() * q_w_h;
    const Eigen::Matrix<T, 3, 3> R_t_h = q_t_h.toRotationMatrix();
    const Eigen::Matrix<T, 3, 1> t_t_h = q_w_t.conjugate() * (t_w_h - t_w_t);

    // Transform points from host to target frame. 3D point in target frame is
    // scaled by idist for numerical stability when idist is close to 0
    // (projection is invariant to scaling).
    PatchVectors<T> p_target_scaled =
        (R_t_h * bearings_host_).colwise() + idist * t_t_h;

    // Project points and interpolate image.
    Patch<T> intensities_target;
    for (int i = 0; i < p_target_scaled.cols(); ++i) {
      Eigen::Matrix<T, 2, 1> uv;
      if (!Project(uv, Eigen::Matrix<T, 3, 1>(p_target_scaled.col(i)))) {
        // If any point of the patch is outside the domain of the projection
        // function, the residual cannot be evaluated. For the benchmark we want
        // to avoid this case and thus return false;
        return false;
      }

      // Mind the order of u and v: Evaluate takes (row, column), but u is
      // left-to-right and v top-to-bottom image axis.
      image_target_.Evaluate(uv[1], uv[0], &intensities_target[i]);
    }

    // Residual is intensity difference between host and target frame.
    residuals = intensities_target - intensities_host_;

    return true;
  }

 private:
  const Patch<double>& intensities_host_;
  const PatchVectors<double>& bearings_host_;
  const Interpolator& image_target_;
  const Intrinsics& intrinsics_;
};

// Relative pose error as one might use in SE(3) pose graph optimization.
// The measurement is a relative pose T_i_j, and the parameters are absolute
// poses T_w_i and T_w_j. For the residual we use the log of the residual
// pose, in split representation SO(3) x R^3.
struct RelativePoseError {
  RelativePoseError(Eigen::Quaterniond q_i_j, Eigen::Vector3d t_i_j)
      : meas_q_i_j_(std::move(q_i_j)), meas_t_i_j_(std::move(t_i_j)) {}

  template <typename T>
  inline bool operator()(const T* const pose_i_ptr,
                         const T* const pose_j_ptr,
                         T* residuals_ptr) const {
    Eigen::Map<const Eigen::Quaternion<T>> q_w_i(pose_i_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_i(pose_i_ptr + 4);
    Eigen::Map<const Eigen::Quaternion<T>> q_w_j(pose_j_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_w_j(pose_j_ptr + 4);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);

    // Compute estimate of relative pose from i to j.
    const Eigen::Quaternion<T> est_q_j_i = q_w_j.conjugate() * q_w_i;
    const Eigen::Matrix<T, 3, 1> est_t_j_i =
        q_w_j.conjugate() * (t_w_i - t_w_j);

    // Compute residual pose.
    const Eigen::Quaternion<T> res_q = meas_q_i_j_.cast<T>() * est_q_j_i;
    const Eigen::Matrix<T, 3, 1> res_t =
        meas_q_i_j_.cast<T>() * est_t_j_i + meas_t_i_j_;

    // Convert quaternion to ceres convention (Eigen stores xyzw, Ceres wxyz).
    Eigen::Matrix<T, 4, 1> res_q_ceres;
    res_q_ceres << res_q.w(), res_q.vec();

    // Residual is log of pose. Use split representation SO(3) x R^3.
    QuaternionToAngleAxis(res_q_ceres.data(), residuals.data());
    residuals.template bottomRows<3>() = res_t;

    return true;
  }

 private:
  // Measurement of relative pose from j to i.
  Eigen::Quaterniond meas_q_i_j_;
  Eigen::Vector3d meas_t_i_j_;
};

// The brdf is based on:
// Burley, Brent, and Walt Disney Animation Studios. "Physically-based shading
// at disney." ACM SIGGRAPH. Vol. 2012. 2012.
//
// The implementation is based on:
// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
struct Brdf {
 public:
  template <typename T>
  inline bool operator()(const T* const material,
                         const T* const c_ptr,
                         const T* const n_ptr,
                         const T* const v_ptr,
                         const T* const l_ptr,
                         const T* const x_ptr,
                         const T* const y_ptr,
                         T* residual) const {
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    T metallic = material[0];
    T subsurface = material[1];
    T specular = material[2];
    T roughness = material[3];
    T specular_tint = material[4];
    T anisotropic = material[5];
    T sheen = material[6];
    T sheen_tint = material[7];
    T clearcoat = material[8];
    T clearcoat_gloss = material[9];

    Eigen::Map<const Vec3> c(c_ptr);
    Eigen::Map<const Vec3> n(n_ptr);
    Eigen::Map<const Vec3> v(v_ptr);
    Eigen::Map<const Vec3> l(l_ptr);
    Eigen::Map<const Vec3> x(x_ptr);
    Eigen::Map<const Vec3> y(y_ptr);

    const T n_dot_l = n.dot(l);
    const T n_dot_v = n.dot(v);

    const Vec3 l_p_v = l + v;
    const Vec3 h = l_p_v / l_p_v.norm();

    const T n_dot_h = n.dot(h);
    const T l_dot_h = l.dot(h);

    const T h_dot_x = h.dot(x);
    const T h_dot_y = h.dot(y);

    const T c_dlum = T(0.3) * c[0] + T(0.6) * c[1] + T(0.1) * c[2];

    const Vec3 c_tint = c / c_dlum;

    const Vec3 c_spec0 =
        Lerp(specular * T(0.08) *
                 Lerp(Vec3(T(1), T(1), T(1)), c_tint, specular_tint),
             c,
             metallic);
    const Vec3 c_sheen = Lerp(Vec3(T(1), T(1), T(1)), c_tint, sheen_tint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    const T fl = SchlickFresnel(n_dot_l);
    const T fv = SchlickFresnel(n_dot_v);
    const T fd_90 = T(0.5) + T(2) * l_dot_h * l_dot_h * roughness;
    const T fd = Lerp(T(1), fd_90, fl) * Lerp(T(1), fd_90, fv);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    const T fss_90 = l_dot_h * l_dot_h * roughness;
    const T fss = Lerp(T(1), fss_90, fl) * Lerp(T(1), fss_90, fv);
    const T ss =
        T(1.25) * (fss * (T(1) / (n_dot_l + n_dot_v) - T(0.5)) + T(0.5));

    // specular
    const T eps = T(0.001);
    const T aspct = Aspect(anisotropic);
    const T ax_temp = Square(roughness) / aspct;
    const T ay_temp = Square(roughness) * aspct;
    const T ax = (ax_temp < eps ? eps : ax_temp);
    const T ay = (ay_temp < eps ? eps : ay_temp);
    const T ds = GTR2Aniso(n_dot_h, h_dot_x, h_dot_y, ax, ay);
    const T fh = SchlickFresnel(l_dot_h);
    const Vec3 fs = Lerp(c_spec0, Vec3(T(1), T(1), T(1)), fh);
    const T roughg = Square(roughness * T(0.5) + T(0.5));
    const T ggxn_dot_l = SmithG_GGX(n_dot_l, roughg);
    const T ggxn_dot_v = SmithG_GGX(n_dot_v, roughg);
    const T gs = ggxn_dot_l * ggxn_dot_v;

    // sheen
    const Vec3 f_sheen = fh * sheen * c_sheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    const T a = Lerp(T(0.1), T(0.001), clearcoat_gloss);
    const T dr = GTR1(n_dot_h, a);
    const T fr = Lerp(T(0.04), T(1), fh);
    const T cggxn_dot_l = SmithG_GGX(n_dot_l, T(0.25));
    const T cggxn_dot_v = SmithG_GGX(n_dot_v, T(0.25));
    const T gr = cggxn_dot_l * cggxn_dot_v;

    const Vec3 result_no_cosine =
        (T(1.0 / constants::pi) * Lerp(fd, ss, subsurface) * c + f_sheen) *
            (T(1) - metallic) +
        gs * fs * ds +
        Vec3(T(0.25), T(0.25), T(0.25)) * clearcoat * gr * fr * dr;
    const Vec3 result = n_dot_l * result_no_cosine;
    residual[0] = result(0);
    residual[1] = result(1);
    residual[2] = result(2);

    return true;
  }

  template <typename T>
  inline T SchlickFresnel(const T& u) const {
    T m = T(1) - u;
    const T m2 = m * m;
    return m2 * m2 * m;  // (1-u)^5
  }

  template <typename T>
  inline T Aspect(const T& anisotropic) const {
    return T(sqrt(T(1) - anisotropic * T(0.9)));
  }

  template <typename T>
  inline T SmithG_GGX(const T& n_dot_v, const T& alpha_g) const {
    const T a = alpha_g * alpha_g;
    const T b = n_dot_v * n_dot_v;
    return T(1) / (n_dot_v + T(sqrt(a + b - a * b)));
  }

  // Generalized-Trowbridge-Reitz (GTR) Microfacet Distribution
  // See paper, Appendix B
  template <typename T>
  inline T GTR1(const T& n_dot_h, const T& a) const {
    T result = T(0);

    if (a >= T(1)) {
      result = T(1 / constants::pi);
    } else {
      const T a2 = a * a;
      const T t = T(1) + (a2 - T(1)) * n_dot_h * n_dot_h;
      result = (a2 - T(1)) / (T(constants::pi) * T(log(a2) * t));
    }
    return result;
  }

  template <typename T>
  inline T GTR2Aniso(const T& n_dot_h,
                     const T& h_dot_x,
                     const T& h_dot_y,
                     const T& ax,
                     const T& ay) const {
    return T(1) / (T(constants::pi) * ax * ay *
                   Square(Square(h_dot_x / ax) + Square(h_dot_y / ay) +
                          n_dot_h * n_dot_h));
  }

  template <typename T>
  inline T Lerp(const T& a, const T& b, const T& u) const {
    return a + u * (b - a);
  }

  template <typename Derived1, typename Derived2>
  inline typename Derived1::PlainObject Lerp(
      const Eigen::MatrixBase<Derived1>& a,
      const Eigen::MatrixBase<Derived2>& b,
      typename Derived1::Scalar alpha) const {
    return (typename Derived1::Scalar(1) - alpha) * a + alpha * b;
  }

  template <typename T>
  inline T Square(const T& x) const {
    return x * x;
  }
};

// Functors for benchmarking CostFunctionToFunctor (static sizes) and
// DynamicCostFunctionToFunctor (dynamic sizes).
//
// CostFunctionToFunctor and DynamicCostFunctionToFunctor allow an existing
// CostFunction object (which only accepts `double` inputs and computes `double`
// residuals and Jacobians via CostFunction::Evaluate) to be used as a templated
// functor inside another cost functor. When called with `Jet<double, N>`
// arguments during automatic differentiation of the outer functor, the adapter:
//   1. Extracts the scalar (`double`) parts of the input Jets.
//   2. Calls the inner CostFunction::Evaluate to compute the inner residuals
//      and Jacobians with respect to the inner inputs.
//   3. Applies the chain rule to multiply the inner Jacobians by the derivative
//      (`.v`) parts of the input Jets to form the output residual Jets.
//
// To benchmark this composition realistically, we split a standard pinhole
// camera projection with radial distortion into two stages:
//
//   - Stage 1 (`InnerProjectionFunctor`): Projects a 3D point already in the
//     camera coordinate frame, `p_cam = [X, Y, Z]`, into 2D pixel coordinates
//     using camera `intrinsics = [focal, k1, k2]`:
//       xp = X / Z,  yp = Y / Z,  r^2 = xp^2 + yp^2
//       distortion = 1 + k1 * r^2 + k2 * r^4
//       residuals  = [focal * distortion * xp, focal * distortion * yp]
//     This inner functor is wrapped in an `AutoDiffCostFunction<..., 2, 3, 3>`,
//     representing a pre-existing CostFunction object.
//
//   - Stage 2 (`OuterProjectionFunctor`): Takes four 3D parameter blocks
//     (`rotation` angle-axis, `translation`, `intrinsics`, and a world-frame
//     `point`), transforms the world point into the camera frame:
//       p_cam = AngleAxisRotatePoint(rotation, point) + translation
//     and then invokes the inner CostFunction via either:
//       * `CostFunctionToFunctor<2, 3, 3>` (when `!kUseDynamicAdapter`), or
//       * `DynamicCostFunctionToFunctor`   (when `kUseDynamicAdapter`).
struct InnerProjectionFunctor {
  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    const T xp = point[0] / point[2];
    const T yp = point[1] / point[2];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (intrinsics[1] + intrinsics[2] * r2);
    residuals[0] = intrinsics[0] * distortion * xp;
    residuals[1] = intrinsics[0] * distortion * yp;
    return true;
  }
};

template <bool kUseDynamicAdapter>
struct OuterProjectionFunctor {
  OuterProjectionFunctor()
      : inner_(std::make_unique<
               AutoDiffCostFunction<InnerProjectionFunctor, 2, 3, 3>>(
            std::make_unique<InnerProjectionFunctor>())) {}

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    // Transform the 3D point from the world frame into the camera frame.
    T p_cam[3];
    AngleAxisRotatePoint(rotation, point, p_cam);
    p_cam[0] += translation[0];
    p_cam[1] += translation[1];
    p_cam[2] += translation[2];

    // Project the camera-frame point into 2D pixel coordinates by invoking the
    // wrapped inner CostFunction through CostFunctionToFunctor (static) or
    // DynamicCostFunctionToFunctor (dynamic).
    if constexpr (!kUseDynamicAdapter) {
      return inner_(intrinsics, p_cam, residuals);
    } else {
      const T* params[2] = {intrinsics, p_cam};
      return inner_(params, residuals);
    }
  }

  std::conditional_t<!kUseDynamicAdapter,
                     CostFunctionToFunctor<2, 3, 3>,
                     DynamicCostFunctionToFunctor>
      inner_;
};

}  // namespace ceres

#endif  // CERES_INTERNAL_BENCHMARK_COST_FUNCTIONS_H_
