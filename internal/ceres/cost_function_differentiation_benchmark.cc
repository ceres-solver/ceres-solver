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
// Authors: darius.rueckert@fau.de (Darius Rueckert)
//          nikolaus@nikolaus-demmel.de (Nikolaus Demmel)
//          sameeragarwal@google.com (Sameer Agarwal)

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "ceres/ceres.h"
#include "ceres/constants.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

namespace ceres {
namespace {

enum DiffType {
  kAutoDiff,
  kDynamicAutoDiff,
  kAutoDiffDynamicResiduals,
  kNumericForward,
  kDynamicNumericForward,
  kNumericCentral,
  kDynamicNumericCentral,
  kNumericRidders,
  kDynamicNumericRidders,
};

// Controls which outputs CostFunction::Evaluate is called to compute:
//   kResidualsOnly:             evaluate residuals only (jacobians = nullptr)
//   kResidualsAndJacobians:     evaluate residuals and all parameter Jacobians
//   kResidualsAndPointJacobian: evaluate residuals and only the 3D point
//                               Jacobian (holding camera parameters constant)
enum EvaluationType {
  kResidualsOnly,
  kResidualsAndJacobians,
  kResidualsAndPointJacobian,
};

// Transforms a static functor into a dynamic one.
template <typename CostFunctionType, int kNumParameterBlocks>
class ToDynamic {
 public:
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<CostFunctionType, Args&&...>>>
  explicit ToDynamic(Args&&... args)
      : cost_function_(std::forward<Args>(args)...) {}

  template <typename T>
  bool operator()(const T* const* parameters, T* residuals) const {
    return Apply(
        parameters, residuals, std::make_index_sequence<kNumParameterBlocks>());
  }

 private:
  template <typename T, size_t... Indices>
  bool Apply(const T* const* parameters,
             T* residuals,
             std::index_sequence<Indices...>) const {
    return cost_function_(parameters[Indices]..., residuals);
  }

  CostFunctionType cost_function_;
};

// Creates a CostFunction wrapping `CostFunctor` with `kNumResiduals` residuals
// and parameter blocks of sizes `Ns...` using the differentiation wrapper
// specified by `kDiffType`.
template <DiffType kDiffType>
struct CostFunctionFactory {
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Args>
  static std::unique_ptr<CostFunction> Create(Args&&... args) {
    constexpr int kNumParameterBlocks = sizeof...(Ns);
    using DynamicFunctor = ToDynamic<CostFunctor, kNumParameterBlocks>;

    if constexpr (kDiffType == kAutoDiff) {
      return std::make_unique<
          AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kAutoDiffDynamicResiduals) {
      return std::make_unique<
          AutoDiffCostFunction<CostFunctor, DYNAMIC, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...),
          kNumResiduals);
    } else if constexpr (kDiffType == kDynamicAutoDiff) {
      auto dynamic_function =
          std::make_unique<DynamicAutoDiffCostFunction<DynamicFunctor>>(
              std::make_unique<DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericForward) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, FORWARD, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericForward) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, FORWARD>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericCentral) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, CENTRAL, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericCentral) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, CENTRAL>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericRidders) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, RIDDERS, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericRidders) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, RIDDERS>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    }
  }
};

// ============================================================================
// Benchmark Cost Functors
// ============================================================================

template <int kParameterBlockSize>
struct ConstantCostFunction {
  template <typename T>
  inline bool operator()(const T* const /*x*/, T* residuals) const {
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
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  inline bool operator()(const T* const camera,
                         const T* const point,
                         T* residuals) const {
    const T ox = T(observed_x);
    const T oy = T(observed_y);

    T p[3];
    AngleAxisRotatePoint(camera, point, p);

    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    residuals[0] = predicted_x - ox;
    residuals[1] = predicted_y - oy;
    return true;
  }

  double observed_x;
  double observed_y;
};

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

    const Eigen::Quaternion<T> q_t_h = q_w_t.conjugate() * q_w_h;
    const Eigen::Matrix<T, 3, 3> R_t_h = q_t_h.toRotationMatrix();
    const Eigen::Matrix<T, 3, 1> t_t_h = q_w_t.conjugate() * (t_w_h - t_w_t);

    PatchVectors<T> p_target_scaled =
        (R_t_h * bearings_host_).colwise() + idist * t_t_h;

    Patch<T> intensities_target;
    for (int i = 0; i < p_target_scaled.cols(); ++i) {
      Eigen::Matrix<T, 2, 1> uv;
      if (!Project(uv, Eigen::Matrix<T, 3, 1>(p_target_scaled.col(i)))) {
        return false;
      }
      image_target_.Evaluate(uv[1], uv[0], &intensities_target[i]);
    }

    residuals = intensities_target - intensities_host_;
    return true;
  }

 private:
  const Patch<double>& intensities_host_;
  const PatchVectors<double>& bearings_host_;
  const Interpolator& image_target_;
  const Intrinsics& intrinsics_;
};

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

    const Eigen::Quaternion<T> est_q_j_i = q_w_j.conjugate() * q_w_i;
    const Eigen::Matrix<T, 3, 1> est_t_j_i =
        q_w_j.conjugate() * (t_w_i - t_w_j);

    const Eigen::Quaternion<T> res_q = meas_q_i_j_.cast<T>() * est_q_j_i;
    const Eigen::Matrix<T, 3, 1> res_t =
        meas_q_i_j_.cast<T>() * est_t_j_i + meas_t_i_j_;

    Eigen::Matrix<T, 4, 1> res_q_ceres;
    res_q_ceres << res_q.w(), res_q.vec();

    QuaternionToAngleAxis(res_q_ceres.data(), residuals.data());
    residuals.template bottomRows<3>() = res_t;
    return true;
  }

 private:
  Eigen::Quaterniond meas_q_i_j_;
  Eigen::Vector3d meas_t_i_j_;
};

struct Brdf {
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

    const T metallic = material[0];
    const T subsurface = material[1];
    const T specular = material[2];
    const T roughness = material[3];
    const T specular_tint = material[4];
    const T anisotropic = material[5];
    const T sheen = material[6];
    const T sheen_tint = material[7];
    const T clearcoat = material[8];
    const T clearcoat_gloss = material[9];

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

    const T fl = SchlickFresnel(n_dot_l);
    const T fv = SchlickFresnel(n_dot_v);
    const T fd_90 = T(0.5) + T(2) * l_dot_h * l_dot_h * roughness;
    const T fd = Lerp(T(1), fd_90, fl) * Lerp(T(1), fd_90, fv);

    const T fss_90 = l_dot_h * l_dot_h * roughness;
    const T fss = Lerp(T(1), fss_90, fl) * Lerp(T(1), fss_90, fv);
    const T ss =
        T(1.25) * (fss * (T(1) / (n_dot_l + n_dot_v) - T(0.5)) + T(0.5));

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

    const Vec3 f_sheen = fh * sheen * c_sheen;

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
    const T m = T(1) - u;
    const T m2 = m * m;
    return m2 * m2 * m;
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

  template <typename T>
  inline T GTR1(const T& n_dot_h, const T& a) const {
    if (a >= T(1)) {
      return T(1 / constants::pi);
    }
    const T a2 = a * a;
    const T t = T(1) + (a2 - T(1)) * n_dot_h * n_dot_h;
    return (a2 - T(1)) / (T(constants::pi) * T(log(a2) * t));
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

// ============================================================================
// CostFunction Differentiation Benchmarks
// ============================================================================

template <int kParameterBlockSize, DiffType kDiffType>
void BM_Constant(benchmark::State& state) {
  constexpr int kNumResiduals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, kNumResiduals> residuals{};
  std::array<double, kNumResiduals * kParameterBlockSize> jacobian_values{};
  double* jacobians[] = {jacobian_values.data()};

  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<ConstantCostFunction<kParameterBlockSize>,
                                  1,
                                  kParameterBlockSize>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals.data(), jacobians));
  }
}

#define REGISTER_CONSTANT_BENCHMARKS(N)          \
  BENCHMARK_TEMPLATE(BM_Constant, N, kAutoDiff); \
  BENCHMARK_TEMPLATE(BM_Constant, N, kDynamicAutoDiff)

REGISTER_CONSTANT_BENCHMARKS(1);
REGISTER_CONSTANT_BENCHMARKS(10);
REGISTER_CONSTANT_BENCHMARKS(20);
REGISTER_CONSTANT_BENCHMARKS(30);
REGISTER_CONSTANT_BENCHMARKS(40);
REGISTER_CONSTANT_BENCHMARKS(50);
REGISTER_CONSTANT_BENCHMARKS(60);

#undef REGISTER_CONSTANT_BENCHMARKS

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Linear1(benchmark::State& state) {
  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Linear1CostFunction,
                                                      1,
                                                      1>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Linear10(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Linear10CostFunction,
                                                      10,
                                                      10>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Rat43(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals = 0.0;
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Rat43CostFunctor, 1, 4>(
          x, y);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, &residuals, jacobians_ptr));
  }
}

#define REGISTER_RAT43_BENCHMARKS(Diff)                        \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsOnly);          \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsAndJacobians)

REGISTER_RAT43_BENCHMARKS(kAutoDiff);
REGISTER_RAT43_BENCHMARKS(kDynamicAutoDiff);
REGISTER_RAT43_BENCHMARKS(kNumericForward);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericForward);
REGISTER_RAT43_BENCHMARKS(kNumericCentral);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericCentral);
REGISTER_RAT43_BENCHMARKS(kNumericRidders);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericRidders);

#undef REGISTER_RAT43_BENCHMARKS

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_SnavelyReprojection(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {
      (kEvalType == kResidualsAndJacobians) ? jacobian1 : nullptr,
      (kEvalType != kResidualsOnly) ? jacobian2 : nullptr,
  };
  double** jacobians_ptr =
      (kEvalType == kResidualsOnly) ? nullptr : jacobians;

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<SnavelyReprojectionError, 2, 9, 3>(x, y);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

#define REGISTER_SNAVELY_ALL_MODES(Diff)                                      \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsOnly);           \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsAndJacobians);   \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsAndPointJacobian)

REGISTER_SNAVELY_ALL_MODES(kAutoDiff);
REGISTER_SNAVELY_ALL_MODES(kDynamicAutoDiff);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kAutoDiffDynamicResiduals,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kAutoDiffDynamicResiduals,
                   kResidualsAndJacobians);
REGISTER_SNAVELY_ALL_MODES(kNumericForward);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericForward);
REGISTER_SNAVELY_ALL_MODES(kNumericCentral);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericCentral);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection, kNumericRidders, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kNumericRidders,
                   kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kDynamicNumericRidders,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kDynamicNumericRidders,
                   kResidualsAndJacobians);

#undef REGISTER_SNAVELY_ALL_MODES

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Photometric(benchmark::State& state) {
  constexpr int PATCH_SIZE = 8;

  using FunctorType = PhotometricError<PATCH_SIZE>;
  using ImageType = Eigen::Matrix<uint8_t, 128, 128, Eigen::RowMajor>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  double parameter_block3[] = {1.};
  double* parameters[] = {parameter_block1, parameter_block2, parameter_block3};

  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  double jacobian1[FunctorType::PATCH_SIZE * FunctorType::POSE_SIZE];
  double jacobian2[FunctorType::PATCH_SIZE * FunctorType::POSE_SIZE];
  double jacobian3[FunctorType::PATCH_SIZE * FunctorType::POINT_SIZE];
  double residuals[FunctorType::PATCH_SIZE];
  double* jacobians[] = {jacobian1, jacobian2, jacobian3};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::mt19937::result_type seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  std::uniform_int_distribution<unsigned int> uniform0255(0, 255);

  FunctorType::Patch<double> intensities_host =
      FunctorType::Patch<double>::NullaryExpr(
          [&]() { return uniform0255(gen); });

  FunctorType::PatchVectors<double> bearings_host =
      FunctorType::PatchVectors<double>::NullaryExpr(
          [&]() { return uniform01(gen); });
  bearings_host.row(2).array() = 1;
  bearings_host.colwise().normalize();

  ImageType image = ImageType::NullaryExpr(
      [&]() { return static_cast<uint8_t>(uniform0255(gen)); });
  FunctorType::Grid grid(image.data(), 0, image.rows(), 0, image.cols());
  FunctorType::Interpolator image_target(grid);

  FunctorType::Intrinsics intrinsics;
  intrinsics << 128, 128, 1, -1, 0.5, 0.5;

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<FunctorType,
                                                      FunctorType::PATCH_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POINT_SIZE>(
          intensities_host, bearings_host, image_target, intrinsics);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_RelativePose(benchmark::State& state) {
  using FunctorType = RelativePoseError;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  double* parameters[] = {parameter_block1, parameter_block2};

  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  double jacobian1[6 * 7];
  double jacobian2[6 * 7];
  double residuals[6];
  double* jacobians[] = {jacobian1, jacobian2};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  Eigen::Quaterniond q_i_j = Eigen::Quaterniond(1, 2, 3, 4).normalized();
  Eigen::Vector3d t_i_j(1, 2, 3);

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<FunctorType, 6, 7, 7>(
          q_i_j, t_i_j);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_RelativePose, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose,
                   kDynamicNumericCentral,
                   kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Brdf(benchmark::State& state) {
  using FunctorType = Brdf;

  double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  auto c = Eigen::Vector3d(0.1, 0.2, 0.3);
  auto n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  auto v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  auto l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  auto x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  auto y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  double* parameters[7] = {
      material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()};

  double jacobian[(10 + 6 * 3) * 3];
  double residuals[3];
  // clang-format off
  double* jacobians[7] = {
      jacobian + 0,      jacobian + 10 * 3, jacobian + 13 * 3,
      jacobian + 16 * 3, jacobian + 19 * 3, jacobian + 22 * 3,
      jacobian + 25 * 3,
  };
  // clang-format on
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<FunctorType, 3, 10, 3, 3, 3, 3, 3, 3>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsAndJacobians);

// ============================================================================
// CostFunctionToFunctor / DynamicCostFunctionToFunctor Benchmarks
// ============================================================================

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

template <DiffType kDiffType>
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
    T p[3];
    AngleAxisRotatePoint(rotation, point, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];
    if constexpr (kDiffType == kAutoDiff) {
      return inner_(intrinsics, p, residuals);
    } else {
      const T* params[2] = {intrinsics, p};
      return inner_(params, residuals);
    }
  }

  std::conditional_t<kDiffType == kAutoDiff,
                     CostFunctionToFunctor<2, 3, 3>,
                     DynamicCostFunctionToFunctor>
      inner_;
};

template <DiffType kDiffType>
void BM_CostFunctionToFunctor(benchmark::State& state) {
  std::unique_ptr<CostFunction> cost_function = std::make_unique<
      AutoDiffCostFunction<OuterProjectionFunctor<kDiffType>, 2, 3, 3, 3, 3>>(
      std::make_unique<OuterProjectionFunctor<kDiffType>>());
  double rot[3] = {0.1, -0.2, 0.05};
  double trans[3] = {0.5, -0.1, 2.0};
  double intr[3] = {500.0, -0.01, 0.001};
  double pt[3] = {0.3, -0.4, 5.0};
  const double* params[4] = {rot, trans, intr, pt};
  double residuals[2];
  double jacobian[4 * 2 * 3];
  double* jacobians[4] = {
      jacobian + 0, jacobian + 6, jacobian + 12, jacobian + 18};

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(params, residuals, jacobians));
  }
}

BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kDynamicAutoDiff);

}  // namespace
}  // namespace ceres

BENCHMARK_MAIN();
