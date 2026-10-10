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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
//
#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARK_BRDF_COST_FUNCTION_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARK_BRDF_COST_FUNCTION_H_

#include <Eigen/Core>
#include <cmath>
#include <memory>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/cost_function_factory.h"
#include "ceres/constants.h"

namespace ceres {

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

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Brdf(benchmark::State& state) {
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

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARK_BRDF_COST_FUNCTION_H_
