#pragma once
#include <Eigen/Core>
#include <cmath>

#include "ceres/codegen/codegen_cost_function.h"
namespace ceres {

// The brdf is based on:
// Burley, Brent, and Walt Disney Animation Studios. "Physically-based shading
// at disney." ACM SIGGRAPH. Vol. 2012. 2012.
//
// The implementation is based on:
// https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf
struct Brdf : public ceres::CodegenCostFunction<3, 10, 3, 3, 3, 3, 3, 3> {
 public:
  Brdf() {}

  template <typename T>
  bool operator()(const T* const material,
                  const T* const c_,
                  const T* const n_,
                  const T* const v_,
                  const T* const l_,
                  const T* const x_,
                  const T* const y_,
                  T* residual) const {
    using ceres::Ternary;
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

    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    Eigen::Map<const Vec3> c(c_);
    Eigen::Map<const Vec3> n(n_);
    Eigen::Map<const Vec3> v(v_);
    Eigen::Map<const Vec3> l(l_);
    Eigen::Map<const Vec3> x(x_);
    Eigen::Map<const Vec3> y(y_);

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
        lerp(specular * T(0.08) *
                 lerp(Vec3(T(1), T(1), T(1)), c_tint, specular_tint),
             c,
             metallic);
    const Vec3 c_sheen = lerp(Vec3(T(1), T(1), T(1)), c_tint, sheen_tint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    const T fl = schlickFresnel(n_dot_l);
    const T fv = schlickFresnel(n_dot_v);
    const T fd_90 = T(0.5) + T(2) * l_dot_h * l_dot_h * roughness;
    const T fd = lerp(T(1), fd_90, fl) * lerp(T(1), fd_90, fv);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    const T fss_90 = l_dot_h * l_dot_h * roughness;
    const T fss = lerp(T(1), fss_90, fl) * lerp(T(1), fss_90, fv);
    const T ss =
        T(1.25) * (fss * (T(1) / (n_dot_l + n_dot_v) - T(0.5)) + T(0.5));

    // specular
    const T eps = T(0.001);
    const T aspct = aspect(anisotropic);
    const T ax_temp = sqr(roughness) / aspct;
    const T ay_temp = sqr(roughness) * aspct;
    const T ax = Ternary(ax_temp < eps, eps, ax_temp);
    const T ay = Ternary(ay_temp < eps, eps, ay_temp);
    const T ds = GTR2_aniso(n_dot_h, h_dot_x, h_dot_y, ax, ay);
    const T fh = schlickFresnel(l_dot_h);
    const Vec3 fs = lerp(c_spec0, Vec3(T(1), T(1), T(1)), fh);
    const T roughg = sqr(roughness * T(0.5) + T(0.5));
    const T ggxn_dot_l = smithG_GGX(n_dot_l, roughg);
    const T ggxn_dot_v = smithG_GGX(n_dot_v, roughg);
    const T gs = ggxn_dot_l * ggxn_dot_v;

    // sheen
    const Vec3 f_sheen = fh * sheen * c_sheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    const T a = lerp(T(0.1), T(0.001), clearcoat_gloss);
    const T dr = GTR1(n_dot_h, a);
    const T fr = lerp(T(0.04), T(1), fh);
    const T cggxn_dot_l = smithG_GGX(n_dot_l, T(0.25));
    const T cggxn_dot_v = smithG_GGX(n_dot_v, T(0.25));
    const T gr = cggxn_dot_l * cggxn_dot_v;

    const Vec3 result_no_cosine =
        (T(1.0 / M_PI) * lerp(fd, ss, subsurface) * c + f_sheen) *
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
  T schlickFresnel(const T& u) const {
    T m = T(1) - u;
    const T m2 = m * m;
    return m2 * m2 * m;  // (1-u)^5
  }

  template <typename T>
  T aspect(const T& anisotropic) const {
    return T(sqrt(T(1) - anisotropic * T(0.9)));
  }

  template <typename T>
  T smithG_GGX(const T& n_dot_v, const T& alpha_g) const {
    const T a = alpha_g * alpha_g;
    const T b = n_dot_v * n_dot_v;
    return T(1) / (n_dot_v + T(sqrt(a + b - a * b)));
  }

  template <typename T>
  T GTR1(const T& n_dot_h, const T& a) const {
    T result = T(0);

    CERES_IF(a >= T(1)) { result = T(1 / M_PI); }
    CERES_ELSE {
      const T a2 = a * a;
      const T t = T(1) + (a2 - T(1)) * n_dot_h * n_dot_h;
      result = (a2 - T(1)) / (T(M_PI) * T(log(a2) * t));
    }
    CERES_ENDIF;
    return result;
  }

  template <typename T>
  T GTR2_aniso(const T& n_dot_h,
               const T& h_dot_x,
               const T& h_dot_y,
               const T& ax,
               const T& ay) const {
    return T(1) /
           (T(M_PI) * ax * ay *
            sqr(sqr(h_dot_x / ax) + sqr(h_dot_y / ay) + n_dot_h * n_dot_h));
  }

  template <typename T>
  inline T lerp(const T& a, const T& b, const T& u) const {
    return a + u * (b - a);
  }

  template <typename Derived1, typename Derived2>
  typename Derived1::PlainObject lerp(const Eigen::MatrixBase<Derived1>& a,
                                      const Eigen::MatrixBase<Derived2>& b,
                                      typename Derived1::Scalar alpha) const {
    return (typename Derived1::Scalar(1) - alpha) * a + alpha * b;
  }

  template <typename T>
  inline T sqr(const T& x) const {
    return x * x;
  }

#ifdef WITH_CODE_GENERATION
#include "benchmarks/brdf.h"
#else
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    return false;
  }
#endif
};

}  // namespace ceres
