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
                  const T* const C_,
                  const T* const N_,
                  const T* const V_,
                  const T* const L_,
                  const T* const X_,
                  const T* const Y_,
                  T* residual) const {
    using ceres::Ternary;
    T metallic = material[0];
    T subsurface = material[1];
    T specular = material[2];
    T roughness = material[3];
    T specularTint = material[4];
    T anisotropic = material[5];
    T sheen = material[6];
    T sheenTint = material[7];
    T clearcoat = material[8];
    T clearcoatGloss = material[9];

    using Vec2 = Eigen::Matrix<T, 2, 1>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    Vec3 C, N, V, L, X, Y;

    C(0) = C_[0];
    C(1) = C_[1];
    C(2) = C_[2];

    N(0) = N_[0];
    N(1) = N_[1];
    N(2) = N_[2];

    V(0) = V_[0];
    V(1) = V_[1];
    V(2) = V_[2];

    L(0) = L_[0];
    L(1) = L_[1];
    L(2) = L_[2];

    X(0) = X_[0];
    X(1) = X_[1];
    X(2) = X_[2];

    Y(0) = Y_[0];
    Y(1) = Y_[1];
    Y(2) = Y_[2];

    const T NdotL = N.dot(L);
    const T NdotV = N.dot(V);

    const Vec3 LpV = L + V;
    const Vec3 H = LpV / LpV.norm();

    const T NdotH = N.dot(H);
    const T LdotH = L.dot(H);

    const T HdotX = H.dot(X);
    const T HdotY = H.dot(Y);

    const Vec3 Cdlin = C;
    const T Cdlum = T(0.3) * Cdlin[0] + T(0.6) * Cdlin[1] + T(0.1) * Cdlin[2];

    const Vec3 Ctint = Cdlin / Cdlum;

    const Vec3 Cspec0 = lerp(
        specular * T(0.08) * lerp(Vec3(T(1), T(1), T(1)), Ctint, specularTint),
        Cdlin,
        metallic);
    const Vec3 Csheen = lerp(Vec3(T(1), T(1), T(1)), Ctint, sheenTint);

    // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
    // and mix in diffuse retro-reflection based on roughness
    const T FL = schlickFresnel(NdotL);
    const T FV = schlickFresnel(NdotV);
    const T Fd90 = T(0.5) + T(2) * LdotH * LdotH * roughness;
    const T Fd = lerp(T(1), Fd90, FL) * lerp(T(1), Fd90, FV);

    // Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
    // 1.25 scale is used to (roughly) preserve albedo
    // Fss90 used to "flatten" retroreflection based on roughness
    const T Fss90 = LdotH * LdotH * roughness;
    const T Fss = lerp(T(1), Fss90, FL) * lerp(T(1), Fss90, FV);
    const T ss = T(1.25) * (Fss * (T(1) / (NdotL + NdotV) - T(0.5)) + T(0.5));

    // specular
    const T eps = T(0.001);
    const T aspct = aspect(anisotropic);
    const T axTemp = sqr(roughness) / aspct;
    const T ayTemp = sqr(roughness) * aspct;
    const T ax = Ternary(axTemp < eps, eps, axTemp);
    const T ay = Ternary(ayTemp < eps, eps, ayTemp);
    const T Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay);
    const T FH = schlickFresnel(LdotH);
    const Vec3 Fs = lerp(Cspec0, Vec3(T(1), T(1), T(1)), FH);
    const T roughg = sqr(roughness * T(0.5) + T(0.5));
    const T GGXNdotL = smithG_GGX(NdotL, roughg);
    const T GGXNdotV = smithG_GGX(NdotV, roughg);
    const T Gs = GGXNdotL * GGXNdotV;

    // sheen
    const Vec3 Fsheen = FH * sheen * Csheen;

    // clearcoat (ior = 1.5 -> F0 = 0.04)
    const T a = lerp(T(0.1), T(0.001), clearcoatGloss);
    const T Dr = GTR1(NdotH, a);
    const T Fr = lerp(T(0.04), T(1), FH);
    const T cGGXNdotL = smithG_GGX(NdotL, T(0.25));
    const T cGGXNdotV = smithG_GGX(NdotV, T(0.25));
    const T Gr = cGGXNdotL * cGGXNdotV;

    const Vec3 resultNoCosine =
        (T(1.0 / M_PI) * lerp(Fd, ss, subsurface) * Cdlin + Fsheen) *
            (T(1) - metallic) +
        Gs * Fs * Ds +
        Vec3(T(0.25), T(0.25), T(0.25)) * clearcoat * Gr * Fr * Dr;
    const Vec3 result = NdotL * resultNoCosine;
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
  T smithG_GGX(const T& Ndotv, const T& alphaG) const {
    const T a = alphaG * alphaG;
    const T b = Ndotv * Ndotv;
    return T(1) / (Ndotv + T(sqrt(a + b - a * b)));
  }

  template <typename T>
  T GTR1(const T& NdotH, const T& a) const {
    T result = T(0);

    CERES_IF(a >= T(1)) { result = T(1 / M_PI); }
    CERES_ELSE {
      const T a2 = a * a;
      const T t = T(1) + (a2 - T(1)) * NdotH * NdotH;
      result = (a2 - T(1)) / (T(M_PI) * T(log(a2) * t));
    }
    CERES_ENDIF;
    return result;
  }

  template <typename T>
  T GTR2_aniso(const T& NdotH,
               const T& HdotX,
               const T& HdotY,
               const T& ax,
               const T& ay) const {
    return T(1) / (T(M_PI) * ax * ay *
                   sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
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
