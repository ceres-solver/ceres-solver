// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_MARGINALIZABLE_PARAMETERIZATION_H_
#define CERES_PUBLIC_MARGINALIZABLE_PARAMETERIZATION_H_

#include <memory>

#include "ceres/internal/autodiff.h"
#include "ceres/local_parameterization.h"
#include "ceres/rotation.h"

namespace ceres {

// The class MarginalizableParameterization defines the function Minus
// which is needed to compute difference between two parameter blocks
// that could be manifold objects.
class CERES_EXPORT MarginalizableParameterization {
 public:
  virtual ~MarginalizableParameterization();

  // Generalization of the subtraction operation,
  //
  //   delta = Minus(x_plus_delta, x)
  //
  // with the condition that Minux(x, x) = 0.
  virtual bool Minus(const double* x_plus_delta, const double* x,
                     double* delta) const = 0;

  // Must return a local size x global size matrix J(x,y) satisfying
  //
  // d/ddelta Minus(Plus(x, delta), y) = J(x, y) * d/ddelta Plus(x, delta).
  virtual bool ComputeMinusJacobian(const double* x,
                                    const double* y,
                                    double* jacobian) const = 0;
};

template <typename Functor, int kGlobalSize, int kLocalSize>
class CERES_EXPORT AutoDiffMarginalizableParameterization
    : public MarginalizableParameterization {
 public:
  AutoDiffMarginalizableParameterization() : functor_(new Functor()) {}

  // Takes ownership of functor
  explicit AutoDiffMarginalizableParameterization(Functor* functor)
      : functor_(functor) {}

  virtual ~AutoDiffMarginalizableParameterization() {}
  bool Minus(const double* x_plus_delta, const double* x,
             double* delta) const override {
    (*functor_)(x_plus_delta, x, delta);
  }

  bool ComputeMinusJacobian(const double* x, const double* y, double* jacobian) const override {
    double delta[kLocalSize];
    for (int i = 0; i < kLocalSize; ++i) {
      delta[i] = 0.0;
    }

    const double* parameter_ptrs[2] = {x, y};
    double* jacobian_ptrs[2] = {jacobian, NULL};

    return internal::AutoDifferentiate<
        kLocalSize, internal::StaticParameterDims<kGlobalSize, kGlobalSize>>(
        *functor_, parameter_ptrs, kLocalSize, delta, jacobian_ptrs);
  }

 private:
  std::unique_ptr<Functor> functor_;
};

// Some basic parameterizations

// Identity Parameterization:
//    Minus(x + delta, x) = delta
class CERES_EXPORT MarginalizableIdentityParameterization
    : public MarginalizableParameterization,
      public IdentityParameterization {
 public:
  explicit MarginalizableIdentityParameterization(int size);
  virtual ~MarginalizableIdentityParameterization() {}
  bool Minus(const double* x_plus_delta, const double* x,
                     double* delta) const override;

  bool ComputeMinusJacobian(const double* x,
    const double* y,
                                    double* jacobian) const override;

 private:
  const int size_;
};

// Functor needed to implement automatically differentiated Minus for
// quaternions.
struct QuaternionMinus {
  template <typename T>
  bool operator()(const T* x_plus_delta, const T* x, T* delta) const {
    T x_inverse[4];
    x_inverse[0] = x[0];
    x_inverse[1] = -x[1];
    x_inverse[2] = -x[2];
    x_inverse[3] = -x[3];

    T x_diff[4];
    QuaternionProduct(x_plus_delta, x_inverse, x_diff);

    if (x_diff[0] == T(1)) {
      delta[0] = x_diff[1];
      delta[1] = x_diff[2];
      delta[2] = x_diff[3];
    } else {
      const T cos_sq_delta = x_diff[0] * x_diff[0];
      const T sin_delta = sqrt(T(1.0) - cos_sq_delta);
      const T norm_delta = asin(sin_delta);
      const T delta_by_sin_delta = norm_delta / sin_delta;

      delta[0] = delta_by_sin_delta * x_diff[1];
      delta[1] = delta_by_sin_delta * x_diff[2];
      delta[2] = delta_by_sin_delta * x_diff[3];
    }
    return true;
  }
};

// Identity Parameterization:
//    Minus(x + delta, x) = delta
class CERES_EXPORT MarginalizableQuaternionParameterization
    : public AutoDiffMarginalizableParameterization<QuaternionMinus, 4, 3>,
      public QuaternionParameterization {
 public:
  virtual ~MarginalizableQuaternionParameterization() {}
};

}  // namespace ceres

//#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_PUBLIC_LOCAL_PARAMETERIZATION_H_