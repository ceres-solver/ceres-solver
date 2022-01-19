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

#ifndef CERES_PUBLIC_AUTODIFF_MANIFOLD_H_
#define CERES_PUBLIC_AUTODIFF_MANIFOLD_H_

#include <memory>

#include "ceres/internal/autodiff.h"
#include "ceres/manifold.h"

namespace ceres {

// Create a Manifold with Jacobians computed via automatic differentiation. For
// more information on manifolds, see include/ceres/manifold.h
//
// To get an auto differentiated manifold, you must define
// a class with a templated Plus and Minus functions that computes
//
//   x_plus_delta = Plus(x, delta);
//   y_minus_x    = Minus(y, x);
//
// the template parameter T. The autodiff framework substitutes appropriate
// "Jet" objects for T in order to compute the derivative when necessary, but
// this is hidden, and you should write the function as if T were a scalar type
// (e.g. a double-precision floating point number).
//
// The function must write the computed value in the last argument (the only
// non-const one) and return true to indicate success.
//
// For example, Quaternions are a three dimensional manifold.
//
// Then given this struct, the auto differentiated local
// parameterization can now be constructed as
//
//   Manifold* manifold = new AutoDiffManifold<QuaternionFunctor, 4, 3>;
//                                                                |  |
//                                    Ambient Size ---------------+  |
//                                   Tangent Size -------------------+
//
// Where the QuaternionFunctor is implemented as follows (taken from
// autodiff_manifold_test.cc):
//
// struct QuaternionFunctor {
//   template <typename T>
//   bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
//     const T squared_norm_delta =
//         delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
//
//     T q_delta[4];
//     if (squared_norm_delta > T(0.0)) {
//       T norm_delta = sqrt(squared_norm_delta);
//       const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
//       q_delta[0] = cos(norm_delta);
//       q_delta[1] = sin_delta_by_delta * delta[0];
//       q_delta[2] = sin_delta_by_delta * delta[1];
//       q_delta[3] = sin_delta_by_delta * delta[2];
//     } else {
//       // We do not just use q_delta = [1,0,0,0] here because that is a
//       // constant and when used for automatic differentiation will
//       // lead to a zero derivative. Instead we take a first order
//       // approximation and evaluate it at zero.
//       q_delta[0] = T(1.0);
//       q_delta[1] = delta[0];
//       q_delta[2] = delta[1];
//       q_delta[3] = delta[2];
//     }
//
//     QuaternionProduct(q_delta, x, x_plus_delta);
//     return true;
//   }
//
//   template <typename T>
//   bool Minus(const T* y, const T* x, T* y_minus_x) const {
//     T minus_x[4] = {x[0], -x[1], -x[2], -x[3]};
//     T ambient_y_minus_x[4];
//     QuaternionProduct(y, minus_x, ambient_y_minus_x);
//     T u_norm = sqrt(ambient_y_minus_x[1] * ambient_y_minus_x[1] +
//                     ambient_y_minus_x[2] * ambient_y_minus_x[2] +
//                     ambient_y_minus_x[3] * ambient_y_minus_x[3]);
//     if (u_norm > 0.0) {
//       T theta = atan2(u_norm, ambient_y_minus_x[0]);
//       y_minus_x[0] = theta * ambient_y_minus_x[1] / u_norm;
//       y_minus_x[1] = theta * ambient_y_minus_x[2] / u_norm;
//       y_minus_x[2] = theta * ambient_y_minus_x[3] / u_norm;
//     } else {
//       // We do not use [0,0,0] here because even though the value part is
//       // a constant, the derivative part is not.
//       y_minus_x[0] = ambient_y_minus_x[1];
//       y_minus_x[1] = ambient_y_minus_x[2];
//       y_minus_x[2] = ambient_y_minus_x[3];
//     }
//     return true;
//   }
// };

template <typename Functor, int kAmbientSize, int kTangentSize>
class AutoDiffManifold : public Manifold {
 public:
  AutoDiffManifold() : functor_(std::make_unique<Functor>()) {}

  // Takes ownership of functor.
  explicit AutoDiffManifold(Functor* functor) : functor_(functor) {}

  ~AutoDiffManifold() override {}

  int AmbientSize() const override { return kAmbientSize; }
  int TangentSize() const override { return kTangentSize; }

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    return functor_->Plus(x, delta, x_plus_delta);
  }

  bool PlusJacobian(const double* x, double* jacobian) const override;

  bool Minus(const double* y,
             const double* x,
             double* y_minus_x) const override {
    return functor_->Minus(y, x, y_minus_x);
  }

  bool MinusJacobian(const double* x, double* jacobian) const override;

 private:
  std::unique_ptr<Functor> functor_;
};

namespace internal {

// The following two helper structs are needed to interface the Plus and Minus
// methods of the ManifoldFunctor with the automatic differentiation which
// expects a Functor with operator().
template <typename Functor>
struct PlusWrapper {
  PlusWrapper(const Functor& functor) : functor(functor) {}
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    return functor.Plus(x, delta, x_plus_delta);
  }
  const Functor& functor;
};

template <typename Functor>
struct MinusWrapper {
  MinusWrapper(const Functor& functor) : functor(functor) {}
  template <typename T>
  bool operator()(const T* y, const T* x, T* y_minus_x) const {
    return functor.Minus(y, x, y_minus_x);
  }
  const Functor& functor;
};
}  // namespace internal

template <typename Functor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<Functor, kAmbientSize, kTangentSize>::PlusJacobian(
    const double* x, double* jacobian) const {
  double zero_delta[kTangentSize];
  for (int i = 0; i < kTangentSize; ++i) {
    zero_delta[i] = 0.0;
  }

  double x_plus_delta[kAmbientSize];
  for (int i = 0; i < kAmbientSize; ++i) {
    x_plus_delta[i] = 0.0;
  }

  const double* parameter_ptrs[2] = {x, zero_delta};
  double* jacobian_ptrs[2] = {nullptr, jacobian};
  return internal::AutoDifferentiate<
      kAmbientSize,
      internal::StaticParameterDims<kAmbientSize, kTangentSize>>(
      internal::PlusWrapper(*functor_),
      parameter_ptrs,
      kAmbientSize,
      x_plus_delta,
      jacobian_ptrs);
}

template <typename Functor, int kAmbientSize, int kTangentSize>
bool AutoDiffManifold<Functor, kAmbientSize, kTangentSize>::MinusJacobian(
    const double* x, double* jacobian) const {
  double y_minus_x[kTangentSize];
  for (int i = 0; i < kTangentSize; ++i) {
    y_minus_x[i] = 0.0;
  }

  const double* parameter_ptrs[2] = {x, x};
  double* jacobian_ptrs[2] = {jacobian, nullptr};
  return internal::AutoDifferentiate<
      kTangentSize,
      internal::StaticParameterDims<kAmbientSize, kAmbientSize>>(
      internal::MinusWrapper(*functor_),
      parameter_ptrs,
      kAmbientSize,
      y_minus_x,
      jacobian_ptrs);
}

}  // namespace ceres

#endif  // CERES_PUBLIC_AUTODIFF_MANIFOLD_H_
