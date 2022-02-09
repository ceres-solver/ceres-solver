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
// Author: vitus@google.com (Mike Vitus)
//         jodebo_beck@gmx.de (Johannes Beck)

#ifndef CERES_PUBLIC_INTERNAL_SPHERE_MANIFOLD_H_
#define CERES_PUBLIC_INTERNAL_SPHERE_MANIFOLD_H_

#include "ceres/internal/householder_vector.h"
#include "ceres/internal/sphere_manifold_functions.h"

namespace ceres {

template <int AmbientSpaceDimension>
SphereManifold<AmbientSpaceDimension>::SphereManifold()
    : size_{AmbientSpaceDimension} {
  static_assert(
      AmbientSpaceDimension != Eigen::Dynamic,
      "The size is set to dynamic. Please call the constructor with a size.");
}

template <int AmbientSpaceDimension>
SphereManifold<AmbientSpaceDimension>::SphereManifold(int size) : size_{size} {
  if (AmbientSpaceDimension != Eigen::Dynamic) {
    CHECK_EQ(AmbientSpaceDimension, size)
        << "Specified size by template parameter differs from the supplied "
           "one.";
  } else {
    CHECK_GT(size_, 1)
        << "The size of the manifold needs to be greater than 1.";
  }
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::Plus(
    const double* x_ptr,
    const double* delta_ptr,
    double* x_plus_delta_ptr) const {
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<const TangentVector> delta(delta_ptr, size_ - 1);
  Eigen::Map<AmbientVector> x_plus_delta(x_plus_delta_ptr, size_);

  const double norm_delta = delta.norm();

  if (norm_delta == 0.0) {
    x_plus_delta = x;
    return true;
  }

  AmbientVector v(size_);
  double beta;

  // NOTE: The explicit template arguments are needed here because
  // ComputeHouseholderVector is templated and some versions of MSVC
  // have trouble deducing the type of v automatically.
  internal::ComputeHouseholderVector<Eigen::Map<const AmbientVector>,
                                     double,
                                     AmbientSpaceDimension>(x, &v, &beta);

  internal::ComputeSphereManifoldPlus(
      v, beta, x, delta, norm_delta, &x_plus_delta);

  return true;
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::PlusJacobian(
    const double* x_ptr, double* jacobian_ptr) const {
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<MatrixPlusJacobian> jacobian(jacobian_ptr, size_, size_ - 1);
  internal::ComputeSphereManifoldPlusJacobian(x, &jacobian);

  return true;
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::Minus(const double* y_ptr,
                                                  const double* x_ptr,
                                                  double* y_minus_x_ptr) const {
  AmbientVector y = Eigen::Map<const AmbientVector>(y_ptr, size_);
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<TangentVector> y_minus_x(y_minus_x_ptr, size_ - 1);

  // Apply hoseholder transformation.
  AmbientVector v(size_);
  double beta;

  // NOTE: The explicit template arguments are needed here because
  // ComputeHouseholderVector is templated and some versions of MSVC
  // have trouble deducing the type of v automatically.
  internal::ComputeHouseholderVector<Eigen::Map<const AmbientVector>,
                                     double,
                                     AmbientSpaceDimension>(x, &v, &beta);
  internal::ComputeSphereManifoldMinus(v, beta, x, y, &y_minus_x);
  return true;
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::MinusJacobian(
    const double* x_ptr, double* jacobian_ptr) const {
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<MatrixMinusJacobian> jacobian(jacobian_ptr, size_ - 1, size_);

  internal::ComputeSphereManifoldMinusJacobian(x, &jacobian);
  return true;
}
}  // namespace ceres

#endif
