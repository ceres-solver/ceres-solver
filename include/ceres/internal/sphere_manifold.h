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

#include "ceres/internal/householder_vector.h"

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

  // Map the delta from the minimum representation to the over parameterized
  // homogeneous vector. See B.2 p.25 equation (106) - (107) for more details.
  const double norm_delta_div_2 = 0.5 * norm_delta;
  const double sin_delta_by_delta =
      std::sin(norm_delta_div_2) / norm_delta_div_2;

  AmbientVector y(size_);
  y.head(size_ - 1) = 0.5 * sin_delta_by_delta * delta;
  y(size_ - 1) = std::cos(norm_delta_div_2);

  AmbientVector v(size_);
  double beta;

  // NOTE: The explicit template arguments are needed here because
  // ComputeHouseholderVector is templated and some versions of MSVC
  // have trouble deducing the type of v automatically.
  internal::ComputeHouseholderVector<Eigen::Map<const AmbientVector>,
                                     double,
                                     AmbientSpaceDimension>(x, &v, &beta);

  // Apply the delta update to remain on the sphere.
  x_plus_delta = x.norm() * internal::ApplyHouseholderVector(y, v, beta);

  return true;
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::PlusJacobian(
    const double* x_ptr, double* jacobian_ptr) const {
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<MatrixPlusJacobian> jacobian(jacobian_ptr, size_, size_ - 1);

  AmbientVector v(size_);
  double beta;

  // NOTE: The explicit template arguments are needed here because
  // ComputeHouseholderVector is templated and some versions of MSVC
  // have trouble deducing the type of v automatically.
  internal::ComputeHouseholderVector<Eigen::Map<const AmbientVector>,
                                     double,
                                     AmbientSpaceDimension>(x, &v, &beta);

  // The Jacobian is equal to J = 0.5 * H.leftCols(size_ - 1) where H is the
  // Householder matrix (H = I - beta * v * v').
  for (int i = 0; i < size_ - 1; ++i) {
    jacobian.col(i) = -0.5 * beta * v(i) * v;
    jacobian.col(i)(i) += 0.5;
  }
  jacobian *= x.norm();

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

  const AmbientVector hy =
      internal::ApplyHouseholderVector(y, v, beta) / x.norm();

  // Calculate y - x. See B.2 p.25 equation (108).
  double y_last = hy[size_ - 1];
  double hy_norm = hy.head(size_ - 1).norm();
  if (hy_norm == 0.0) {
    y_minus_x.setZero();
  } else {
    y_minus_x =
        2.0 * std::atan2(hy_norm, y_last) / hy_norm * hy.head(size_ - 1);
  }

  return true;
}

template <int AmbientSpaceDimension>
bool SphereManifold<AmbientSpaceDimension>::MinusJacobian(
    const double* x_ptr, double* jacobian_ptr) const {
  Eigen::Map<const AmbientVector> x(x_ptr, size_);
  Eigen::Map<MatrixMinusJacobian> jacobian(jacobian_ptr, size_ - 1, size_);

  AmbientVector v(size_);
  double beta;

  // NOTE: The explicit template arguments are needed here because
  // ComputeHouseholderVector is templated and some versions of MSVC
  // have trouble deducing the type of v automatically.
  internal::ComputeHouseholderVector<Eigen::Map<const AmbientVector>,
                                     double,
                                     AmbientSpaceDimension>(x, &v, &beta);

  // The Jacobian is equal to J = 2.0 * H.leftCols(size_ - 1) where H is the
  // Householder matrix (H = I - beta * v * v').
  for (int i = 0; i < size_ - 1; ++i) {
    jacobian.row(i) = -2.0 * beta * v(i) * v;
    jacobian.row(i)(i) += 2.0;
  }
  jacobian /= x.norm();

  return true;
}
}  // namespace ceres