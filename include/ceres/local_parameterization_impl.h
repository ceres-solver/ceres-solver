// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: vitus@google.com (Michael Vitus)

#ifndef CERES_PUBLIC_LOCAL_PARAMETERIZATION_IMPL_H_
#define CERES_PUBLIC_LOCAL_PARAMETERIZATION_IMPL_H_

#include "ceres/internal/eigen.h"
#include "ceres/internal/householder_matrix.h"
#include "glog/logging.h"

namespace ceres {

template <int N>
HomogeneousVectorParameterization<N>::HomogeneousVectorParameterization() {
  CHECK_GT(N, 1) << "The size of the homogeneous vector needs to be greater "
                 << "than 1.";
}

template <int N>
bool HomogeneousVectorParameterization<N>::Plus(const double* x,
                                                const double* delta,
                                                double* x_plus_delta) const {
  ConstVectorRef x_ref(x, N);
  ConstVectorRef delta_ref(delta, N - 1);
  VectorRef x_plus_delta_ref(x_plus_delta, N);

  CHECK_LT(abs(x_ref.squaredNorm() - 1.0),
           std::numeric_limits<double>::epsilon())
      << "The homogeneous vector x must be unit norm. ||x||_2 = "
      << x_ref.norm();

  const double squared_norm_delta = delta_ref.squaredNorm();

  Vector y(N);
  if (squared_norm_delta > 0.0) {
    const double norm_delta = sqrt(squared_norm_delta);
    const double norm_delta_div_2 = 0.5 * norm_delta;
    const double sin_delta_by_delta = sin(norm_delta_div_2) /
        norm_delta_div_2;

    y.head<N - 1>() = 0.5 * sin_delta_by_delta * delta_ref;
    y(N - 1) = cos(norm_delta_div_2);
  } else {
    y.head<N - 1>() = 0.5 * delta_ref;
    y(N - 1) = 1.0;
    y.normalize();
  }

  Vector v(N);
  double beta;
  internal::ComputeHouseholderVector(x_ref, &v, &beta);

  x_plus_delta_ref = y - beta * v * (v.transpose() * y);

  return true;
}

template <int N>
bool HomogeneousVectorParameterization<N>::ComputeJacobian(
    const double* x, double* jacobian) const {
  ConstVectorRef x_ref(x, N);
  MatrixRef jacobian_ref(jacobian, N, N - 1);

  Matrix H(N, N);
  internal::ComputeHouseholderMatrix(x_ref, &H);

  jacobian_ref = 0.5 * H.leftCols<N - 1>();

  return true;
}
}  // namespace ceres

#endif  // CERES_PUBLIC_LOCAL_PARAMETERIZATION_IMPL_H_
