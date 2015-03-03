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

#include "ceres/internal/householder_matrix.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

void ComputeHouseholderVector(const Vector& x, Vector* v, double* beta) {
  CHECK_NOTNULL(beta);
  CHECK_NOTNULL(v);
  CHECK_GT(x.rows(), 1);
  CHECK_EQ(x.rows(), v->rows());

  double sigma = x.head(x.rows() - 1).squaredNorm();
  *v = x;
  (*v)(v->rows() - 1) = 1.0;

  *beta = 0.0;
  const double& x_pivot = x(x.rows() - 1);

  if (sigma <= std::numeric_limits<double>::epsilon()) {
    if (x_pivot < 0.0)
      *beta = 2.0;
  } else {
    const double mu = sqrt(x_pivot * x_pivot + sigma);
    double v_pivot = 1.0;

    if (x_pivot <= 0.0)
      v_pivot = x_pivot - mu;
    else
      v_pivot = -sigma / (x_pivot + mu);

    *beta = 2.0 * v_pivot * v_pivot / (sigma + v_pivot * v_pivot);

    v->head(v->rows() - 1) /= v_pivot;
  }
}

void ComputeHouseholderMatrix(const Vector& x, Matrix* H) {
  CHECK_NOTNULL(H);
  const int N = x.rows();

  Vector v(N);
  double beta;
  ComputeHouseholderVector(x, &v, &beta);

  H->resize(N, N);
  H->setIdentity();
  *H -= beta * v * v.transpose();
}

}  // namespace internal
}  // namespace ceres
