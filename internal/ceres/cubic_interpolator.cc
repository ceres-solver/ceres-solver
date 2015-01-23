// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// This implementation was inspired by the description at
// http://www.paulinternet.nl/?page=bicubic
#include "ceres/cubic_interpolator.h"

#include <math.h>

namespace ceres {
namespace {

inline void CatmullRomSpline(const double p0,
                             const double p1,
                             const double p2,
                             const double p3,
                             const double x,
                             double* f,
                             double* fx) {
  // Use Horner's rule to evaluate the function value and its
  // derivative.
  if (f != NULL) {
    *f =
        p1 + 0.5 * x * (
            p2 - p0 + x * (
                2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (
                    3.0 * (p1 - p2) + p3 - p0)));
  }

  if (fx != NULL) {
    *fx =
        0.5 * (
            p2 - p0 + x * (
                2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) + x * 3.0 * (
                    3.0 * (p1 - p2) + p3 - p0)));
  }
}

}  // namespace

CubicInterpolator1::CubicInterpolator1(const int num_values,
                                       const double* values)
    : num_values_(num_values),
      values_(CHECK_NOTNULL(values)) {
  CHECK_GT(num_values, 1);
}

bool CubicInterpolator1::Evaluate(double x,
                                  double* value,
                                  double* value_x) const {
  if (x < 0 || x > num_values_ - 1) {
    return false;
  }

  int n = floor(x);

  // Handle the case where the point sits exactly on the right boundary.
  if (n == num_values_ - 1) {
    n -= 1;
  }

  const double p1 = values_[n];
  const double p2 = values_[n + 1];
  const double p0 = (n > 0) ? value[n - 1] : (2.0 * p1 - p2);
  const double p3 = (n < (num_values_ - 2)) ? values_[n + 2] : (2.0 * p2 - p1);
  CatmullRomSpline(p0, p1, p2, p3, x - n, value, value_x);
  return true;
}

}  // namespace ceres
