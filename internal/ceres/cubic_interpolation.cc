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

#include "ceres/cubic_interpolation.h"

#include <math.h>
#include "glog/logging.h"

namespace ceres {
namespace {

// Given samples from a function sampled at four equally spaced points,
//
//   p0 = f(-1)
//   p1 = f(0)
//   p2 = f(1)
//   p3 = f(2)
//
// Evaluate the cubic Hermite spline (also known as the Catmull-Rom
// spline) at a point x that lies in the interval [0, 1].
//
// This is also the interpolation kernel proposed by R. Keys, in:
//
// "Cubic convolution interpolation for digital image processing".
// IEEE Transactions on Acoustics, Speech, and Signal Processing
// 29 (6): 1153–1160.
//
// For the case of a = -0.5.
//
// For more details see
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
// http://en.wikipedia.org/wiki/Bicubic_interpolation
inline void CubicHermiteSpline(const double p0,
                               const double p1,
                               const double p2,
                               const double p3,
                               const double x,
                               double* f,
                               double* dfdx) {
  const double a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const double b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const double c = 0.5 * (-p0 + p2);
  const double d = p1;

  // Use Horner's rule to evaluate the function value and its
  // derivative.

  // f = ax^3 + bx^2 + cx + d
  if (f != NULL) {
    *f = d + x * (c + x * (b + x * a));
  }

  // dfdx = 3ax^2 + 2bx + c
  if (dfdx != NULL) {
    *dfdx = c + x * (2.0 * b + 3.0 * a * x);
  }
}

}  // namespace

CubicInterpolator::CubicInterpolator(const double* values, const int num_values)
    : values_(CHECK_NOTNULL(values)),
      num_values_(num_values) {
  CHECK_GT(num_values, 1);
}

bool CubicInterpolator::Evaluate(const double x,
                                 double* f,
                                 double* dfdx) const {
  if (x < 0 || x > num_values_ - 1) {
    LOG(ERROR) << "x =  " << x
               << " is not in the interval [0, " << num_values_ << "].";
    return false;
  }

  int n = floor(x);

  // Handle the case where the point sits exactly on the right boundary.
  if (n == num_values_ - 1) {
    n -= 1;
  }

  const double p1 = values_[n];
  const double p2 = values_[n + 1];
  const double p0 = (n > 0) ? values_[n - 1] : (2.0 * p1 - p2);
  const double p3 = (n < (num_values_ - 2)) ? values_[n + 2] : (2.0 * p2 - p1);
  CubicHermiteSpline(p0, p1, p2, p3, x - n, f, dfdx);
  return true;
}

BiCubicInterpolator::BiCubicInterpolator(const double* values,
                                         const int num_rows,
                                         const int num_cols)
    : values_(CHECK_NOTNULL(values)),
      num_rows_(num_rows),
      num_cols_(num_cols) {
  CHECK_GT(num_rows, 1);
  CHECK_GT(num_cols, 1);
}

bool BiCubicInterpolator::Evaluate(const double r,
                                   const double c,
                                   double* f,
                                   double* dfdr,
                                   double* dfdc) const {
  if (r < 0 || r > num_rows_ - 1 || c < 0 || c > num_cols_ - 1) {
    LOG(ERROR) << "(r, c) =  " << r << ", " << c
               << " is not in the square defined by [0, 0] "
               << " and [" << num_rows_ << ", " << num_cols_ << "]";
    return false;
  }

  int row = floor(r);
  // Handle the case where the point sits exactly on the bottom
  // boundary.
  if (row == num_rows_ - 1) {
    row -= 1;
  }

  int col = floor(c);
  // Handle the case where the point sits exactly on the right
  // boundary.
  if (col == num_cols_ - 1) {
    col -= 1;
  }

#define v(n, m) values_[n * num_cols_ + m]

  // BiCubic interpolation requires 16 values around the point being
  // evaluated.  We will use pij, to indicate the elements of the 4x4
  // array of values.
  //
  //          col
  //      p00 p01 p02 p03
  // row  p10 p11 p12 p13
  //      p20 p21 p22 p23
  //      p30 p31 p32 p33
  //
  // The point (r,c) being evaluated is assumed to lie in the square
  // defined by p11, p12, p22 and p21.

  // These four entries are guaranteed to be in the values_ array.
  const double p11 = v(row, col);
  const double p12 = v(row, col + 1);
  const double p21 = v(row + 1, col);
  const double p22 = v(row + 1, col + 1);

  // If we are in rows >= 1, then choose the element from the row - 1,
  // otherwise linearly interpolate from row and row + 1.
  const double p01 = (row > 0) ? v(row - 1, col) : 2 * p11 - p21;
  const double p02 = (row > 0) ? v(row - 1, col + 1) : 2 * p12 - p22;

  // If we are in row < num_rows_ - 2, then pick the element from the
  // row + 2, otherwise linearly interpolate from row and row + 1.
  const double p31 = (row < num_rows_ - 2) ? v(row + 2, col) : 2 * p21 - p11;
  const double p32 = (row < num_rows_ - 2) ? v(row + 2, col + 1) : 2 * p22 - p12;  // NOLINT

  // Same logic as above, applies to the columns instead of rows.
  const double p10 = (col > 0) ? v(row, col - 1) : 2 * p11 - p12;
  const double p20 = (col > 0) ? v(row + 1, col - 1) : 2 * p21 - p22;
  const double p13 = (col < num_cols_ - 2) ? v(row, col + 2) : 2 * p12 - p11;
  const double p23 = (col < num_cols_ - 2) ? v(row + 1, col + 2) : 2 * p22 - p21;  // NOLINT

  // The four corners of the block require a bit more care.  Let us
  // consider the evaluation of p00, the other four corners follow in
  // the same manner.
  //
  // There are four cases in which we need to evaluate p00.
  //
  // row > 0, col > 0 : v(row, col)
  // row = 0, col > 1 : Interpolate p10 & p20
  // row > 1, col = 0 : Interpolate p01 & p02
  // row = 0, col = 0 : Interpolate p10 & p20, or p01 & p02.
  double p00, p03;
  if (row > 0) {
    if (col > 0) {
      p00 = v(row - 1, col - 1);
    } else {
      p00 = 2 * p01 - p02;
    }

    if (col < num_cols_ - 2) {
      p03 = v(row - 1, col + 2);
    } else {
      p03 = 2 * p02 - p01;
    }
  } else {
    p00 = 2 * p10 - p20;
    p03 = 2 * p13 - p23;
  }

  double p30, p33;
  if (row < num_rows_ - 2) {
    if (col > 0) {
      p30 = v(row + 2, col - 1);
    } else {
      p30 = 2 * p31 - p32;
    }

    if (col < num_cols_ - 2) {
      p33 = v(row + 2, col + 2);
    } else {
      p33 = 2 * p32 - p31;
    }
  } else {
    p30 = 2 * p20 - p10;
    p33 = 2 * p23 - p13;
  }

  // Interpolate along each of the four rows, evaluating the function
  // value and the horizontal derivative in each row.
  double f0, f1, f2, f3;
  double df0dc, df1dc, df2dc, df3dc;
  CubicHermiteSpline(p00, p01, p02, p03, c - col, &f0, &df0dc);
  CubicHermiteSpline(p10, p11, p12, p13, c - col, &f1, &df1dc);
  CubicHermiteSpline(p20, p21, p22, p23, c - col, &f2, &df2dc);
  CubicHermiteSpline(p30, p31, p32, p33, c - col, &f3, &df3dc);

  // Interpolate vertically the interpolated value from each row and
  // compute the derivative along the columns.
  CubicHermiteSpline(f0, f1, f2, f3, r - row, f, dfdr);
  if (dfdc != NULL) {
    // Interpolate vertically the derivative along the columns.
    CubicHermiteSpline(df0dc, df1dc, df2dc, df3dc, r - row, dfdc, NULL);
  }

  return true;
#undef v
}

}  // namespace ceres
