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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_CUBIC_INTERPOLATION_H_
#define CERES_PUBLIC_CUBIC_INTERPOLATION_H_

#include "ceres/internal/port.h"
#include "Eigen/Core"
#include "glog/logging.h"

namespace ceres {

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
// This is also the interpolation kernel (for the case of a = 0.5) as
// proposed by R. Keys, in:
//
// "Cubic convolution interpolation for digital image processing".
// IEEE Transactions on Acoustics, Speech, and Signal Processing
// 29 (6): 1153–1160.
//
// For more details see
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
// http://en.wikipedia.org/wiki/Bicubic_interpolation
//
// f if not NULL will contain the interpolated function values.
// dfdx if not NULL will contain the interpolated derivative values.
template <int kDataDimension>
void CubicHermiteSpline(const Eigen::Matrix<double, kDataDimension, 1>& p0,
                        const Eigen::Matrix<double, kDataDimension, 1>& p1,
                        const Eigen::Matrix<double, kDataDimension, 1>& p2,
                        const Eigen::Matrix<double, kDataDimension, 1>& p3,
                        const double x,
                        double* f,
                        double* dfdx) {
  DCHECK_GE(x, 0.0);
  DCHECK_LE(x, 1.0);
  typedef Eigen::Matrix<double, kDataDimension, 1> VType;
  const VType a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const VType b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const VType c = 0.5 * (-p0 + p2);
  const VType d = p1;

  // Use Horner's rule to evaluate the function value and its
  // derivative.

  // f = ax^3 + bx^2 + cx + d
  if (f != NULL) {
    Eigen::Map<VType>(f, kDataDimension) = d + x * (c + x * (b + x * a));
  }

  // dfdx = 3ax^2 + 2bx + c
  if (dfdx != NULL) {
    Eigen::Map<VType>(dfdx, kDataDimension) = c + x * (2.0 * b + 3.0 * a * x);
  }
}

// Given as input a one dimensional array like object, which provides
// the following interface.
//
//   struct Array {
//     enum { DATA_DIMENSION = 2; };
//     void GetValue(int n, double* f) const;
//     int NumValues() const;
//   };
//
// Where, GetValue gives us the value of a function f (possibly vector
// valued) on the integers:
//
//   [0, ..., NumValues() - 1].
//
// and the enum DATA_DIMENSION indicates the dimensionality of the
// function being interpolated. For example if you are interpolating a
// color image with three channels (Red, Green & Blue), then
// DATA_DIMENSION = 3.
//
// CubicInterpolator uses cubic Hermite splines to produce a smooth
// approximation to it that can be used to evaluate the f(x) and f'(x)
// at any real valued point in the interval:
//
//   [0, NumValues() - 1].
//
// For more details on cubic interpolation see
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
//
// Example usage:
//
//  const double data[] = {1.0, 2.0, 5.0, 6.0};
//  Array1D<double, 1> array(x, 4);
//  CubicInterpolator<Array1D<double, 1> > interpolator(array);
//  double f, dfdx;
//  CHECK(interpolator.Evaluator(1.5, &f, &dfdx));
template<typename Array>
class CERES_EXPORT CubicInterpolator {
 public:
  explicit CubicInterpolator(const Array& array)
      : array_(array) {
    CHECK_GT(array.NumValues(), 1);
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Array::DATA_DIMENSION, 1);
  }

  bool Evaluate(double x, double* f, double* dfdx) const {
    const int num_values = array_.NumValues();
    if (x < 0 || x > num_values - 1) {
      LOG(ERROR) << "x =  " << x
                 << " is not in the interval [0, " << num_values - 1 << "].";
      return false;
    }

    int n = floor(x);
    // Deal with the case where the point sits exactly on the right
    // boundary.
    if (n == num_values - 1) {
      n -= 1;
    }

    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> p0, p1, p2, p3;

    // The point being evaluated is now expected to lie in the
    // internal corresponding to p1 and p2.
    array_.GetValue(n, p1.data());
    array_.GetValue(n + 1, p2.data());

    // If we are at n >=1, the choose the element at n - 1, otherwise
    // linearly interpolate from p1 and p2.
    if (n > 0) {
      array_.GetValue(n - 1, p0.data());
    } else {
      p0 = 2 * p1 - p2;
    }

    // If we are at n < num_values_ - 2, then choose the element n +
    // 2, otherwise linearly interpolate from p1 and p2.
    if (n < num_values - 2) {
      array_.GetValue(n + 2, p3.data());
    } else {
      p3 = 2 * p2 - p1;
    }

    CubicHermiteSpline<Array::DATA_DIMENSION>(p0, p1, p2, p3, x - n, f, dfdx);

    return true;
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  bool Evaluate(const double& x, double* f) const {
    return Evaluate(x, f, NULL);
  }

  template<typename JetT> bool Evaluate(const JetT& x, JetT* f) const {
    double fx[Array::DATA_DIMENSION], dfdx[Array::DATA_DIMENSION];
    if (!Evaluate(x.a, fx, dfdx)) {
      return false;
    }

    for (int i = 0; i < Array::DATA_DIMENSION; ++i) {
      f[i].a = fx[i];
      f[i].v = dfdx[i] * x.v;
    }
    return true;
  }

  int NumValues() const { return array_.NumValues(); }

private:
  const Array& array_;
};

// Given as input a two dimensional array like object, which provides
// the following interface:
//
//   struct Array {
//     enum { DATA_DIMENSION = 1 };
//     void GetValue(int row, int col, double* f) const;
//     int NumRows() const;
//     int NumCols() const;
//   };
//
// Where, GetValue gives us the value of a function f (possibly vector
// valued) on the integer grid:
//
//   [0, ..., NumRows() - 1] x [0, ..., NumCols() - 1]
//
// and the enum DATA_DIMENSION indicates the dimensionality of the
// function being interpolated. For example if you are interpolating a
// color image with three channels (Red, Green & Blue), then
// DATA_DIMENSION = 3.
//
// BiCubicInterpolator uses the cubic convolution interpolation
// algorithm of R. Keys, to produce a smooth approximation to it that
// can be used to evaluate the f(r,c), df(r, c)/dr and df(r,c)/dc at
// any real valued point in the quad:
//
//   [0, NumRows() - 1] x [0, NumCols() - 1]
//
// For more details on the algorithm used here see:
//
// "Cubic convolution interpolation for digital image processing".
// Robert G. Keys, IEEE Trans. on Acoustics, Speech, and Signal
// Processing 29 (6): 1153–1160, 1981.
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
// http://en.wikipedia.org/wiki/Bicubic_interpolation
//
// Example usage:
//
// const double data[] = {1.0, 3.0, -1.0, 4.0,
//                         3.6, 2.1,  4.2, 2.0,
//                        2.0, 1.0,  3.1, 5.2};
//  Array2D<double, 1>  array(data, 3, 4);
//  BiCubicInterpolator<Array2D<double, 1> > interpolator(array);
//  double f, dfdr, dfdc;
//  CHECK(interpolator.Evaluate(1.2, 2.5, &f, &dfdr, &dfdc));

template<typename Array>
class CERES_EXPORT BiCubicInterpolator {
 public:
  explicit BiCubicInterpolator(const Array& array)
      : array_(array) {
    CHECK_GT(array.NumRows(), 1);
    CHECK_GT(array.NumCols(), 1);
    // The + casts the enum into an int before doing the
    // comparison. It is needed to prevent
    // "-Wunnamed-type-template-args" related errors.
    CHECK_GE(+Array::DATA_DIMENSION, 1);
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  bool Evaluate(double r, double c,
                double* f, double* dfdr, double* dfdc) const {
    const int num_rows = array_.NumRows();
    const int num_cols = array_.NumCols();

    if (r < 0 || r > num_rows - 1 || c < 0 || c > num_cols - 1) {
      LOG(ERROR) << "(r, c) =  (" << r << ", " << c << ")"
                 << " is not in the square defined by [0, 0] "
                 << " and [" << num_rows - 1 << ", " << num_cols - 1 << "]";
      return false;
    }

    int row = floor(r);
    // Handle the case where the point sits exactly on the bottom
    // boundary.
    if (row == num_rows - 1) {
      row -= 1;
    }

    int col = floor(c);
    // Handle the case where the point sits exactly on the right
    // boundary.
    if (col == num_cols - 1) {
      col -= 1;
    }

    // BiCubic interpolation requires 16 values around the point being
    // evaluated.  We will use pij, to indicate the elements of the
    // 4x4 array of values.
    //
    //          col
    //      p00 p01 p02 p03
    // row  p10 p11 p12 p13
    //      p20 p21 p22 p23
    //      p30 p31 p32 p33
    //
    // The point (r,c) being evaluated is assumed to lie in the square
    // defined by p11, p12, p22 and p21.

    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> p00, p01, p02, p03;
    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> p10, p11, p12, p13;
    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> p20, p21, p22, p23;
    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> p30, p31, p32, p33;

    array_.GetValue(row,     col,     p11.data());
    array_.GetValue(row,     col + 1, p12.data());
    array_.GetValue(row + 1, col,     p21.data());
    array_.GetValue(row + 1, col + 1, p22.data());

    // If we are in rows >= 1, then choose the element from the row - 1,
    // otherwise linearly interpolate from row and row + 1.
    if (row > 0) {
      array_.GetValue(row - 1, col,     p01.data());
      array_.GetValue(row - 1, col + 1, p02.data());
    } else {
      p01 = 2 * p11 - p21;
      p02 = 2 * p12 - p22;
    }

    // If we are in row < num_rows - 2, then pick the element from the
    // row + 2, otherwise linearly interpolate from row and row + 1.
    if (row < num_rows - 2) {
      array_.GetValue(row + 2, col,     p31.data());
      array_.GetValue(row + 2, col + 1, p32.data());
    } else {
      p31 = 2 * p21 - p11;
      p32 = 2 * p22 - p12;
    }

    // Same logic as above, applies to the columns instead of rows.
    if (col > 0) {
      array_.GetValue(row,     col - 1, p10.data());
      array_.GetValue(row + 1, col - 1, p20.data());
    } else {
      p10 = 2 * p11 - p12;
      p20 = 2 * p21 - p22;
    }

    if (col < num_cols - 2) {
      array_.GetValue(row,     col + 2, p13.data());
      array_.GetValue(row + 1, col + 2, p23.data());
    } else {
      p13 = 2 * p12 - p11;
      p23 = 2 * p22 - p21;
    }

    // The four corners of the block require a bit more care.  Let us
    // consider the evaluation of p00, the other three corners follow
    // in the same manner.
    //
    // There are four cases in which we need to evaluate p00.
    //
    // row > 0, col > 0 : v(row, col)
    // row = 0, col > 0 : Interpolate p10 & p20
    // row > 0, col = 0 : Interpolate p01 & p02
    // row = 0, col = 0 : Interpolate p10 & p20, or p01 & p02.
    if (row > 0) {
      if (col > 0) {
        array_.GetValue(row - 1, col - 1, p00.data());
      } else {
        p00 = 2 * p01 - p02;
      }

      if (col < num_cols - 2) {
        array_.GetValue(row - 1, col + 2, p03.data());
      } else {
        p03 = 2 * p02 - p01;
      }
    } else {
      p00 = 2 * p10 - p20;
      p03 = 2 * p13 - p23;
    }

    if (row < num_rows - 2) {
      if (col > 0) {
        array_.GetValue(row + 2, col - 1, p30.data());
      } else {
        p30 = 2 * p31 - p32;
      }

      if (col < num_cols - 2) {
        array_.GetValue(row + 2, col + 2, p33.data());
      } else {
        p33 = 2 * p32 - p31;
      }
    } else {
      p30 = 2 * p20 - p10;
      p33 = 2 * p23 - p13;
    }

    // Interpolate along each of the four rows, evaluating the function
    // value and the horizontal derivative in each row.
    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> f0, f1, f2, f3;
    Eigen::Matrix<double, Array::DATA_DIMENSION, 1> df0dc, df1dc, df2dc, df3dc;

    CubicHermiteSpline<Array::DATA_DIMENSION>(p00, p01, p02, p03, c - col,
                                              f0.data(), df0dc.data());
    CubicHermiteSpline<Array::DATA_DIMENSION>(p10, p11, p12, p13, c - col,
                                              f1.data(), df1dc.data());
    CubicHermiteSpline<Array::DATA_DIMENSION>(p20, p21, p22, p23, c - col,
                                              f2.data(), df2dc.data());
    CubicHermiteSpline<Array::DATA_DIMENSION>(p30, p31, p32, p33, c - col,
                                              f3.data(), df3dc.data());

    // Interpolate vertically the interpolated value from each row and
    // compute the derivative along the columns.
    CubicHermiteSpline<Array::DATA_DIMENSION>(f0, f1, f2, f3, r - row, f, dfdr);
    if (dfdc != NULL) {
      // Interpolate vertically the derivative along the columns.
      CubicHermiteSpline<Array::DATA_DIMENSION>(df0dc, df1dc, df2dc, df3dc,
                                                r - row, dfdc, NULL);
    }

    return true;
  }

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  bool Evaluate(const double& r, const double& c, double* f) const {
    return Evaluate(r, c, f, NULL, NULL);
  }

  template<typename JetT> bool Evaluate(const JetT& r,
                                        const JetT& c,
                                        JetT* f) const {
    double frc[Array::DATA_DIMENSION];
    double dfdr[Array::DATA_DIMENSION];
    double dfdc[Array::DATA_DIMENSION];
    if (!Evaluate(r.a, c.a, frc, dfdr, dfdc)) {
      return false;
    }

    for (int i = 0; i < Array::DATA_DIMENSION; ++i) {
      f[i].a = frc[i];
      f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
    }

    return true;
  }

  int NumRows() const { return array_.NumRows(); }
  int NumCols() const { return array_.NumCols(); }

 private:
  const Array& array_;
};

// An object that implements the one dimensional array like object
// needed by the CubicInterpolator where the source of the function
// values is an array of type T.
//
// The function being provided can be vector valued, in which case
// kDataDimension > 1. The dimensional slices of the function maybe
// interleaved, or they maybe stacked, i.e, if the function has
// kDataDimension = 2, if kInterleaved = true, then it is stored as
//
//   f01, f02, f11, f12 ....
//
// and if kInterleaved = false, then it is stored as
//
//  f01, f11, .. fn1, f02, f12, .. , fn2
template <typename T, int kDataDimension = 1, bool kInterleaved = true>
struct Array1D {
  enum { DATA_DIMENSION = kDataDimension };

  Array1D(const T* data, const int num_values)
      : data_(data), num_values_(num_values) {
  }

  void GetValue(const int n, double* f) const {
    DCHECK_GE(n, 0);
    DCHECK_LT(n, num_values_);

    for (int i = 0; i < kDataDimension; ++i) {
      if (kInterleaved) {
        f[i] = static_cast<double>(data_[kDataDimension * n + i]);
      } else {
        f[i] = static_cast<double>(data_[i * num_values_ + n]);
      }
    }
  }

  int NumValues() const { return num_values_; }

 private:
  const T* data_;
  const int num_values_;
};

// An object that implements the two dimensional array like object
// needed by the BiCubicInterpolator where the source of the function
// values is an array of type T.
//
// The function being provided can be vector valued, in which case
// kDataDimension > 1. The data maybe stored in row or column major
// format and the various dimensional slices of the function maybe
// interleaved, or they maybe stacked, i.e, if the function has
// kDataDimension = 2, is stored in row-major format and if
// kInterleaved = true, then it is stored as
//
//   f001, f002, f011, f012, ...
//
// A commonly occuring example are color images (RGB) where the three
// channels are stored interleaved.
//
// If kInterleaved = false, then it is stored as
//
//  f001, f011, ..., fnm1, f002, f012, ...
template <typename T,
          int kDataDimension = 1,
          bool kRowMajor = true,
          bool kInterleaved = true>
struct Array2D {
  enum { DATA_DIMENSION = kDataDimension };

  Array2D(const T* data, const int num_rows, const int num_cols)
      : data_(data), num_rows_(num_rows), num_cols_(num_cols) {
    CHECK_GE(kDataDimension, 1);
  }

  void GetValue(const int r, const int c, double* f) const {
    DCHECK_GE(r, 0);
    DCHECK_LT(r, num_rows_);
    DCHECK_GE(c, 0);
    DCHECK_LT(c, num_cols_);

    const int n = (kRowMajor) ? num_cols_ * r + c : num_rows_ * c + r;
    for (int i = 0; i < kDataDimension; ++i) {
      if (kInterleaved) {
        f[i] = static_cast<double>(data_[kDataDimension * n + i]);
      } else {
        f[i] = static_cast<double>(data_[i * (num_rows_ * num_cols_) + n]);
      }
    }
  }

  int NumRows() const { return num_rows_; }
  int NumCols() const { return num_cols_; }

 private:
  const T* data_;
  const int num_rows_;
  const int num_cols_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_CUBIC_INTERPOLATOR_H_
