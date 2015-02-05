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

#include "ceres/internal/port.h"
#include "ceres/internal/eigen.h"
#include "glog/logging.h"

#ifndef CERES_PUBLIC_CUBIC_INTERPOLATION_H_
#define CERES_PUBLIC_CUBIC_INTERPOLATION_H_

namespace ceres {

template <int kDimension>
void CubicHermiteSpline(const Eigen::Matrix<double, kDimension, 1>& p0,
                        const Eigen::Matrix<double, kDimension, 1>& p1,
                        const Eigen::Matrix<double, kDimension, 1>& p2,
                        const Eigen::Matrix<double, kDimension, 1>& p3,
                        const double x,
                        double* f,
                        double* dfdx) {
  typedef Eigen::Matrix<double, kDimension, 1> VType;
  const VType a = 0.5 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3);
  const VType b = 0.5 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3);
  const VType c = 0.5 * (-p0 + p2);
  const VType d = p1;

  // Use Horner's rule to evaluate the function value and its
  // derivative.

  // f = ax^3 + bx^2 + cx + d
   if (f != NULL) {
     Eigen::Map<VType>(f, kDimension) = d + x * (c + x * (b + x * a));
  }

  // dfdx = 3ax^2 + 2bx + c
  if (dfdx != NULL) {
    Eigen::Map<VType>(dfdx, kDimension) = c + x * (2.0 * b + 3.0 * a * x);
  }
}

template <typename T, int kDimension, bool kInterleaved = true>
struct Array1DProvider {
  Array1DProvider(const T* data, const int num_values)
      : data_(data), num_values_(num_values) {
  }

  bool GetValue(const int n, double* values) const {
    if (n < 0 || n > num_values_ - 1) {
      return false;
    }

    for (int i = 0; i < kDimension; ++i) {
      if (kInterleaved) {
        values[i] = static_cast<double>(data_[kDimension * n + i]);
      } else {
        values[i] = static_cast<double>(data_[i * num_values_ + n]);
      }
    }
    return true;
  }

  int NumValues() const { return num_values_; }

 private:
  const T* data_;
  const int num_values_;
};

template <typename T, int kDimension, bool kRowMajor = true, bool kInterleaved = true>
struct Array2DProvider {
  Array2DProvider(const T* data, const int num_rows, const int num_cols)
      : data_(data), num_rows_(num_rows), num_cols_(num_cols) {
  }

  bool GetValue(const int r, const int c, double* values) const {
    if (r < 0 || r > num_rows_ - 1 || c < 0 || c > num_cols_ - 1) {
      return false;
    }

    const int n = (kRowMajor) ? num_cols_ * r + c : num_rows_ * c + r;
    for (int i = 0; i < kDimension; ++i) {
      if (kInterleaved) {
        values[i] = static_cast<double>(data_[kDimension * n + i]);
      } else {
        values[i] = static_cast<double>(data_[i * (num_rows_ * num_cols_) + n]);
      }
    }
    return true;
  }

  int NumRows() const { return num_rows_; }
  int NumCols() const { return num_cols_; }

 private:
  const T* data_;
  const int num_rows_;
  const int num_cols_;
};

// Array1DProvider - T, Interleaved or not.
// Array2DProvider - T, Row/Col Major, Interleaved or not.

template<typename DataProvider, int kDimension>
class CERES_EXPORT CubicInterpolator {
 public:
  explicit CubicInterpolator(const DataProvider& data)
      : data_(data) {
    CHECK_GT(data.NumValues(), 1);
  }

  bool Evaluate(double x, double* f, double* dfdx) const {
    const int num_values = data_.NumValues();
    if (x < 0 || x > num_values - 1) {
      LOG(ERROR) << "x =  " << x
                 << " is not in the interval [0, " << num_values - 1 << "].";
      return false;
    }

    int n = floor(x);
    // Handle the case where the point sits exactly on the right boundary.
    if (n == num_values - 1) {
      n -= 1;
    }

    Eigen::Matrix<double, kDimension, 1> p0, p1, p2, p3;
    data_.GetValue(n, p1.data());
    data_.GetValue(n + 1, p2.data());
    if (n > 0) {
      data_.GetValue(n - 1, p0.data());
    } else {
      p0 = 2 * p1 - p2;
    }

    if (n < num_values - 2) {
      data_.GetValue(n + 2, p3.data());
    } else {
      p3 = 2 * p2 - p1;
    }

    CubicHermiteSpline(p0, p1, p2, p3, x - n, f, dfdx);
    return true;
  }

  bool Evaluate(const double& x, double* f) const {
    return Evaluate(x, f, NULL);
  }

  template<typename JetT> bool Evaluate(const JetT& x, JetT* f) const {
    double dfdx[kDimension];
    double fx[kDimension];
    if (!Evaluate(x.a, fx, dfdx)) {
      return false;
    }

    for (int i = 0; i < kDimension; ++i) {
      f[i].a = fx[i];
      f[i].v = dfdx[i] * x.v;
    }

    return true;
  }

  int NumValues() const { return data_.NumValues(); }

private:
  const DataProvider& data_;
};

template<typename DataProvider, int kDimension>
class CERES_EXPORT BiCubicInterpolator {
 public:
  BiCubicInterpolator(const DataProvider& data)
      : data_(data) {
    CHECK_GT(data.NumRows(), 1);
    CHECK_GT(data.NumCols(), 1);
  }

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  bool Evaluate(double r, double c,
                double* f, double* dfdr, double* dfdc) const {
    const int num_rows = data_.NumRows();
    const int num_cols = data_.NumCols();

    if (r < 0 || r > num_rows - 1 || c < 0 || c > num_cols - 1) {
      LOG(ERROR) << "(r, c) =  " << r << ", " << c
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

    Eigen::Matrix<double, kDimension, 1> p00, p01, p02, p03;
    Eigen::Matrix<double, kDimension, 1> p10, p11, p12, p13;
    Eigen::Matrix<double, kDimension, 1> p20, p21, p22, p23;
    Eigen::Matrix<double, kDimension, 1> p30, p31, p32, p33;

    data_.GetValue(row,     col,     p11.data());
    data_.GetValue(row,     col + 1, p12.data());
    data_.GetValue(row + 1, col,     p21.data());
    data_.GetValue(row + 1, col + 1, p22.data());

    // If we are in rows >= 1, then choose the element from the row - 1,
    // otherwise linearly interpolate from row and row + 1.
    if (row > 0) {
      data_.GetValue(row - 1, col,     p01.data());
      data_.GetValue(row - 1, col + 1, p02.data());
    } else {
      p01 = 2 * p11 - p21;
      p02 = 2 * p12 - p22;
    }

    // If we are in row < num_rows - 2, then pick the element from
    // the row + 2, otherwise linearly interpolate from row and row +
    // 1.
    if (row < num_rows - 2) {
      data_.GetValue(row + 2, col,     p31.data());
      data_.GetValue(row + 2, col + 1, p32.data());
    } else {
      p31 = 2 * p21 - p22;
      p32 = 2 * p22 - p12;
    }

    // Same logic as above, applies to the columns instead of rows.
    if (col > 0) {
      data_.GetValue(row,     col - 1, p10.data());
      data_.GetValue(row + 1, col - 1, p20.data());
    } else {
      p10 = 2 * p11 - p12;
      p20 = 2 * p21 - p22;
    }

    if (col < num_cols - 2) {
      data_.GetValue(row,     col + 2, p13.data());
      data_.GetValue(row + 1, col + 2, p23.data());
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
    // row = 0, col > 1 : Interpolate p10 & p20
    // row > 1, col = 0 : Interpolate p01 & p02
    // row = 0, col = 0 : Interpolate p10 & p20, or p01 & p02.
    if (row > 0) {
      if (col > 0) {
        data_.GetValue(row - 1, col - 1, p00.data());
      } else {
        p00 = 2 * p01 - p02;
      }

      if (col < num_cols - 2) {
        data_.GetValue(row - 1, col + 2, p03.data());
      } else {
        p03 = 2 * p02 - p01;
      }
    } else {
      p00 = 2 * p10 - p20;
      p03 = 2 * p13 - p23;
    }

    if (row < num_rows - 2) {
      if (col > 0) {
        data_.GetValue(row + 2, col - 1, p30.data());
      } else {
        p30 = 2 * p31 - p32;
      }

      if (col < num_cols - 2) {
        data_.GetValue(row + 2, col + 2, p33.data());
      } else {
        p33 = 2 * p32 - p31;
      }
    } else {
      p30 = 2 * p20 - p10;
      p33 = 2 * p23 - p13;
    }

    // Interpolate along each of the four rows, evaluating the function
    // value and the horizontal derivative in each row.
    Eigen::Matrix<double, kDimension, 1> f0, f1, f2, f3;
    Eigen::Matrix<double, kDimension, 1> df0dc, df1dc, df2dc, df3dc;
    CubicHermiteSpline(p00, p01, p02, p03, c - col, f0.data(), df0dc.data());
    CubicHermiteSpline(p10, p11, p12, p13, c - col, f1.data(), df1dc.data());
    CubicHermiteSpline(p20, p21, p22, p23, c - col, f2.data(), df2dc.data());
    CubicHermiteSpline(p30, p31, p32, p33, c - col, f3.data(), df3dc.data());

    // Interpolate vertically the interpolated value from each row and
    // compute the derivative along the columns.
    CubicHermiteSpline(f0, f1, f2, f3, r - row, f, dfdr);
    if (dfdc != NULL) {
      // Interpolate vertically the derivative along the columns.
      CubicHermiteSpline(df0dc, df1dc, df2dc, df3dc, r - row, dfdc, NULL);
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
    double frc[kDimension], dfdr[kDimension], dfdc[kDimension];
    if (!Evaluate(r.a, c.a, frc, dfdr, dfdc)) {
      return false;
    }

    for (int i = 0; i < kDimension; ++i) {
      f[i].a = frc[i];
      f[i].v = dfdr[i] * r.v + dfdc[i] * c.v;
    }

    return true;
  }

  int NumRows() const { return data_.NumRows(); }
  int NumCols() const { return data_.NumCols(); }

 private:
  const DataProvider& data_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_CUBIC_INTERPOLATOR_H_
