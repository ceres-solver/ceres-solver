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

#ifndef CERES_PUBLIC_CUBIC_INTERPOLATION_H_
#define CERES_PUBLIC_CUBIC_INTERPOLATION_H_

namespace ceres {

// This class takes as input a one dimensional array of values that is
// assumed to be integer valued samples from a function f(x),
// evaluated at x = 0, ... , n - 1 and uses cubic Hermite splines to
// produce a smooth approximation to it that can be used to evaluate
// the f(x) and f'(x) at any fractional point in the interval [0,
// n-1].
//
// Besides this, the reason this class is included with Ceres is that
// the Evaluate method is overloaded so that the user can use it as
// part of their automatically differentiated CostFunction objects
// without worrying about the fact that they are working with a
// numerically interpolated object.
//
// For more details on cubic interpolation see
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
//
// Example usage:
//
//  const double x[] = {1.0, 2.0, 5.0, 6.0};
//  CubicInterpolator interpolator(x, 4);
//  double f, dfdx;
//  CHECK(interpolator.Evaluator(1.5, &f, &dfdx));
class CERES_EXPORT CubicInterpolator {
 public:
  // values is an array containing the values of the function to be
  // interpolated on the integer lattice [0, num_values - 1].
  //
  // values should be a valid pointer for the lifetime of this object.
  CubicInterpolator(const double* values, int num_values);

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if x is out of bounds.
  bool Evaluate(double x, double* f, double* dfdx) const;

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  bool Evaluate(const double& x, double* f) const {
    return Evaluate(x, f, NULL);
  }

  template<typename JetT> bool Evaluate(const JetT& x, JetT* f) const {
    double dfdx;
    if (!Evaluate(x.a, &f->a, &dfdx)) {
      return false;
    }
    f->v = dfdx * x.v;
    return true;
  }

  int num_values() const { return num_values_; }

 private:
  const double* values_;
  const int num_values_;
};

// This class takes as input a row-major array of values that is
// assumed to be integer valued samples from a function f(x),
// evaluated on the integer lattice [0, num_rows - 1] x [0, num_cols -
// 1]; and uses the cubic convolution interpolation algorithm of
// R. Keys, to produce a smooth approximation to it that can be used
// to evaluate the f(r,c), df(r, c)/dr and df(r,c)/dc at any
// fractional point inside this lattice.
//
// For more details on cubic interpolation see
//
// "Cubic convolution interpolation for digital image processing".
// IEEE Transactions on Acoustics, Speech, and Signal Processing
// 29 (6): 1153–1160.
//
// http://en.wikipedia.org/wiki/Cubic_Hermite_spline
// http://en.wikipedia.org/wiki/Bicubic_interpolation
class CERES_EXPORT BiCubicInterpolator {
 public:
  // values is a row-major array containing the values of the function
  // to be interpolated on the integer lattice [0, num_rows - 1] x [0,
  // num_cols - 1];
  //
  // values should be a valid pointer for the lifetime of this object.
  BiCubicInterpolator(const double* values, int num_rows, int num_cols);

  // Evaluate the interpolated function value and/or its
  // derivative. Returns false if r or c is out of bounds.
  bool Evaluate(double r, double c,
                double* f, double* dfdr, double* dfdc) const;

  // The following two Evaluate overloads are needed for interfacing
  // with automatic differentiation. The first is for when a scalar
  // evaluation is done, and the second one is for when Jets are used.
  bool Evaluate(const double& r, const double& c, double* f) const {
    return Evaluate(r, c, f, NULL, NULL);
  }

  template<typename JetT> bool Evaluate(const JetT& r,
                                        const JetT& c,
                                        JetT* f) const {
    double dfdr, dfdc;
    if (!Evaluate(r.a, c.a, &f->a, &dfdr, &dfdc)) {
      return false;
    }
    f->v = dfdr * r.v + dfdc * c.v;
    return true;
  }

  int num_rows() const { return num_rows_; }
  int num_cols() const { return num_cols_; }

 private:
  const double* values_;
  const int num_rows_;
  const int num_cols_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_CUBIC_INTERPOLATOR_H_
