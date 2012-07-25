// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include "ceres/polynomial_solver.h"

#include <glog/logging.h>
#include "Eigen/Dense"
#include "ceres/internal/port.h"
#include <cmath>

namespace ceres {
namespace internal {
namespace {
// Balancing function as described by Parlett and Reinsch.
// In contrast to their description, this function includes
// the diagonal in the computations, which slightly alters
// the meaning of the parameter gamma.
// For companion matrices, this is not a big issue, as the
// diagonal is mostly zero (only the last diagonal entry
// is non-zero).
void BalanceCompanionMatrix(Matrix* C) {
  CHECK_NOTNULL(C);
  Matrix& CC = *C;

  const int degree = CC.rows();
  const double gamma = 0.9;

  // Greedily scale row/column pairs until there is no change.
  bool scaling_has_changed;
  do {
    scaling_has_changed = false;
    // Compute the fraction of the row 1-norm abs_row and the
    // column 1-norm abs_col.

    for (int i=0; i<degree; ++i) {
      const double col_norm = CC.col(i).lpNorm<1>();
      const double row_norm = CC.row(i).lpNorm<1>();

      // Decompose row_norm/col_norm into mantissa * 2^exponent,
      // where 0.5 <= mantissa < 1. Discard mantissa (return value
      // of frexp), as only the exponent is needed.
      int exponent = 0;
      std::frexp(row_norm/col_norm, &exponent);
      exponent /= 2;

      if(exponent != 0) {
        const double scaled_col_norm = std::ldexp(col_norm, exponent);
        const double scaled_row_norm = std::ldexp(row_norm, -exponent);
        if (scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm)) {
          // Accept the new scaling. (Multiplication by powers of 2 should not
          // introduce rounding errors (ignoring non-normalized numbers and
          // over- or underflow))
          scaling_has_changed = true;
          CC.row(i) *= std::ldexp(1.0, -exponent);
          CC.col(i) *= std::ldexp(1.0, exponent);
        }
      }
    }
  } while (scaling_has_changed);

  VLOG(3) << "Balanced companion matrix is\n" << CC;
}

void BuildCompanionMatrix(const Vector& polynomial, Matrix* C) {
  CHECK_NOTNULL(C);
  Matrix& CC = *C;

  const int degree = polynomial.size() - 1;

  CC.resize(degree, degree);
  CC.setZero();
  CC(0,degree-1) = -polynomial(degree);
  for (int i=1; i<degree; ++i) {
    CC(i,i-1) = 1.0;
    CC(i,degree-1) = -polynomial(degree-i);
  }
}
}

int FindPolynomialRoots(const Vector& polynomial_in, Vector* real, Vector* imag) {
  const double epsilon = std::numeric_limits<double>::epsilon();

  Vector polynomial = polynomial_in;
  int degree = polynomial.size() - 1;

  if (degree < 0) {
    // This is not a valid polynomial.
    return -1;
  }

  if (real) {
   real->resize(degree);
  }

  if (imag) {
    imag->resize(degree);
  }

  // Divide by leading term
  for (int i=0; i<degree; ++i) {
    polynomial(i+1) /= polynomial(0);
  }
  polynomial(0) = 1.0;

  if (degree > 0) {
    if (degree == 1) {
      if (real) {
        (*real)(0) = -polynomial[1];
      }
      if (imag) {
        (*imag)(0) = 0.0;
      }
    } else {
      // Build and balance the companion matrix to the polynomial.
      Matrix C(degree, degree);
      BuildCompanionMatrix(polynomial, &C);
      BalanceCompanionMatrix(&C);

      // Find its (complex) eigenvalues.
      Eigen::EigenSolver< Matrix > solver(C, Eigen::EigenvaluesOnly);
      if (solver.info() != Eigen::Success) {
        LOG(WARNING) << "Failed to extract eigenvalues from companion matrix.";
        return -1;
      }

      // Output roots
      for (int i=0; i<degree; ++i) {
        if (real != NULL) {
          (*real)(i) = solver.eigenvalues()(i).real();
        }
        if (imag != NULL) {
          (*imag)(i) = solver.eigenvalues()(i).imag();
        }
      }
    }
  }
  return 0;
}

}  // namespace internal
}  // namespace ceres
