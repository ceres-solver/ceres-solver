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

#include <cmath>
#include <cstddef>

#include "Eigen/Dense"
#include "ceres/internal/port.h"

namespace ceres {
namespace internal {
namespace {
// Balancing function as described by Parlett and Reinsch,
// "Balancing a Matrix for Calculation of Eigenvalues and
// Eigenvectors".
// In contrast to their description, this function includes
// the diagonal in the computations, which slightly alters
// the meaning of the parameter gamma.
// For companion matrices, this is not a big issue, as the
// diagonal is mostly zero (only the last diagonal entry
// is non-zero).
void BalanceCompanionMatrix(Matrix* companion_matrix_ptr) {
  CHECK_NOTNULL(companion_matrix_ptr);
  Matrix& companion_matrix = *companion_matrix_ptr;

  const int degree = companion_matrix.rows();
  const double gamma = 0.9;

  // Greedily scale row/column pairs until there is no change.
  bool scaling_has_changed;
  do {
    scaling_has_changed = false;

    for (int i = 0; i < degree; ++i) {
      // This computes the norm over the whole row/column, which
      // is different from Parlett and Reinsch.
      const double row_norm = companion_matrix.row(i).lpNorm<1>();
      const double col_norm = companion_matrix.col(i).lpNorm<1>();

      // Decompose row_norm/col_norm into mantissa * 2^exponent,
      // where 0.5 <= mantissa < 1. Discard mantissa (return value
      // of frexp), as only the exponent is needed.
      int exponent = 0;
      std::frexp(row_norm/col_norm, &exponent);
      exponent /= 2;

      if (exponent != 0) {
        const double scaled_col_norm = std::ldexp(col_norm, exponent);
        const double scaled_row_norm = std::ldexp(row_norm, -exponent);
        if (scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm)) {
          // Accept the new scaling. (Multiplication by powers of 2 should not
          // introduce rounding errors (ignoring non-normalized numbers and
          // over- or underflow))
          scaling_has_changed = true;
          companion_matrix.row(i) *= std::ldexp(1.0, -exponent);
          companion_matrix.col(i) *= std::ldexp(1.0, exponent);
        }
      }
    }
  } while (scaling_has_changed);

  VLOG(3) << "Balanced companion matrix is\n" << companion_matrix;
}

void BuildCompanionMatrix(const Vector& polynomial,
    Matrix* companion_matrix_ptr) {
  CHECK_NOTNULL(companion_matrix_ptr);
  Matrix& companion_matrix = *companion_matrix_ptr;

  const int degree = polynomial.size() - 1;

  companion_matrix.resize(degree, degree);
  companion_matrix.setZero();
  companion_matrix.diagonal(-1) = 1.0;
  companion_matrix.col(degree-1) = -polynomial.reverse().head(degree-1);
}
}

bool FindPolynomialRoots(const Vector& polynomial_in,
    Vector* real, Vector* imaginary) {
  const double epsilon = std::numeric_limits<double>::epsilon();

  Vector polynomial = polynomial_in;
  int degree = polynomial.size() - 1;

  if (degree < 0) {
    // This is not a valid polynomial.
    LOG(ERROR) << "Invalid polynomial of size 0 passed to FindPolynomialRoots";
    return false;
  }

  // Divide by leading term
  if (polynomial(0) == 0.0) {
    LOG(ERROR) << "Leading polynomial coefficient is zero in "
      "FindPolynomialRoots";
    return false;
  }

  for (int i = 0; i < degree; ++i) {
    polynomial(i+1) /= polynomial(0);
  }
  polynomial(0) = 1.0;

  // Is the polynomial constant?
  if (degree == 0) {
    LOG(WARN) << "Trying to extract roots from a constant "
      "polynomial in FindPolynomialRoots";
    return true;
  }

  // Separately handle linear polynomials.
  if (degree == 1) {
    if (real != NULL) {
      real->resize(1);
      (*real)(0) = -polynomial[1];
    }
    if (imag) {
      imaginary->resize(1);
      imaginary->setZero();
    }
  }

  // The degree is now known to be at least 2.
  // Build and balance the companion matrix to the polynomial.
  Matrix companion_matrix(degree, degree);
  BuildCompanionMatrix(polynomial, &companion_matrix);
  BalanceCompanionMatrix(&companion_matrix);

  // Find its (complex) eigenvalues.
  Eigen::EigenSolver< Matrix > solver(companion_matrix, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    LOG(ERROR) << "Failed to extract eigenvalues from companion matrix.";
    return false;
  }

  // Output roots
  if (real != NULL) {
    *real = solver.eigenvalues().real();
  } else {
    LOG(WARN) << "NULL pointer passed as real argument to FindPolynomialRoots. "
     "Real parts of the roots will not be returned.";
  }
  if (imaginary != NULL) {
    *imaginary = solver.eigenvalues().imag();
  }
  return true;
}

}  // namespace internal
}  // namespace ceres
