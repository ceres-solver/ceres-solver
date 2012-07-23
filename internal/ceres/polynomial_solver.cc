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

double ComputeCompanionMatrixRowNorm(const Matrix& C, int row) {
  const int N = C.rows();
  if (row == 0) {
    return abs(C(0,N-1));
  } else if (row == N-1) {
    return C(N-1,N-2);
  }
  return C(row,row-1) + abs(C(row,N-1));
}

double ComputeCompanionMatrixColumnNorm(const Matrix& C, int col) {
  const int N = C.rows();
  if (col == N-1) {
    double sum = 0.0;
    for (int i=0; i<N-1; ++i) {
      sum += C(i,N-1);
    }
    return sum;
  }
  return C(col+1,col);
}

// Scale a row/column pair of the companion matrix C and adjust the row and column norms
// by recomputing the norms that have changed.
void ScaleCompanionMatrixRowAndColumn(Matrix* C, Vector* row_norms, Vector* col_norms, int i, int exponent) {
  CHECK_NOTNULL(C);
  CHECK_NOTNULL(row_norms);
  CHECK_NOTNULL(col_norms);

  Matrix& CC = *C;
  const int N = CC.rows();
  if (i == 0) {
    CC(0, N-1) = std::ldexp(CC(0, N-1), -exponent);
    CC(1, 0) = std::ldexp(CC(1, 0), exponent);

    (*row_norms)(0) = ComputeCompanionMatrixRowNorm(CC, 0);
    (*row_norms)(1) = ComputeCompanionMatrixRowNorm(CC, 1);
    (*col_norms)(N-1) = ComputeCompanionMatrixColumnNorm(CC, N-1);
    (*col_norms)(0) = ComputeCompanionMatrixColumnNorm(CC, 0);
  } else if (i == N-1) {
    CC(N-1, N-2) = std::ldexp(CC(N-1, N-2), -exponent);
    for (int j=0; j<N-1; ++j) {
      CC(j, N-1) = std::ldexp(CC(j, N-1), exponent);
    }

    for (int j=0; j<N; ++j) {
      (*row_norms)(j) = ComputeCompanionMatrixRowNorm(CC, j);
    }
    (*col_norms)(N-1) = ComputeCompanionMatrixColumnNorm(CC, N-1);
    (*col_norms)(N-2) = ComputeCompanionMatrixColumnNorm(CC, N-2);
  } else {
    CC(i, i-1) = std::ldexp(CC(i, i-1), -exponent);
    CC(i, N-1) = std::ldexp(CC(i, N-1), -exponent);
    CC(i+1, i) = std::ldexp(CC(i+1, i), exponent);

    (*row_norms)(i) = ComputeCompanionMatrixRowNorm(CC, i);
    (*row_norms)(i+1) = ComputeCompanionMatrixRowNorm(CC, i+1);
    (*col_norms)(i-1) = ComputeCompanionMatrixColumnNorm(CC, i-1);
    (*col_norms)(i) = ComputeCompanionMatrixColumnNorm(CC, i);
    (*col_norms)(N-1) = ComputeCompanionMatrixColumnNorm(CC, N-1);
  }
}

void BalanceCompanionMatrix(Matrix* C) {
  CHECK_NOTNULL(C);
  Matrix& CC = *C;
  const int degree = CC.rows();
  const double gamma = 0.9;

  // Greedily scale row/column pairs until there is no change.
  // This only looks at the off-diagonal part of C, as the diagonal
  // is not affected by the scaling D^-1 C D.
  Vector col_norms(degree);
  Vector row_norms(degree);
  for (int i=0; i<degree; ++i) {
    row_norms(i) = ComputeCompanionMatrixRowNorm(CC, i);
    col_norms(i) = ComputeCompanionMatrixColumnNorm(CC, i);
  }

  bool scaling_has_changed;
  do {
    scaling_has_changed = false;
    // Compute the fraction of the row 1-norm abs_row and the
    // column 1-norm abs_col.

    for (int i=0; i<degree; ++i) {
      int exponent = 0;
      const double col_norm = col_norms(i);
      const double row_norm = row_norms(i);
      std::frexp(row_norm/col_norm, &exponent);
      exponent /= 2;
      if(exponent != 0) {
        const double scaled_col_norm = std::ldexp(col_norm, exponent);
        const double scaled_row_norm = std::ldexp(row_norm, -exponent);
        if (scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm)) {
          // Accept the new scaling.
          scaling_has_changed = true;
          ScaleCompanionMatrixRowAndColumn(C, &row_norms, &col_norms, i, exponent);
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

  // TODO(markus) remove roots x = 0

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
