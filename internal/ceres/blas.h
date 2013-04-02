// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Simple blas functions for use in the Schur Eliminator. These are
// fairly basic implementations which already yield a significant
// speedup in the eliminator performance.

#include "ceres/internal/eigen.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

// C op A * B;
//
// where op can be +=, -=, or =.
//
// The template parameters (kRowA, kColA, kRowB, kColB) allow
// specialization of the loop at compile time. If this information is
// not available, then Eigen::Dynamic should be used as the template
// argument.
//
// kOperation =  1  -> C += A * B
// kOperation = -1  -> C -= A * B
// kOperation =  0  -> C  = A * B
//
// The function can write into matrice C which are larger than the
// matrix A * B. This is done by specifying the true size of C via
// row_stride_c and col_stride_c, and then indicating where A * B
// should be written into by start_row_c and start_col_c.
//
// Graphically if row_stride_c = 10, col_stride_c = 12, start_row_c =
// 4 and start_col_c = 5, then if A = 3x2 and B = 2x4, we get
//
//   ------------
//   ------------
//   ------------
//   ------------
//   -----xxxx---
//   -----xxxx---
//   -----xxxx---
//   ------------
//   ------------
//   ------------
template<int kRowA, int kColA, int kRowB, int kColB, int kOperation>
inline void MatrixMatrixMultiply(const double* A,
                                 const int num_row_a,
                                 const int num_col_a,
                                 const double* B,
                                 const int num_row_b,
                                 const int num_col_b,
                                 double* C,
                                 const int start_row_c,
                                 const int start_col_c,
                                 const int row_stride_c,
                                 const int col_stride_c) {
  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK_GT(num_row_b, 0);
  DCHECK_GT(num_col_b, 0);
  DCHECK_GE(start_row_c, 0);
  DCHECK_GE(start_col_c, 0);
  DCHECK_GT(row_stride_c, 0);
  DCHECK_GT(col_stride_c, 0);

  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));
  DCHECK((kRowB == Eigen::Dynamic) || (kRowB == num_row_b));
  DCHECK((kColB == Eigen::Dynamic) || (kColB == num_col_b));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);
  const int NUM_ROW_B = (kColB != Eigen::Dynamic ? kRowB : num_row_b);
  const int NUM_COL_B = (kColB != Eigen::Dynamic ? kColB : num_col_b);
  DCHECK_EQ(NUM_COL_A, NUM_ROW_B);

  const int NUM_ROW_C = NUM_ROW_A;
  const int NUM_COL_C = NUM_COL_B;
  DCHECK_LT(start_row_c + NUM_ROW_C, row_stride_c);
  DCHECK_LT(start_col_c + NUM_COL_C, col_stride_c);

  for (int r = 0; r < NUM_ROW_C; ++r) {
    for (int c = 0; c < NUM_COL_C; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < NUM_COL_A; ++k) {
        tmp += A[r * NUM_COL_A + k] * B[k * NUM_COL_B + c];
      }

      const int index = (r + start_row_c) * col_stride_c + start_col_c + c;
      if (kOperation > 0) {
        C[index] += tmp;
      } else if (kOperation < 0) {
        C[index] -= tmp;
      } else {
        C[index] = tmp;
      }
    }
  }
}

// C op A' * B;
//
// where op can be +=, -=, or =.
//
// The template parameters (kRowA, kColA, kRowB, kColB) allow
// specialization of the loop at compile time. If this information is
// not available, then Eigen::Dynamic should be used as the template
// argument.
//
// kOperation =  1  -> C += A' * B
// kOperation = -1  -> C -= A' * B
// kOperation =  0  -> C  = A' * B
//
// The function can write into matrice C which are larger than the
// matrix A' * B. This is done by specifying the true size of C via
// row_stride_c and col_stride_c, and then indicating where A * B
// should be written into by start_row_c and start_col_c.
//
// Graphically if row_stride_c = 10, col_stride_c = 12, start_row_c =
// 4 and start_col_c = 5, then if A = 2x3 and B = 2x4, we get
//
//   ------------
//   ------------
//   ------------
//   ------------
//   -----xxxx---
//   -----xxxx---
//   -----xxxx---
//   ------------
//   ------------
//   ------------
template<int kRowA, int kColA, int kRowB, int kColB, int kOperation>
inline void MatrixTransposeMatrixMultiply(const double* A,
                                          const int num_row_a,
                                          const int num_col_a,
                                          const double* B,
                                          const int num_row_b,
                                          const int num_col_b,
                                          double* C,
                                          const int start_row_c,
                                          const int start_col_c,
                                          const int row_stride_c,
                                          const int col_stride_c) {
  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK_GT(num_row_b, 0);
  DCHECK_GT(num_col_b, 0);
  DCHECK_GE(start_row_c, 0);
  DCHECK_GE(start_col_c, 0);
  DCHECK_GT(row_stride_c, 0);
  DCHECK_GT(col_stride_c, 0);

  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));
  DCHECK((kRowB == Eigen::Dynamic) || (kRowB == num_row_b));
  DCHECK((kColB == Eigen::Dynamic) || (kColB == num_col_b));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);
  const int NUM_ROW_B = (kColB != Eigen::Dynamic ? kRowB : num_row_b);
  const int NUM_COL_B = (kColB != Eigen::Dynamic ? kColB : num_col_b);
  DCHECK_EQ(NUM_ROW_A, NUM_ROW_B);

  const int NUM_ROW_C = NUM_COL_A;
  const int NUM_COL_C = NUM_COL_B;
  DCHECK_LT(start_row_c + NUM_ROW_C, row_stride_c);
  DCHECK_LT(start_col_c + NUM_COL_C, col_stride_c);

  for (int r = 0; r < NUM_ROW_C; ++r) {
    for (int c = 0; c < NUM_COL_C; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < NUM_ROW_A; ++k) {
        tmp += A[k * NUM_COL_A + r] * B[k * NUM_COL_B + c];
      }

      const int index = (r + start_row_c) * col_stride_c + start_col_c + c;
      if (kOperation > 0) {
        C[index]+= tmp;
      } else if (kOperation < 0) {
        C[index]-= tmp;
      } else {
        C[index]= tmp;
      }
    }
  }
}

// c op A' * b;
//
// where op can be +=, -=, or =.
//
// The template parameters (kRowA, kColA) allow specialization of the
// loop at compile time. If this information is not available, then
// Eigen::Dynamic should be used as the template argument.
//
// kOperation =  1  -> c += A' * b
// kOperation = -1  -> c -= A' * b
// kOperation =  0  -> c  = A' * b
template<int kRowA, int kColA, int kOperation>
inline void MatrixTransposeVectorMultiply(const double* A,
                                          const int num_row_a,
                                          const int num_col_a,
                                          const double* b,
                                          double* c) {
  DCHECK_GT(num_row_a, 0);
  DCHECK_GT(num_col_a, 0);
  DCHECK((kRowA == Eigen::Dynamic) || (kRowA == num_row_a));
  DCHECK((kColA == Eigen::Dynamic) || (kColA == num_col_a));

  const int NUM_ROW_A = (kRowA != Eigen::Dynamic ? kRowA : num_row_a);
  const int NUM_COL_A = (kColA != Eigen::Dynamic ? kColA : num_col_a);

  for (int r = 0; r < NUM_COL_A; ++r) {
    double tmp = 0.0;
    for (int c = 0; c < NUM_ROW_A; ++c) {
      tmp += A[c * NUM_COL_A + r] * b[c];
    }

    if (kOperation > 0) {
      c[r] += tmp;
    } else if (kOperation < 0) {
      c[r] -= tmp;
    } else {
      c[r] = tmp;
    }
  }
}

}  // namespace internal
}  // namespace ceres
