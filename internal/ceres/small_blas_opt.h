// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: yangfan34@lenovo.com (Lenovo Research Device+ Lab - Shanghai)
//
// Optimization for simple blas functions used in the Schur Eliminator.
// These are fairly basic implementations which already yield a significant
// speedup in the eliminator performance.

#ifndef CERES_INTERNAL_SMALL_BLAS_OPT_H_
#define CERES_INTERNAL_SMALL_BLAS_OPT_H_

namespace ceres {
namespace internal {

// The following macros are used to share code
#define CERES_GEMM_OPT_NAIVE_HEADER              \
  double c0 = 0.0;                               \
  double c1 = 0.0;                               \
  double c2 = 0.0;                               \
  double c3 = 0.0;                               \
  double* pa = (double*)AA;                      \
  double* pb = (double*)BB;                      \
  int col_r = COL_A & 0x00000003;                \
  int col_m = COL_A - col_r;

#define CERES_GEMM_OPT_MAT1X4_MUL                \
  pb = (double*)BB + bi;                         \
  c0 += av * *pb++;                              \
  c1 += av * *pb++;                              \
  c2 += av * *pb++;                              \
  c3 += av * *pb++;                              \
  bi += ldb;

#define CERES_GEMM_OPT_MVM_MAT4X1_MUL            \
  c0 += *(pa                 ) * *pb;            \
  c1 += *(pa + lda           ) * *pb;            \
  c2 += *(pa +       (lda<<1)) * *pb;            \
  c3 += *(pa + lda + (lda<<1)) * *pb++;          \
  pa++;

#define CERES_GEMM_OPT_MTV_MAT4X1_MUL            \
  c0 += *(pa    ) * *pb;                         \
  c1 += *(pa + 1) * *pb;                         \
  c2 += *(pa + 2) * *pb;                         \
  c3 += *(pa + 3) * *pb++;                       \
  pa += lda;

#define CERES_GEMM_OPT_STORE_MAT1X4              \
  if (kOperation > 0) {                          \
    *CC++ += c0;                                 \
    *CC++ += c1;                                 \
    *CC++ += c2;                                 \
    *CC++ += c3;                                 \
  } else if (kOperation < 0) {                   \
    *CC++ -= c0;                                 \
    *CC++ -= c1;                                 \
    *CC++ -= c2;                                 \
    *CC++ -= c3;                                 \
  } else {                                       \
    *CC++ = c0;                                  \
    *CC++ = c1;                                  \
    *CC++ = c2;                                  \
    *CC++ = c3;                                  \
  }

// Matrix-Matrix Multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a * B;
// where op can be +=, -=, or =, indicated by kOperation.
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3   op   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
static void MMM_mat1x4(int COL_A, const double* AA, const double* BB,
		int ldb, double* CC, int kOperation) {
  CERES_GEMM_OPT_NAIVE_HEADER
  double av = 0.0;
  int bi = 0;

  for(int k = 0; k < col_m; k += 4) {
    av = pa[k + 0];
    CERES_GEMM_OPT_MAT1X4_MUL

    av = pa[k + 1];
    CERES_GEMM_OPT_MAT1X4_MUL

    av = pa[k + 2];
    CERES_GEMM_OPT_MAT1X4_MUL

    av = pa[k + 3];
    CERES_GEMM_OPT_MAT1X4_MUL
  }

  for(int k = col_m; k < COL_A; k++) {
    av = pa[k];
    CERES_GEMM_OPT_MAT1X4_MUL
  }

  CERES_GEMM_OPT_STORE_MAT1X4
}

// Matrix Transpose-Matrix multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a' * B;
// where op can be +=, -=, or = indicated by kOperation.
//
//                        Matrix A
//
//                        A0
//                        A1
//                        A2
//                        A3
//                        .
//                        .
//                        .
//
//  Matrix C              Matrix A'                  Matrix B
//
//  C0, C1, C2, C3   op   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// NOTE: COL_A means the columns of A'
static void MTM_mat1x4(int COL_A, const double* AA, int lda,
		const double* BB, int ldb, double* CC, int kOperation) {
  CERES_GEMM_OPT_NAIVE_HEADER
  double av = 0.0;
  int ai = 0;
  int bi = 0;

  for(int k = 0; k < col_m; k += 4) {
    av = pa[ai];
    CERES_GEMM_OPT_MAT1X4_MUL
    ai += lda;

    av = pa[ai];
    CERES_GEMM_OPT_MAT1X4_MUL
    ai += lda;

    av = pa[ai];
    CERES_GEMM_OPT_MAT1X4_MUL
    ai += lda;

    av = pa[ai];
    CERES_GEMM_OPT_MAT1X4_MUL
    ai += lda;
  }

  for(int k = col_m; k < COL_A; k++) {
    av = pa[ai];
    CERES_GEMM_OPT_MAT1X4_MUL
    ai += lda;
  }

  CERES_GEMM_OPT_STORE_MAT1X4
}

// Matrix-Vector Multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A * b;
// where op can be +=, -=, or =, indicated by kOperation.
//
//  Vector c              Matrix A                   Vector b
//
//  C0               op   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
static void MVM_mat4x1(int COL_A, const double* AA, int lda,
		const double* BB, double* CC, int kOperation) {
  CERES_GEMM_OPT_NAIVE_HEADER

  for(int k = 0; k < col_m; k += 4) {
    CERES_GEMM_OPT_MVM_MAT4X1_MUL
    CERES_GEMM_OPT_MVM_MAT4X1_MUL
    CERES_GEMM_OPT_MVM_MAT4X1_MUL
    CERES_GEMM_OPT_MVM_MAT4X1_MUL
  }

  for(int k = col_m; k < COL_A; k++) {
    CERES_GEMM_OPT_MVM_MAT4X1_MUL
  }

  CERES_GEMM_OPT_STORE_MAT1X4
}

// Matrix Transpose-Vector multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A' * b;
// where op can be +=, -=, or =, indicated by kOperation.
//
//                        Matrix A
//
//                        A0, A4, A8, Ac
//                        A1, A5, A9, Ad
//                        A2, A6, Aa, Ae
//                        A3, A7, Ab, Af
//                        . , . , . , .
//                        . , . , . , .
//                        . , . , . , .
//
//  Vector c              Matrix A'                  Vector b
//
//  C0               op   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
// NOTE: COL_A means the columns of A'
static void MTV_mat4x1(int COL_A, const double* AA, int lda,
		const double* BB, double* CC, int kOperation) {
  CERES_GEMM_OPT_NAIVE_HEADER

  for(int k = 0; k < col_m; k += 4) {
    CERES_GEMM_OPT_MTV_MAT4X1_MUL
    CERES_GEMM_OPT_MTV_MAT4X1_MUL
    CERES_GEMM_OPT_MTV_MAT4X1_MUL
    CERES_GEMM_OPT_MTV_MAT4X1_MUL
  }

  for(int k = col_m; k < COL_A; k++) {
    CERES_GEMM_OPT_MTV_MAT4X1_MUL
  }

  CERES_GEMM_OPT_STORE_MAT1X4
}

#undef STORE_MATRIX_C_1X4

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SMALL_BLAS_OPT_H_
