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
// Author: yangfan34@lenovo.com (Lenovo Research Shanghai)
//
// Optimization for simple blas functions used in the Schur Eliminator.
// These are fairly basic implementations which already yield a significant
// speedup in the eliminator performance.

#ifndef CERES_INTERNAL_SMALL_BLAS_OPT_H_
#define CERES_INTERNAL_SMALL_BLAS_OPT_H_

#ifdef __cplusplus
extern "C" {
#endif

void MMM_mat1x4_p1_asm(int COL_A,
		const double* AA, const double* BB, int ldb, double* CC);
void MMM_mat1x4_m1_asm(int COL_A,
		const double* AA, const double* BB, int ldb, double* CC);
void MMM_mat1x4_00_asm(int COL_A,
		const double* AA, const double* BB, int ldb, double* CC);

void MMM_mat4x4_p1_asm(int COL_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);
void MMM_mat4x4_m1_asm(int COL_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);
void MMM_mat4x4_00_asm(int COL_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);

void MTM_mat1x4_p1_asm(int ROW_A,
		const double* AA, int lda, const double* BB, int ldb, double* CC);
void MTM_mat1x4_m1_asm(int ROW_A,
		const double* AA, int lda, const double* BB, int ldb, double* CC);
void MTM_mat1x4_00_asm(int ROW_A,
		const double* AA, int lda, const double* BB, int ldb, double* CC);

void MTM_mat4x4_p1_asm(int ROW_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);
void MTM_mat4x4_m1_asm(int ROW_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);
void MTM_mat4x4_00_asm(int ROW_A,
        const double* AA, int lda, const double* BB, int ldb, double* CC, int ldc);

#ifdef __cplusplus
}
#endif

namespace ceres {
namespace internal {

// The following macros are used to share code

#define MMM_MAT1X4_MAIN                                       \
    double* pa = (double*)AA;                                 \
    double* pb = NULL;                                        \
    double av = 0.0;                                          \
    int bi = 0;                                               \
    int col_r = COL_A & 0x00000003;                           \
    int col_m = COL_A - col_r;                                \
                                                              \
    for(int k = 0; k < col_m; k += 4) {                       \
        av = AA[k + 0];                                       \
        pb = (double*)BB + bi;                                \
        c0 += av * *pb++;                                     \
        c1 += av * *pb++;                                     \
        c2 += av * *pb++;                                     \
        c3 += av * *pb++;                                     \
        bi += ldb;                                            \
                                                              \
        av = AA[k + 1];                                       \
        pb = (double*)BB + bi;                                \
        c0 += av * *pb++;                                     \
        c1 += av * *pb++;                                     \
        c2 += av * *pb++;                                     \
        c3 += av * *pb++;                                     \
        bi += ldb;                                            \
                                                              \
        av = AA[k + 2];                                       \
        pb = (double*)BB + bi;                                \
        c0 += av * *pb++;                                     \
        c1 += av * *pb++;                                     \
        c2 += av * *pb++;                                     \
        c3 += av * *pb++;                                     \
        bi += ldb;                                            \
                                                              \
        av = AA[k + 3];                                       \
        pb = (double*)BB + bi;                                \
        c0 += av * *pb++;                                     \
        c1 += av * *pb++;                                     \
        c2 += av * *pb++;                                     \
        c3 += av * *pb++;                                     \
        bi += ldb;                                            \
    }                                                         \
                                                              \
    for(int k = col_m; k < COL_A; k++) {                      \
        av = AA[k];                                           \
        pb = (double*)BB + bi;                                \
        c0 += av * *pb++;                                     \
        c1 += av * *pb++;                                     \
        c2 += av * *pb++;                                     \
        c3 += av * *pb++;                                     \
        bi += ldb;                                            \
    }

#define MTM_MAT1X4_MAIN                                       \
    double* pa = (double*)AA;                                 \
    double* pb = NULL;                                        \
    double av = 0.0;                                          \
    int ai = 0;                                               \
    int bi = 0;                                               \
    int col_r = ROW_A & 0x00000003;                           \
    int col_m = ROW_A - col_r;                                \
                                                              \
    for(int k = 0; k < col_m; k += 4) {                       \
       av = AA[ai];                                           \
       pb = (double*)BB + bi;                                 \
       c0 += av * *pb++;                                      \
       c1 += av * *pb++;                                      \
       c2 += av * *pb++;                                      \
       c3 += av * *pb++;                                      \
       ai += lda;                                             \
       bi += ldb;                                             \
                                                              \
       av = AA[ai];                                           \
       pb = (double*)BB + bi;                                 \
       c0 += av * *pb++;                                      \
       c1 += av * *pb++;                                      \
       c2 += av * *pb++;                                      \
       c3 += av * *pb++;                                      \
       ai += lda;                                             \
       bi += ldb;                                             \
                                                              \
       av = AA[ai];                                           \
       pb = (double*)BB + bi;                                 \
       c0 += av * *pb++;                                      \
       c1 += av * *pb++;                                      \
       c2 += av * *pb++;                                      \
       c3 += av * *pb++;                                      \
       ai += lda;                                             \
       bi += ldb;                                             \
                                                              \
       av = AA[ai];                                           \
       pb = (double*)BB + bi;                                 \
       c0 += av * *pb++;                                      \
       c1 += av * *pb++;                                      \
       c2 += av * *pb++;                                      \
       c3 += av * *pb++;                                      \
       ai += lda;                                             \
       bi += ldb;                                             \
    }                                                         \
                                                              \
    for(int k = col_m; k < ROW_A; k++) {                      \
       av = AA[ai];                                           \
       pb = (double*)BB + bi;                                 \
       c0 += av * *pb++;                                      \
       c1 += av * *pb++;                                      \
       c2 += av * *pb++;                                      \
       c3 += av * *pb++;                                      \
       ai += lda;                                             \
       bi += ldb;                                             \
    }

#define MVM_MAT4X1_MAIN                                       \
    double* pa = (double*)AA;                                 \
    double* pb = (double*)BB;                                 \
    int col_r = COL_A & 0x00000003;                           \
    int col_m = COL_A - col_r;                                \
                                                              \
    for(int k = 0; k < col_m; k += 4) {                       \
        c0 += *(pa                 ) * *pb;                   \
        c1 += *(pa + lda           ) * *pb;                   \
        c2 += *(pa +       (lda<<1)) * *pb;                   \
        c3 += *(pa + lda + (lda<<1)) * *pb++;                 \
        pa++;                                                 \
                                                              \
        c0 += *(pa                 ) * *pb;                   \
        c1 += *(pa + lda           ) * *pb;                   \
        c2 += *(pa +       (lda<<1)) * *pb;                   \
        c3 += *(pa + lda + (lda<<1)) * *pb++;                 \
        pa++;                                                 \
                                                              \
        c0 += *(pa                 ) * *pb;                   \
        c1 += *(pa + lda           ) * *pb;                   \
        c2 += *(pa +       (lda<<1)) * *pb;                   \
        c3 += *(pa + lda + (lda<<1)) * *pb++;                 \
        pa++;                                                 \
                                                              \
        c0 += *(pa                 ) * *pb;                   \
        c1 += *(pa + lda           ) * *pb;                   \
        c2 += *(pa +       (lda<<1)) * *pb;                   \
        c3 += *(pa + lda + (lda<<1)) * *pb++;                 \
        pa++;                                                 \
    }                                                         \
                                                              \
    for(int k = col_m; k < COL_A; k++) {                      \
        c0 += *(pa                 ) * *pb;                   \
        c1 += *(pa + lda           ) * *pb;                   \
        c2 += *(pa +       (lda<<1)) * *pb;                   \
        c3 += *(pa + lda + (lda<<1)) * *pb++;                 \
        pa++;                                                 \
    }

#define MTV_MAT4X1_MAIN                                       \
    double* pa = (double*)AA;                                 \
    double* pb = (double*)BB;                                 \
    int col_r = ROW_A & 0x00000003;                           \
    int col_m = ROW_A - col_r;                                \
                                                              \
    for(int k = 0; k < col_m; k += 4) {                       \
        c0 += *(pa    ) * *pb;                                \
        c1 += *(pa + 1) * *pb;                                \
        c2 += *(pa + 2) * *pb;                                \
        c3 += *(pa + 3) * *pb++;                              \
        pa += lda;                                            \
                                                              \
        c0 += *(pa    ) * *pb;                                \
        c1 += *(pa + 1) * *pb;                                \
        c2 += *(pa + 2) * *pb;                                \
        c3 += *(pa + 3) * *pb++;                              \
        pa += lda;                                            \
                                                              \
        c0 += *(pa    ) * *pb;                                \
        c1 += *(pa + 1) * *pb;                                \
        c2 += *(pa + 2) * *pb;                                \
        c3 += *(pa + 3) * *pb++;                              \
        pa += lda;                                            \
                                                              \
        c0 += *(pa    ) * *pb;                                \
        c1 += *(pa + 1) * *pb;                                \
        c2 += *(pa + 2) * *pb;                                \
        c3 += *(pa + 3) * *pb++;                              \
        pa += lda;                                            \
    }                                                         \
                                                              \
    for(int k = col_m; k < ROW_A; k++) {                      \
        c0 += *(pa    ) * *pb;                                \
        c1 += *(pa + 1) * *pb;                                \
        c2 += *(pa + 2) * *pb;                                \
        c3 += *(pa + 3) * *pb++;                              \
        pa += lda;                                            \
    }                                                         \


// for(kk = 0; kk < col_m; kk += 4) {
//     The multiplication of A(4x4) * B(4x4) is divided into 4 parts,
//
//     step 1 (load two upper rows of A, load two left columns of B),
//     accumulate C0, C1, C4, C5
//
//     C0, C1           +=   A0, A1, A2, A3    *   B0, B1
//     C4, C5                A4, A5, A6, A7        B4, B5
//                                                 B8, B9
//                                                 Bc, Bd
//
//     step 2 (load two lower rows of A, reuse two left columns of B),
//     accumulate C8, C9, Cc, Cd
//
//                      +=                     *   B0, B1
//                                                 B4, B5
//     C8, C9                A8, A9, Aa, Ab        B8, B9
//     Cc, Cd                Ac, Ad, Ae, Af        Bc, Bd
//
//     step 3 (reuse two lower rows of A, load two right columns of B),
//     accumulate Ca, Cb, Ce, Cf
//
//                      +=                     *           B2, B3
//                                                         B6, B7
//              Ca, Cb       A8, A9, Aa, Ab                Ba, Bb
//              Ce, Cf       Ac, Ad, Ae, Af                Be, Bf
//
//     step 4 (reload two upper rows of A, reuse two right columns of B),
//     accumulate C2, C3, C6, C7
//
//              C2, C3  +=   A0, A1, A2, A3    *           B2, B3
//              C6, C7       A4, A5, A6, A7                B6, B7
//                                                         Ba, Bb
//                                                         Be, Bf
// }
//
// if(col_r & 0x00000002) {
//     ...
// }
//
// if(col_r & 0x00000001) {
//     ...
// }
#define MMM_MAT4X4_MAIN                                                  \
    double* pa = NULL;                                                   \
    double* pb = NULL;                                                   \
    int kk = 0;                                                          \
    int bi = 0;                                                          \
    int col_r = COL_A & 0x00000003;                                      \
    int col_m = COL_A - col_r;                                           \
    double A0, A1, A2, A3;                                               \
    double A4, A5, A6, A7;                                               \
    double B0, B1;                                                       \
    double B4, B5;                                                       \
    double B8, B9;                                                       \
    double Bc, Bd;                                                       \
                                                                         \
    for(kk = 0; kk < col_m; kk += 4) {                                   \
        pa = (double*)AA + kk;                                           \
        A0 = pa[0]; A1 = pa[1]; A2 = pa[2]; A3 = pa[3]; pa += lda;       \
        A4 = pa[0]; A5 = pa[1]; A6 = pa[2]; A7 = pa[3]; pa += lda;       \
                                                                         \
        pb = (double*)BB + bi;                                           \
        B0 = pb[0             ]; B1 = pb[1                 ];            \
        B4 = pb[ldb           ]; B5 = pb[ldb            + 1];            \
        B8 = pb[      (ldb<<1)]; B9 = pb[      (ldb<<1) + 1];            \
        Bc = pb[ldb + (ldb<<1)]; Bd = pb[ldb + (ldb<<1) + 1];            \
                                                                         \
        C0 += A0 * B0; C1 += A0 * B1; C4 += A4 * B0; C5 += A4 * B1;      \
        C0 += A1 * B4; C1 += A1 * B5; C4 += A5 * B4; C5 += A5 * B5;      \
        C0 += A2 * B8; C1 += A2 * B9; C4 += A6 * B8; C5 += A6 * B9;      \
        C0 += A3 * Bc; C1 += A3 * Bd; C4 += A7 * Bc; C5 += A7 * Bd;      \
                                                                         \
        A0 = pa[0]; A1 = pa[1]; A2 = pa[2]; A3 = pa[3]; pa += lda;       \
        A4 = pa[0]; A5 = pa[1]; A6 = pa[2]; A7 = pa[3]; pa += lda;       \
                                                                         \
        C8 += A0 * B0; C9 += A0 * B1; Cc += A4 * B0; Cd += A4 * B1;      \
        C8 += A1 * B4; C9 += A1 * B5; Cc += A5 * B4; Cd += A5 * B5;      \
        C8 += A2 * B8; C9 += A2 * B9; Cc += A6 * B8; Cd += A6 * B9;      \
        C8 += A3 * Bc; C9 += A3 * Bd; Cc += A7 * Bc; Cd += A7 * Bd;      \
                                                                         \
        pb = pb + 2;                                                     \
        B0 = pb[0             ]; B1 = pb[1                 ];            \
        B4 = pb[ldb           ]; B5 = pb[ldb            + 1];            \
        B8 = pb[      (ldb<<1)]; B9 = pb[      (ldb<<1) + 1];            \
        Bc = pb[ldb + (ldb<<1)]; Bd = pb[ldb + (ldb<<1) + 1];            \
                                                                         \
        Ca += A0 * B0; Cb += A0 * B1; Ce += A4 * B0; Cf += A4 * B1;      \
        Ca += A1 * B4; Cb += A1 * B5; Ce += A5 * B4; Cf += A5 * B5;      \
        Ca += A2 * B8; Cb += A2 * B9; Ce += A6 * B8; Cf += A6 * B9;      \
        Ca += A3 * Bc; Cb += A3 * Bd; Ce += A7 * Bc; Cf += A7 * Bd;      \
                                                                         \
        pa = (double*)AA + kk;                                           \
        A0 = pa[0]; A1 = pa[1]; A2 = pa[2]; A3 = pa[3]; pa += lda;       \
        A4 = pa[0]; A5 = pa[1]; A6 = pa[2]; A7 = pa[3];                  \
                                                                         \
        C2 += A0 * B0; C3 += A0 * B1; C6 += A4 * B0; C7 += A4 * B1;      \
        C2 += A1 * B4; C3 += A1 * B5; C6 += A5 * B4; C7 += A5 * B5;      \
        C2 += A2 * B8; C3 += A2 * B9; C6 += A6 * B8; C7 += A6 * B9;      \
        C2 += A3 * Bc; C3 += A3 * Bd; C6 += A7 * Bc; C7 += A7 * Bd;      \
        bi += ldb << 2;                                                  \
    }                                                                    \
                                                                         \
    if(col_r & 0x00000002) {                                             \
        pa = (double*)AA + kk;                                           \
        B0 = pa[0             ]; B1 = pa[1                 ];            \
        B4 = pa[lda           ]; B5 = pa[lda            + 1];            \
        B8 = pa[      (lda<<1)]; B9 = pa[      (lda<<1) + 1];            \
        Bc = pa[lda + (lda<<1)]; Bd = pa[lda + (lda<<1) + 1];            \
                                                                         \
        pb = (double*)BB + bi;                                           \
        A0 = pb[0]; A1 = pb[1]; A2 = pb[2]; A3 = pb[3]; bi += ldb;       \
        pb = (double*)BB + bi;                                           \
        A4 = pb[0]; A5 = pb[1]; A6 = pb[2]; A7 = pb[3]; bi += ldb;       \
                                                                         \
        C0 += B0 * A0; C1 += B0 * A1; C2 += B0 * A2; C3 += B0 * A3;      \
        C0 += B1 * A4; C1 += B1 * A5; C2 += B1 * A6; C3 += B1 * A7;      \
        C4 += B4 * A0; C5 += B4 * A1; C6 += B4 * A2; C7 += B4 * A3;      \
        C4 += B5 * A4; C5 += B5 * A5; C6 += B5 * A6; C7 += B5 * A7;      \
        C8 += B8 * A0; C9 += B8 * A1; Ca += B8 * A2; Cb += B8 * A3;      \
        C8 += B9 * A4; C9 += B9 * A5; Ca += B9 * A6; Cb += B9 * A7;      \
        Cc += Bc * A0; Cd += Bc * A1; Ce += Bc * A2; Cf += Bc * A3;      \
        Cc += Bd * A4; Cd += Bd * A5; Ce += Bd * A6; Cf += Bd * A7;      \
        kk += 2;                                                         \
    }                                                                    \
                                                                         \
    if(col_r & 0x00000001) {                                             \
        pa = (double*)AA + kk;                                           \
        B0 = pa[0             ];                                         \
        B4 = pa[lda           ];                                         \
        B8 = pa[      (lda<<1)];                                         \
        Bc = pa[lda + (lda<<1)];                                         \
                                                                         \
        pb = (double*)BB + bi;                                           \
        A0 = pb[0]; A1 = pb[1]; A2 = pb[2]; A3 = pb[3];                  \
                                                                         \
        C0 += B0 * A0; C1 += B0 * A1; C2 += B0 * A2; C3 += B0 * A3;      \
        C4 += B4 * A0; C5 += B4 * A1; C6 += B4 * A2; C7 += B4 * A3;      \
        C8 += B8 * A0; C9 += B8 * A1; Ca += B8 * A2; Cb += B8 * A3;      \
        Cc += Bc * A0; Cd += Bc * A1; Ce += Bc * A2; Cf += Bc * A3;      \
    }

// for(kk = 0; kk < col_m; kk += 4) {
//     The multiplication of A'(4x4) * B(4x4) is divided into 4 parts,
//
//     step 1 (load two upper rows of A', load two left columns of B),
//     accumulate C0, C1, C4, C5
//
//     C0, C1           +=   A0, A1, A2, A3    *   B0, B1
//     C4, C5                A4, A5, A6, A7        B4, B5
//                                                 B8, B9
//                                                 Bc, Bd
//
//     step 2 (load two lower rows of A', reuse two left columns of B),
//     accumulate C8, C9, Cc, Cd
//
//                      +=                     *   B0, B1
//                                                 B4, B5
//     C8, C9                A8, A9, Aa, Ab        B8, B9
//     Cc, Cd                Ac, Ad, Ae, Af        Bc, Bd
//
//     step 3 (reuse two lower rows of A', load two right columns of B),
//     accumulate Ca, Cb, Ce, Cf
//
//                      +=                     *           B2, B3
//                                                         B6, B7
//              Ca, Cb       A8, A9, Aa, Ab                Ba, Bb
//              Ce, Cf       Ac, Ad, Ae, Af                Be, Bf
//
//     step 4 (reload two upper rows of A', reuse two right columns of B),
//     accumulate C2, C3, C6, C7
//
//              C2, C3  +=   A0, A1, A2, A3    *           B2, B3
//              C6, C7       A4, A5, A6, A7                B6, B7
//                                                         Ba, Bb
//                                                         Be, Bf
// }
//
// if(col_r & 0x00000002) {
//     ...
// }
//
// if(col_r & 0x00000001) {
//     ...
// }
#define MTM_MAT4X4_MAIN                                                  \
    double* pa = NULL;                                                   \
    double* pb = NULL;                                                   \
    int kk = 0;                                                          \
    int ai = 0;                                                          \
    int bi = 0;                                                          \
    int col_r = ROW_A & 0x00000003;                                      \
    int col_m = ROW_A - col_r;                                           \
    double A0, A1, A2, A3;                                               \
    double A4, A5, A6, A7;                                               \
    double B0, B1;                                                       \
    double B4, B5;                                                       \
    double B8, B9;                                                       \
    double Bc, Bd;                                                       \
                                                                         \
    for(kk = 0; kk < col_m; kk += 4) {                                   \
        pa = (double*)AA + ai;                                           \
        A0 = pa[0             ]; A4 = pa[1                 ];            \
        A1 = pa[lda           ]; A5 = pa[lda            + 1];            \
        A2 = pa[      (lda<<1)]; A6 = pa[      (lda<<1) + 1];            \
        A3 = pa[lda + (lda<<1)]; A7 = pa[lda + (lda<<1) + 1];            \
                                                                         \
        pb = (double*)BB + bi;                                           \
        B0 = pb[0             ]; B1 = pb[1                 ];            \
        B4 = pb[ldb           ]; B5 = pb[ldb            + 1];            \
        B8 = pb[      (ldb<<1)]; B9 = pb[      (ldb<<1) + 1];            \
        Bc = pb[ldb + (ldb<<1)]; Bd = pb[ldb + (ldb<<1) + 1];            \
                                                                         \
        C0 += A0 * B0; C1 += A0 * B1; C4 += A4 * B0; C5 += A4 * B1;      \
        C0 += A1 * B4; C1 += A1 * B5; C4 += A5 * B4; C5 += A5 * B5;      \
        C0 += A2 * B8; C1 += A2 * B9; C4 += A6 * B8; C5 += A6 * B9;      \
        C0 += A3 * Bc; C1 += A3 * Bd; C4 += A7 * Bc; C5 += A7 * Bd;      \
                                                                         \
        pa = pa + 2;                                                     \
        A0 = pa[0             ]; A4 = pa[1                 ];            \
        A1 = pa[lda           ]; A5 = pa[lda            + 1];            \
        A2 = pa[      (lda<<1)]; A6 = pa[      (lda<<1) + 1];            \
        A3 = pa[lda + (lda<<1)]; A7 = pa[lda + (lda<<1) + 1];            \
                                                                         \
        C8 += A0 * B0; C9 += A0 * B1; Cc += A4 * B0; Cd += A4 * B1;      \
        C8 += A1 * B4; C9 += A1 * B5; Cc += A5 * B4; Cd += A5 * B5;      \
        C8 += A2 * B8; C9 += A2 * B9; Cc += A6 * B8; Cd += A6 * B9;      \
        C8 += A3 * Bc; C9 += A3 * Bd; Cc += A7 * Bc; Cd += A7 * Bd;      \
                                                                         \
        pb = pb + 2;                                                     \
        B0 = pb[0             ]; B1 = pb[1                 ];            \
        B4 = pb[ldb           ]; B5 = pb[ldb            + 1];            \
        B8 = pb[      (ldb<<1)]; B9 = pb[      (ldb<<1) + 1];            \
        Bc = pb[ldb + (ldb<<1)]; Bd = pb[ldb + (ldb<<1) + 1];            \
                                                                         \
        Ca += A0 * B0; Cb += A0 * B1; Ce += A4 * B0; Cf += A4 * B1;      \
        Ca += A1 * B4; Cb += A1 * B5; Ce += A5 * B4; Cf += A5 * B5;      \
        Ca += A2 * B8; Cb += A2 * B9; Ce += A6 * B8; Cf += A6 * B9;      \
        Ca += A3 * Bc; Cb += A3 * Bd; Ce += A7 * Bc; Cf += A7 * Bd;      \
                                                                         \
        pa = (double*)AA + ai;                                           \
        A0 = pa[0             ]; A4 = pa[1                 ];            \
        A1 = pa[lda           ]; A5 = pa[lda            + 1];            \
        A2 = pa[      (lda<<1)]; A6 = pa[      (lda<<1) + 1];            \
        A3 = pa[lda + (lda<<1)]; A7 = pa[lda + (lda<<1) + 1];            \
                                                                         \
        C2 += A0 * B0; C3 += A0 * B1; C6 += A4 * B0; C7 += A4 * B1;      \
        C2 += A1 * B4; C3 += A1 * B5; C6 += A5 * B4; C7 += A5 * B5;      \
        C2 += A2 * B8; C3 += A2 * B9; C6 += A6 * B8; C7 += A6 * B9;      \
        C2 += A3 * Bc; C3 += A3 * Bd; C6 += A7 * Bc; C7 += A7 * Bd;      \
                                                                         \
        ai += lda << 2;                                                  \
        bi += ldb << 2;                                                  \
    }                                                                    \
                                                                         \
    if(col_r & 0x00000002) {                                             \
        pa = (double*)AA + ai;                                           \
        B0 = pa[0]; B4 = pa[1]; B8 = pa[2]; Bc = pa[3]; ai += lda;       \
        pa = (double*)AA + ai;                                           \
        B1 = pa[0]; B5 = pa[1]; B9 = pa[2]; Bd = pa[3]; ai += lda;       \
                                                                         \
        pb = (double*)BB + bi;                                           \
        A0 = pb[0]; A1 = pb[1]; A2 = pb[2]; A3 = pb[3]; bi += ldb;       \
        pb = (double*)BB + bi;                                           \
        A4 = pb[0]; A5 = pb[1]; A6 = pb[2]; A7 = pb[3]; bi += ldb;       \
                                                                         \
        C0 += B0 * A0; C1 += B0 * A1; C2 += B0 * A2; C3 += B0 * A3;      \
        C0 += B1 * A4; C1 += B1 * A5; C2 += B1 * A6; C3 += B1 * A7;      \
        C4 += B4 * A0; C5 += B4 * A1; C6 += B4 * A2; C7 += B4 * A3;      \
        C4 += B5 * A4; C5 += B5 * A5; C6 += B5 * A6; C7 += B5 * A7;      \
        C8 += B8 * A0; C9 += B8 * A1; Ca += B8 * A2; Cb += B8 * A3;      \
        C8 += B9 * A4; C9 += B9 * A5; Ca += B9 * A6; Cb += B9 * A7;      \
        Cc += Bc * A0; Cd += Bc * A1; Ce += Bc * A2; Cf += Bc * A3;      \
        Cc += Bd * A4; Cd += Bd * A5; Ce += Bd * A6; Cf += Bd * A7;      \
                                                                         \
        kk += 2;                                                         \
    }                                                                    \
                                                                         \
    if(col_r & 0x00000001) {                                             \
        pa = (double*)AA + ai;                                           \
        B0 = pa[0]; B4 = pa[1]; B8 = pa[2]; Bc = pa[3];                  \
                                                                         \
        pb = (double*)BB + bi;                                           \
        A0 = pb[0]; A1 = pb[1]; A2 = pb[2]; A3 = pb[3];                  \
                                                                         \
        C0 += B0 * A0; C1 += B0 * A1; C2 += B0 * A2; C3 += B0 * A3;      \
        C4 += B4 * A0; C5 += B4 * A1; C6 += B4 * A2; C7 += B4 * A3;      \
        C8 += B8 * A0; C9 += B8 * A1; Ca += B8 * A2; Cb += B8 * A3;      \
        Cc += Bc * A0; Cd += Bc * A1; Ce += Bc * A2; Cf += Bc * A3;      \
    }

// Matrix-Matrix Multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a * B;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> c  = a * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3    =   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MMM_mat1x4_00(int COL_A, const double* AA, const double* BB,
		int ldb, double* CC) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat1x4_00_asm(COL_A, AA, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MMM_MAT1X4_MAIN;

	*CC++ = c0;
	*CC++ = c1;
	*CC++ = c2;
	*CC++ = c3;
#endif
}

// Matrix-Matrix Multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a * B;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation =  1  -> c += a * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3   +=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MMM_mat1x4_p1(int COL_A, const double* AA, const double* BB,
		int ldb, double* CC)
{
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat1x4_p1_asm(COL_A, AA, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MMM_MAT1X4_MAIN;

	*CC++ += c0;
	*CC++ += c1;
	*CC++ += c2;
	*CC++ += c3;
#endif
}

// Matrix-Matrix Multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a * B;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> c -= a * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3   -=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MMM_mat1x4_m1(int COL_A, const double* AA, const double* BB,
		int ldb, double* CC) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat1x4_m1_asm(COL_A, AA, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MMM_MAT1X4_MAIN;

	*CC++ -= c0;
	*CC++ -= c1;
	*CC++ -= c2;
	*CC++ -= c3;
#endif
}

// Matrix-Matrix Multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A * B;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> C  = A * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3    =   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MMM_mat4x4_00(int COL_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat4x4_00_asm(COL_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MMM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] = C0; CC[1] = C1; CC[2] = C2; CC[3] = C3; CC += ldc;
	CC[0] = C4; CC[1] = C5; CC[2] = C6; CC[3] = C7; CC += ldc;
	CC[0] = C8; CC[1] = C9; CC[2] = Ca; CC[3] = Cb; CC += ldc;
	CC[0] = Cc; CC[1] = Cd; CC[2] = Ce; CC[3] = Cf;
#endif
}

// Matrix-Matrix Multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A * B;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation = 1  -> C += A * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3   +=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MMM_mat4x4_p1(int COL_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat4x4_p1_asm(COL_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MMM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] += C0; CC[1] += C1; CC[2] += C2; CC[3] += C3; CC += ldc;
	CC[0] += C4; CC[1] += C5; CC[2] += C6; CC[3] += C7; CC += ldc;
	CC[0] += C8; CC[1] += C9; CC[2] += Ca; CC[3] += Cb; CC += ldc;
	CC[0] += Cc; CC[1] += Cd; CC[2] += Ce; CC[3] += Cf;
#endif
}

// Matrix-Matrix Multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A * B;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> C -= A * B
//
//  Matrix C              Matrix A                   Matrix B
//
//  C0, C1, C2, C3   -=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MMM_mat4x4_m1(int COL_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MMM_mat4x4_m1_asm(COL_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MMM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] -= C0; CC[1] -= C1; CC[2] -= C2; CC[3] -= C3; CC += ldc;
	CC[0] -= C4; CC[1] -= C5; CC[2] -= C6; CC[3] -= C7; CC += ldc;
	CC[0] -= C8; CC[1] -= C9; CC[2] -= Ca; CC[3] -= Cb; CC += ldc;
	CC[0] -= Cc; CC[1] -= Cd; CC[2] -= Ce; CC[3] -= Cf;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a' * B;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> c  = a' * B
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
//  C0, C1, C2, C3    =   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MTM_mat1x4_00(int ROW_A, const double* AA, int lda,
		const double* BB, int ldb, double* CC) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat1x4_00_asm(ROW_A, AA, lda, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTM_MAT1X4_MAIN;

	*CC++ = c0;
	*CC++ = c1;
	*CC++ = c2;
	*CC++ = c3;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a' * B;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation =  1  -> c += a' * B
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
//  C0, C1, C2, C3   +=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MTM_mat1x4_p1(int ROW_A, const double* AA, int lda,
		const double* BB, int ldb, double* CC) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat1x4_p1_asm(ROW_A, AA, lda, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTM_MAT1X4_MAIN;

	*CC++ += c0;
	*CC++ += c1;
	*CC++ += c2;
	*CC++ += c3;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 1x4 of Matrix C in one batch
//
// c op a' * B;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> c -= a' * B
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
//  C0, C1, C2, C3   -=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//                                                   B4, B5, B6, B7
//                                                   B8, B9, Ba, Bb
//                                                   Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// unroll for loops
// utilize the data resided in cache
// reuse data resided in A0~A3
static void MTM_mat1x4_m1(int ROW_A, const double* AA, int lda,
		const double* BB, int ldb, double* CC) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat1x4_m1_asm(ROW_A, AA, lda, BB, ldb, CC);
#else
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTM_MAT1X4_MAIN;

	*CC++ -= c0;
	*CC++ -= c1;
	*CC++ -= c2;
	*CC++ -= c3;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A' * B;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> C  = A' * B
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
//  Matrix C              Matrix A'                  Matrix B
//
//  C0, C1, C2, C3    =   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MTM_mat4x4_00(int ROW_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat4x4_00_asm(ROW_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MTM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] = C0; CC[1] = C1; CC[2] = C2; CC[3] = C3; CC += ldc;
	CC[0] = C4; CC[1] = C5; CC[2] = C6; CC[3] = C7; CC += ldc;
	CC[0] = C8; CC[1] = C9; CC[2] = Ca; CC[3] = Cb; CC += ldc;
	CC[0] = Cc; CC[1] = Cd; CC[2] = Ce; CC[3] = Cf;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A' * B;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation =  1  -> C += A' * B
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
//  Matrix C              Matrix A'                  Matrix B
//
//  C0, C1, C2, C3   +=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MTM_mat4x4_p1(int ROW_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat4x4_p1_asm(ROW_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MTM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] += C0; CC[1] += C1; CC[2] += C2; CC[3] += C3; CC += ldc;
	CC[0] += C4; CC[1] += C5; CC[2] += C6; CC[3] += C7; CC += ldc;
	CC[0] += C8; CC[1] += C9; CC[2] += Ca; CC[3] += Cb; CC += ldc;
	CC[0] += Cc; CC[1] += Cd; CC[2] += Ce; CC[3] += Cf;
#endif
}

// Matrix Transpose-Matrix multiplication
// Figure out 4x4 of Matrix C in one batch
//
// C op A' * B;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> C -= A' * B
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
//  Matrix C              Matrix A'                  Matrix B
//
//  C0, C1, C2, C3   -=   A0, A1, A2, A3, ...    *   B0, B1, B2, B3
//  C4, C5, C6, C7        A4, A5, A6, A7, ...        B4, B5, B6, B7
//  C8, C9, Ca, Cb        A8, A9, Aa, Ab, ...        B8, B9, Ba, Bb
//  Cc, Cd, Ce, Cf        Ac, Ad, Ae, Af, ...        Bc, Bd, Be, Bf
//                                                   . , . , . , .
//                                                   . , . , . , .
//                                                   . , . , . , .
//
// the C code is unrolled as asm-like style, only for asm template, without
// expecting performance improvment. In fact you probably get a worse performance
// if you use this call in your case.
// SMALL_BLAS_OPT_MAT4x4_ON is added for switch on/off xxx_mat4x4_xx() calls
static void MTM_mat4x4_m1(int ROW_A, const double* AA, int lda, const double* BB,
		int ldb, double* CC, int ldc) {
#ifdef SMALL_BLAS_OPT_ARM64_ARCH
	MTM_mat4x4_m1_asm(ROW_A, AA, lda, BB, ldb, CC, ldc);
#else
	double C0 = 0.0, C1 = 0.0, C2 = 0.0, C3 = 0.0;
	double C4 = 0.0, C5 = 0.0, C6 = 0.0, C7 = 0.0;
	double C8 = 0.0, C9 = 0.0, Ca = 0.0, Cb = 0.0;
	double Cc = 0.0, Cd = 0.0, Ce = 0.0, Cf = 0.0;

	MTM_MAT4X4_MAIN;

	// store 4x4 of Matrix C
	CC[0] -= C0; CC[1] -= C1; CC[2] -= C2; CC[3] -= C3; CC += ldc;
	CC[0] -= C4; CC[1] -= C5; CC[2] -= C6; CC[3] -= C7; CC += ldc;
	CC[0] -= C8; CC[1] -= C9; CC[2] -= Ca; CC[3] -= Cb; CC += ldc;
	CC[0] -= Cc; CC[1] -= Cd; CC[2] -= Ce; CC[3] -= Cf;
#endif
}

// Matrix-Vector Multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A * b;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> c  = A * b
//
//  Vector c              Matrix A                   Vector b
//
//  C0                =   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
static void MVM_mat4x1_00(int COL_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MVM_MAT4X1_MAIN;

	*CC++ = c0;
	*CC++ = c1;
	*CC++ = c2;
	*CC++ = c3;
}

// Matrix-Vector Multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A * b;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation =  1  -> c += A * b
//
//  Vector c              Matrix A                   Vector b
//
//  C0               +=   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
static void MVM_mat4x1_p1(int COL_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MVM_MAT4X1_MAIN;

	*CC++ += c0;
	*CC++ += c1;
	*CC++ += c2;
	*CC++ += c3;
}

// Matrix-Vector Multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A * b;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> c -= A * b
//
//  Vector c              Matrix A                   Vector b
//
//  C0               -=   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
static void MVM_mat4x1_m1(int COL_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MVM_MAT4X1_MAIN;

	*CC++ -= c0;
	*CC++ -= c1;
	*CC++ -= c2;
	*CC++ -= c3;
}

// Matrix Transpose-Vector multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A' * b;
// where op can be +=, -=, or =.
// xxx_00 -> kOperation =  0  -> c  = A' * b
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
//  C0                =   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
//
// unroll for loops
// utilize the data resided in cache
static void MTV_mat4x1_00(int ROW_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTV_MAT4X1_MAIN;

	*CC++ = c0;
	*CC++ = c1;
	*CC++ = c2;
	*CC++ = c3;
}

// Matrix Transpose-Vector multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A' * b;
// where op can be +=, -=, or =.
// xxx_p1 -> kOperation =  1  -> c += A' * b
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
//  C0               +=   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
// unroll for loops
// utilize the data resided in cache
static void MTV_mat4x1_p1(int ROW_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTV_MAT4X1_MAIN;

	*CC++ += c0;
	*CC++ += c1;
	*CC++ += c2;
	*CC++ += c3;
}

// Matrix Transpose-Vector multiplication
// Figure out 4x1 of vector c in one batch
//
// c op A' * b;
// where op can be +=, -=, or =.
// xxx_m1 -> kOperation = -1  -> c -= A' * b
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
//  C0               -=   A0, A1, A2, A3, ...    *   B0
//  C1                    A4, A5, A6, A7, ...        B1
//  C2                    A8, A9, Aa, Ab, ...        B2
//  C3                    Ac, Ad, Ae, Af, ...        B3
//                                                   .
//                                                   .
//                                                   .
// unroll for loops
// utilize the data resided in cache
static void MTV_mat4x1_m1(int ROW_A, const double* AA, int lda,
		const double* BB, double* CC) {
	double c0 = 0.0;
	double c1 = 0.0;
	double c2 = 0.0;
	double c3 = 0.0;

	MTV_MAT4X1_MAIN;

	*CC++ -= c0;
	*CC++ -= c1;
	*CC++ -= c2;
	*CC++ -= c3;
}

#undef MMM_MAT1X4_MAIN
#undef MTM_MAT1X4_MAIN
#undef MVM_MAT4X1_MAIN
#undef MTV_MAT4X1_MAIN
#undef MMM_MAT4X4_MAIN
#undef MTM_MAT4X4_MAIN

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SMALL_BLAS_OPT_H_
