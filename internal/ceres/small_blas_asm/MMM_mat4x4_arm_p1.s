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
// These codes aim to arm64-v8a platform, speeding up 64-bit float small
// matrices multiplication which can not utilize Eigen math library
// because of dynamic parameters.

.text
.align 4
.global MMM_mat4x4_p1_asm
//////////////////////////////////////////////////////
// void MMM_mat4x4_p1_asm(int COL_A,
//                       const double* A, int lda,
//                       const double* B, int ldb,
//                       double* C, int ldc);
//
// x0: COL_A & the stride of Matrix A
// x1: A &  the pointer to Matrix A
// x2: lda & the stride of Matrix A
// x3: B &  the pointer to Matrix B
// x4: ldb & the stride of Matrix B
// x5: C &  the pointer to Matrix C
//
// p1: plus 1,  means C[index]+= tmp;
// m1: minus 1, means C[index]-= tmp;
// 00: zero,    means C[index] = tmp;
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
/////////////////////////////////////////////////////

// Refer to chapter 5.1.1 from
// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
//
// x9...x15: Temporary registers
// x8:       Indirect result location register
// x0...x7:  Parameter/result registers
//
// We can use these registers freely without saving/restoring their values

COL_A .req x0
A     .req x1
lda   .req x2
B     .req x3
ldb   .req x4
C     .req x5
ldc   .req x6

bi    .req x7
col_r .req x8
col_m .req x9
kk    .req x10
pa    .req x11
pb    .req x12

sc    .req x7  // reuse of x7
xx    .req x0  // reuse of x0

A0    .req d0
A1    .req d1
A2    .req d2
A3    .req d3
A4    .req d4
A5    .req d5
A6    .req d6
A7    .req d7

M0    .req d8
M1    .req d9
M4    .req d10
M5    .req d11
M8    .req d12
M9    .req d13
Mc    .req d14
Md    .req d15

C0    .req d16
C1    .req d17
C2    .req d18
C3    .req d19
C4    .req d20
C5    .req d21
C6    .req d22
C7    .req d23
C8    .req d24
C9    .req d25
Ca    .req d26
Cb    .req d27
Cc    .req d28
Cd    .req d29
Ce    .req d30
Cf    .req d31

MMM_mat4x4_p1_asm:

	// Registers v8-v15 must be preserved by a callee across subroutine calls
	// Refer to chapter 5.1.2 from
	// http://infocenter.arm.com/help/topic/com.arm.doc.ihi0055b/IHI0055B_aapcs64.pdf
	stp   d8 ,d9 ,[sp, #-16]
	stp   d10,d11,[sp, #-32]
	stp   d12,d13,[sp, #-48]
	stp   d14,d15,[sp, #-64]

	// load Matrix C 4x4 into Cx
	mov   sc, C
	ldp   C0, C1, [sc]
	ldp   C2, C3, [sc, #16]
	add   sc, sc, ldc, LSL #3          // sc += ldc
	ldp   C4, C5, [sc]
	ldp   C6, C7, [sc, #16]
	add   sc, sc, ldc, LSL #3          // sc += ldc
	ldp   C8, C9, [sc]
	ldp   Ca, Cb, [sc, #16]
	add   sc, sc, ldc, LSL #3          // sc += ldc
	ldp   Cc, Cd, [sc]
	ldp   Ce, Cf, [sc, #16]

	and   col_r, COL_A, #3
	sub   col_m, COL_A, col_r
	mov   bi, #0
	mov   kk, #0

loop_4:
	cmp   kk, col_m
	b.GE  loop_4_end                   // finish the loop if kk >= col_m

	// The multiplication of A(4x4) * B(4x4) is divided into 4 parts,
	// step 1 (load two upper rows of A, load two left columns of B),
	// accumulate C0, C1, C4, C5
	//
	//    C0, C1           +=   A0, A1, A2, A3    *   B0, B1
	//    C4, C5                A4, A5, A6, A7        B4, B5
	//                                                B8, B9
	//                                                Bc, Bd
	add   pa, A , kk, LSL #3           // pa = A + kk
	ldp   A0, A1, [pa]
	ldp   A2, A3, [pa, #16]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   A4, A5, [pa]
	ldp   A6, A7, [pa, #16]
	add   pa, pa, lda, LSL #3          // pa += lda

	add   pb, B , bi, LSL #3           // pb = B + bi
	ldp   M0, M1, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   M4, M5, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   M8, M9, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   Mc, Md, [pb]

	fmadd C0, A0, M0, C0
	fmadd C1, A0, M1, C1
	fmadd C4, A4, M0, C4
	fmadd C5, A4, M1, C5

	fmadd C0, A1, M4, C0
	fmadd C1, A1, M5, C1
	fmadd C4, A5, M4, C4
	fmadd C5, A5, M5, C5

	fmadd C0, A2, M8, C0
	fmadd C1, A2, M9, C1
	fmadd C4, A6, M8, C4
	fmadd C5, A6, M9, C5

	fmadd C0, A3, Mc, C0
	fmadd C1, A3, Md, C1
	fmadd C4, A7, Mc, C4
	fmadd C5, A7, Md, C5

	// step 2 (load two lower rows of A, reuse two left columns of B),
	// registers are not enough, here we reuse Ax
	// accumulate C8, C9, Cc, Cd
	//
	//                     +=                     *   B0, B1
	//                                                B4, B5
	//    C8, C9                A8, A9, Aa, Ab        B8, B9
	//    Cc, Cd                Ac, Ad, Ae, Af        Bc, Bd
	ldp   A0, A1, [pa]
	ldp   A2, A3, [pa, #16]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   A4, A5, [pa]
	ldp   A6, A7, [pa, #16]
	add   pa, pa, lda, LSL #3          // pa += lda

	fmadd C8, A0, M0, C8
	fmadd C9, A0, M1, C9
	fmadd Cc, A4, M0, Cc
	fmadd Cd, A4, M1, Cd

	fmadd C8, A1, M4, C8
	fmadd C9, A1, M5, C9
	fmadd Cc, A5, M4, Cc
	fmadd Cd, A5, M5, Cd

	fmadd C8, A2, M8, C8
	fmadd C9, A2, M9, C9
	fmadd Cc, A6, M8, Cc
	fmadd Cd, A6, M9, Cd

	fmadd C8, A3, Mc, C8
	fmadd C9, A3, Md, C9
	fmadd Cc, A7, Mc, Cc
	fmadd Cd, A7, Md, Cd

	// step 3 (reuse two lower rows of A, load two right columns of B),
	// registers are not enough, here we reuse Bx
	// accumulate Ca, Cb, Ce, Cf
	//
	//                    +=                     *           B2, B3
	//                                                       B6, B7
	//            Ca, Cb       A8, A9, Aa, Ab                Ba, Bb
	//            Ce, Cf       Ac, Ad, Ae, Af                Be, Bf
	add   pb, B , bi, LSL #3           // pb = B + bi
	add   pb, pb, #16                  // pb += 2

	ldp   M0, M1, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   M4, M5, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   M8, M9, [pb]
	add   pb, pb, ldb, LSL #3          // pb += ldb
	ldp   Mc, Md, [pb]

	fmadd Ca, A0, M0, Ca
	fmadd Cb, A0, M1, Cb
	fmadd Ce, A4, M0, Ce
	fmadd Cf, A4, M1, Cf

	fmadd Ca, A1, M4, Ca
	fmadd Cb, A1, M5, Cb
	fmadd Ce, A5, M4, Ce
	fmadd Cf, A5, M5, Cf

	fmadd Ca, A2, M8, Ca
	fmadd Cb, A2, M9, Cb
	fmadd Ce, A6, M8, Ce
	fmadd Cf, A6, M9, Cf

	fmadd Ca, A3, Mc, Ca
	fmadd Cb, A3, Md, Cb
	fmadd Ce, A7, Mc, Ce
	fmadd Cf, A7, Md, Cf

	// step 4 (reload two upper rows of A, reuse two right columns of B),
	// accumulate C2, C3, C6, C7
	//
	//            C2, C3  +=   A0, A1, A2, A3    *           B2, B3
	//            C6, C7       A4, A5, A6, A7                B6, B7
	//                                                       Ba, Bb
	//                                                       Be, Bf
	add   pa, A , kk, LSL #3           // pa = A + kk
	ldp   A0, A1, [pa]
	ldp   A2, A3, [pa, #16]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   A4, A5, [pa]
	ldp   A6, A7, [pa, #16]

	fmadd C2, A0, M0, C2
	fmadd C3, A0, M1, C3
	fmadd C6, A4, M0, C6
	fmadd C7, A4, M1, C7
	add   bi, bi, ldb                 // bi += ldb

	fmadd C2, A1, M4, C2
	fmadd C3, A1, M5, C3
	fmadd C6, A5, M4, C6
	fmadd C7, A5, M5, C7
	add   bi, bi, ldb                 // bi += ldb

	fmadd C2, A2, M8, C2
	fmadd C3, A2, M9, C3
	fmadd C6, A6, M8, C6
	fmadd C7, A6, M9, C7
	add   bi, bi, ldb                 // bi += ldb

	fmadd C2, A3, Mc, C2
	fmadd C3, A3, Md, C3
	add   bi, bi, ldb                 // bi += ldb
	add   kk, kk, #4                  // kk += 4
	fmadd C6, A7, Mc, C6
	fmadd C7, A7, Md, C7

	b     loop_4

loop_4_end:
	and   xx, col_r, #2
	cmp   xx, #2
	b.NE  mod2_end

	// load two columns in Matrix A 4x4 to Bx
	//    M0, M1,
	//    M4, M5,
	//    M8, M9,
	//    Mc, Md,
	add   pa, A , kk , LSL #3          // pa = A + kk
	ldp   M0, M1, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   M4, M5, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   M8, M9, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   Mc, Md, [pa]

	// load two rows in Matrix B 4x4 to Ax
	//    A0, A1, A2, A3,
	//    A4, A5, A6, A7,
	add   pb, B , bi, LSL #3           // pb = B + bi
	add   bi, bi, ldb                  // bi += ldb
	ldp   A0, A1, [pb]
	ldp   A2, A3, [pb, #16]
	add   pb, B , bi, LSL #3           // pb = B + bi
	add   bi, bi, ldb                  // bi += ldb
	ldp   A4, A5, [pb]
	ldp   A6, A7, [pb, #16]

	// accumulate C0, C1, C2, C3
	fmadd C0, M0, A0, C0
	fmadd C1, M0, A1, C1
	fmadd C2, M0, A2, C2
	fmadd C3, M0, A3, C3

	fmadd C0, M1, A4, C0
	fmadd C1, M1, A5, C1
	fmadd C2, M1, A6, C2
	fmadd C3, M1, A7, C3

	// accumulate C4, C5, C6, C7
	fmadd C4, M4, A0, C4
	fmadd C5, M4, A1, C5
	fmadd C6, M4, A2, C6
	fmadd C7, M4, A3, C7

	fmadd C4, M5, A4, C4
	fmadd C5, M5, A5, C5
	fmadd C6, M5, A6, C6
	fmadd C7, M5, A7, C7

	// accumulate C8, C9, Ca, Cb
	fmadd C8, M8, A0, C8
	fmadd C9, M8, A1, C9
	fmadd Ca, M8, A2, Ca
	fmadd Cb, M8, A3, Cb

	fmadd C8, M9, A4, C8
	fmadd C9, M9, A5, C9
	fmadd Ca, M9, A6, Ca
	fmadd Cb, M9, A7, Cb

	add   kk, kk, #2                   // kk += 2

	// accumulate Cc, Cd, Ce, Cf
	fmadd Cc, Mc, A0, Cc
	fmadd Cd, Mc, A1, Cd
	fmadd Ce, Mc, A2, Ce
	fmadd Cf, Mc, A3, Cf

	fmadd Cc, Md, A4, Cc
	fmadd Cd, Md, A5, Cd
	fmadd Ce, Md, A6, Ce
	fmadd Cf, Md, A7, Cf

mod2_end:
	and   xx, col_r, #1
	cmp   xx, #1
	b.NE  mod1_end

	// load last column in Matrix A 4x4 to Mx
	//    M0,
	//    M4,
	//    M8,
	//    Mc,
	add   pa, A , kk , LSL #3          // pa = A + kk
	ldp   M0, M1, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   M4, M5, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   M8, M9, [pa]
	add   pa, pa, lda, LSL #3          // pa += lda
	ldp   Mc, Md, [pa]

	// load last row in Matrix B 4x4 to Ax
	//    A0, A1, A2, A3,
	add   pb, B , bi, LSL #3           // pb = B + bi
	ldp   A0, A1, [pb]
	ldp   A2, A3, [pb, #16]

	// accumulate C0, C1, C2, C3
	fmadd C0, M0, A0, C0
	fmadd C1, M0, A1, C1
	fmadd C2, M0, A2, C2
	fmadd C3, M0, A3, C3
	// accumulate C4, C5, C6, C7
	fmadd C4, M4, A0, C4
	fmadd C5, M4, A1, C5
	fmadd C6, M4, A2, C6
	fmadd C7, M4, A3, C7
	// accumulate C8, C9, Ca, Cb
	fmadd C8, M8, A0, C8
	fmadd C9, M8, A1, C9
	fmadd Ca, M8, A2, Ca
	fmadd Cb, M8, A3, Cb
	// accumulate Cc, Cd, Ce, Cf
	fmadd Cc, Mc, A0, Cc
	fmadd Cd, Mc, A1, Cd
	fmadd Ce, Mc, A2, Ce
	fmadd Cf, Mc, A3, Cf

mod1_end:

	// restore v8-v15
	ldp   d8 ,d9 ,[sp, #-16]
	ldp   d10,d11,[sp, #-32]
	ldp   d12,d13,[sp, #-48]
	ldp   d14,d15,[sp, #-64]

	// write back matrix C
	stp   C0, C1, [C]
	stp   C2, C3, [C, #16]
	add   C , C , ldc, LSL #3          // C += ldc
	stp   C4, C5, [C]
	stp   C6, C7, [C, #16]
	add   C , C , ldc, LSL #3          // C += ldc
	stp   C8, C9, [C]
	stp   Ca, Cb, [C, #16]
	add   C , C , ldc, LSL #3          // C += ldc
	stp   Cc, Cd, [C]
	stp   Ce, Cf, [C, #16]

	ret
