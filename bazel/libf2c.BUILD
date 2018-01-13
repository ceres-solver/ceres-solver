# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2018 Google Inc. All rights reserved.
# http://ceres-solver.org/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Authors: mierle@gmail.com (Keir Mierle)

# Some files need to be renamed; this is just a fancy way to strip the '0' from
# the header names in the uncompressed libf2c sources.
[genrule(
    name = "copy_header_" + header_name,
    srcs = [ header_name + '.h0' ],
    outs = [ header_name + '.h' ],
    cmd = "cp $(SRCS) $@"
 )
 for header_name in ['f2c', 'signal1', 'sysdep1']]

# This is a configure check binary, than spits out the arith.h header.
cc_binary(
    name = "arithchk",
    srcs = ["arithchk.c"],
    copts = ["-DNO_FPINIT"]
)

genrule(
    name = "make_arith_h",
    outs = [ 'arith.h' ],
    cmd = "$(location arithchk) > $@",
    tools = ['arithchk']
)

MISC = [
    "f77vers.c",
    "i77vers.c",
    "s_rnge.c",
    "abort_.c",
    "exit_.c",
    "getarg_.c",
    "iargc_.c",
    "getenv_.c",
    "signal_.c",
    "s_stop.c",
    "s_paus.c",
    "cabs.c",
    "derf_.c",
    "derfc_.c",
    "erf_.c",
    "erfc_.c",
    "sig_die.c",
    "uninit.c",
]

POW = [
    "pow_ci.c",
    "pow_dd.c",
    "pow_di.c",
    "pow_hh.c",
    "pow_ii.c",
    "pow_ri.c",
    "pow_zi.c",
    "pow_zz.c",
]

CX = [
    "c_abs.c",
    "c_cos.c",
    "c_div.c",
    "c_exp.c",
    "c_log.c",
    "c_sin.c",
    "c_sqrt.c",
]

DCX = [
    "z_abs.c",
    "z_cos.c",
    "z_div.c",
    "z_exp.c",
    "z_log.c",
    "z_sin.c",
    "z_sqrt.c",
]

REAL = [
    "r_abs.c",
    "r_acos.c",
    "r_asin.c",
    "r_atan.c",
    "r_atn2.c",
    "r_cnjg.c",
    "r_cos.c",
    "r_cosh.c",
    "r_dim.c",
    "r_exp.c",
    "r_imag.c",
    "r_int.c",
    "r_lg10.c",
    "r_log.c",
    "r_mod.c",
    "r_nint.c",
    "r_sign.c",
    "r_sin.c",
    "r_sinh.c",
    "r_sqrt.c",
    "r_tan.c",
    "r_tanh.c",
]

DBL = [
    "d_abs.c",
    "d_acos.c",
    "d_asin.c",
    "d_atan.c",
    "d_atn2.c",
    "d_cnjg.c",
    "d_cos.c",
    "d_cosh.c",
    "d_dim.c",
    "d_exp.c",
    "d_imag.c",
    "d_int.c",
    "d_lg10.c",
    "d_log.c",
    "d_mod.c",
    "d_nint.c",
    "d_prod.c",
    "d_sign.c",
    "d_sin.c",
    "d_sinh.c",
    "d_sqrt.c",
    "d_tan.c",
    "d_tanh.c",
]

INT = [
    "i_abs.c",
    "i_dim.c",
    "i_dnnt.c",
    "i_indx.c",
    "i_len.c",
    "i_mod.c",
    "i_nint.c",
    "i_sign.c",
    "lbitbits.c",
    "lbitshft.c",
]

HALF = [
    "h_abs.c",
    "h_dim.c",
    "h_dnnt.c",
    "h_indx.c",
    "h_len.c",
    "h_mod.c",
    "h_nint.c",
    "h_sign.c",
]

CMP = [
    "l_ge.c",
    "l_gt.c",
    "l_le.c",
    "l_lt.c",
    "hl_ge.c",
    "hl_gt.c",
    "hl_le.c",
    "hl_lt.c",
]

EFL = [
    "ef1asc_.c",
    "ef1cmc_.c",
]

CHAR = [
    "f77_aloc.c",
    "s_cat.c",
    "s_cmp.c",
    "s_copy.c",
]

I77 = [
    "backspac.c",
    "close.c",
    "dfe.c",
    "dolio.c",
    "due.c",
    "endfile.c",
    "err.c",
    "fmt.c",
    "fmtlib.c",
    "ftell_.c",
    "iio.c",
    "ilnw.c",
    "inquire.c",
    "lread.c",
    "lwrite.c",
    "open.c",
    "rdfmt.c",
    "rewind.c",
    "rsfe.c",
    "rsli.c",
    "rsne.c",
    "sfe.c",
    "sue.c",
    "typesize.c",
    "uio.c",
    "util.c",
    "wref.c",
    "wrtfmt.c",
    "wsfe.c",
    "wsle.c",
    "wsne.c",
    "xwsne.c",
]

QINT = [
    "pow_qq.c",
    "qbitbits.c",
    "qbitshft.c",
    "ftell64_.c",
]

TIME = [
    "dtime_.c",
    "etime_.c",
]

cc_library(
    name = "f2c",
    srcs = MISC +
           POW +
           CX +
           DCX +
           REAL +
           DBL +
           INT +
           HALF +
           CMP +
           EFL +
           CHAR +
           I77 +
           TIME + [
        "arith.h",
        "f2c.h",
        "fio.h",
        "fmt.h",
        "fp.h",
        "lio.h",
        "signal1.h",
        "sysdep1.h",
    ],
    visibility = ["//visibility:public"],
)

