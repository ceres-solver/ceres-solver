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

# TODO(keir): Replace this with a better version, like from TensorFlow.
# See https://github.com/ceres-solver/ceres-solver/issues/337.
cc_library(
    name = 'eigen',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**']),
    visibility = ['//visibility:public'],
)

# TODO(keir): Need to support OpenMP, which may be used inside the Eigen BLAS.
cc_library(
    name = "blas",
    hdrs = glob(['blas/*.h']),
    srcs = [
        "Eigen/src/misc/blas.h",
        "blas/complex_double.cpp",
        "blas/complex_single.cpp",
        "blas/double.cpp",
        "blas/f2c/chbmv.c",
        "blas/f2c/chpmv.c",
        "blas/f2c/complexdots.c",
        "blas/f2c/ctbmv.c",
        "blas/f2c/drotm.c",
        "blas/f2c/drotmg.c",
        "blas/f2c/dsbmv.c",
        "blas/f2c/dspmv.c",
        "blas/f2c/dtbmv.c",
        "blas/f2c/lsame.c",
        "blas/f2c/srotm.c",
        "blas/f2c/srotmg.c",
        "blas/f2c/ssbmv.c",
        "blas/f2c/sspmv.c",
        "blas/f2c/stbmv.c",
        "blas/f2c/zhbmv.c",
        "blas/f2c/zhpmv.c",
        "blas/f2c/ztbmv.c",
        "blas/f2c/datatypes.h",
        "blas/single.cpp",
        "blas/xerbla.cpp",
    ],
    includes = [
        "blas",
        "blas/f2c",
    ],
    visibility = ["//visibility:public"],
    deps = ['eigen']
)

# DOES NOT WORK!
# Seems to trigger a Bazel bug; a failing command runs when I copy and paste
# the printed command and run it manually.
#
# Eigen implements a small but very useful subset of the LAPACK API.  It does
# not have the full abilities of LAPACK, but it is mostly C++ code, which makes
# it suitable for use on platforms where there is no FORTRAN compiler, e.g.
# Android.
cc_library(
    name = "lapack",
    hdrs = [
        "lapack/lapack_common.h"
        ],
    srcs = [
        "lapack/complex_double.cpp",
        "lapack/complex_single.cpp",
        "lapack/cholesky.cpp",
        "lapack/single.cpp",
        "lapack/double.cpp",
        "lapack/lapack_common.h",
    ] + glob(["lapack/*.c"]),
    includes = [
        ".",
        "lapack",
    ],
    visibility = ["//visibility:public"],
    deps = [
        'eigen',
        'blas',
        '@org_netlib_libf2c//:f2c',
    ],
)

# TODO libf2c
