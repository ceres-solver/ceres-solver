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

# DOES NOT WORK: WIP ONLY

cc_library(
    name = "suitesparseconfig",
    srcs = ["SuiteSparse_config/SuiteSparse_config.c"],
    hdrs = ["SuiteSparse_config/SuiteSparse_config.h"],
)

AMD_srcs = [
    "AMD/Source/amd_1.c",
    "AMD/Source/amd_2.c",
    "AMD/Source/amd_aat.c",
    "AMD/Source/amd_control.c",
    "AMD/Source/amd_defaults.c",
    "AMD/Source/amd_dump.c",
    "AMD/Source/amd_info.c",
    "AMD/Source/amd_order.c",
    "AMD/Source/amd_post_tree.c",
    "AMD/Source/amd_postorder.c",
    "AMD/Source/amd_preprocess.c",
    "AMD/Source/amd_valid.c",
]

# Parts of the library are shared between the integer and long integer versions.
cc_library(
    name = "amd_common",
    srcs = ["AMD/Source/amd_global.c"],
    visibility = ["//visibility:private"],
)

# Integer version of the AMD library
cc_library(
    name = "amd_int",  # version 2.2.0
    srcs = AMD_srcs + [],
    hdrs = [
        "AMD/Include/amd.h",
        "AMD/Include/amd_internal.h",
    ],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
    ],
    includes = [
        "AMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":amd_common",
        ":suitesparseconfig",
    ],
)

# Long integer version of the AMD library
cc_library(
    name = "amd_long",  # version 2.2.0
    srcs = AMD_srcs + [],
    hdrs = [
        "AMD/Include/amd.h",
        "AMD/Include/amd_internal.h",
    ],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
    ],
    includes = [
        "AMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":amd_common",
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "amd",
    deps = [
        ":amd_int",
        ":amd_long",
    ],
)

COLAMD_srcs = ["COLAMD/Source/colamd.c"]

cc_library(
    name = "colamd_int",  # version 2.7.0
    srcs = COLAMD_srcs + ["COLAMD/Include/colamd.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
    ],
    includes = [
        "COLAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "colamd_long",  # version 2.7.0
    srcs = COLAMD_srcs + ["COLAMD/Include/colamd.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
    ],
    includes = [
        "COLAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "colamd",
    deps = [
        ":colamd_int",
        ":colamd_long",
    ],
)

CAMD_srcs = [
    "CAMD/Source/camd_1.c",
    "CAMD/Source/camd_control.c",
    "CAMD/Source/camd_postorder.c",
    "CAMD/Source/camd_2.c",
    "CAMD/Source/camd_defaults.c",
    "CAMD/Source/camd_info.c",
    "CAMD/Source/camd_preprocess.c",
    "CAMD/Source/camd_aat.c",
    "CAMD/Source/camd_dump.c",
    "CAMD/Source/camd_order.c",
    "CAMD/Source/camd_valid.c",
]

cc_library(
    name = "camd_common",
    srcs = ["CAMD/Source/camd_global.c"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "camd_int",
    srcs = CAMD_srcs + [
        "CAMD/Include/camd.h",
        "CAMD/Include/camd_internal.h",
    ],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
    ],
    includes = [
        "CAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":camd_common",
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "camd_long",
    srcs = CAMD_srcs + [
        "CAMD/Include/camd.h",
        "CAMD/Include/camd_internal.h",
    ],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
    ],
    includes = [
        "CAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":camd_common",
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "camd",
    deps = [
        ":camd_int",
        ":camd_long",
    ],
)

CCOLAMD_srcs = ["CCOLAMD/Source/ccolamd.c"]

cc_library(
    name = "ccolamd_int",
    srcs = CCOLAMD_srcs + ["CCOLAMD/Include/ccolamd.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
    ],
    includes = [
        "CCOLAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "ccolamd_long",  # version 2.7.0
    srcs = CCOLAMD_srcs + ["CCOLAMD/Include/ccolamd.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
    ],
    includes = [
        "CCOLAMD/Include",
        "SuiteSparse_config",
    ],
    linkopts = ["-lm"],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "ccolamd",
    deps = [
        ":ccolamd_int",
        ":ccolamd_long",
    ],
)

CHOLMOD_srcs = [
    "CHOLMOD/Core/cholmod_aat.c",
    "CHOLMOD/Core/cholmod_complex.c",
    "CHOLMOD/Core/cholmod_memory.c",
    "CHOLMOD/Core/cholmod_add.c",
    "CHOLMOD/Core/cholmod_copy.c",
    "CHOLMOD/Core/cholmod_sparse.c",
    "CHOLMOD/Core/cholmod_band.c",
    "CHOLMOD/Core/cholmod_dense.c",
    "CHOLMOD/Core/cholmod_transpose.c",
    "CHOLMOD/Core/cholmod_change_factor.c",
    "CHOLMOD/Core/cholmod_error.c",
    "CHOLMOD/Core/cholmod_triplet.c",
    "CHOLMOD/Core/cholmod_common.c",
    "CHOLMOD/Core/cholmod_factor.c",
    "CHOLMOD/Check/cholmod_check.c",
    "CHOLMOD/Check/cholmod_read.c",
    "CHOLMOD/Check/cholmod_write.c",
    "CHOLMOD/Cholesky/cholmod_amd.c",
    "CHOLMOD/Cholesky/cholmod_rcond.c",
    "CHOLMOD/Cholesky/cholmod_analyze.c",
    "CHOLMOD/Cholesky/cholmod_resymbol.c",
    "CHOLMOD/Cholesky/cholmod_colamd.c",
    "CHOLMOD/Cholesky/cholmod_rowcolcounts.c",
    "CHOLMOD/Cholesky/cholmod_etree.c",
    "CHOLMOD/Cholesky/cholmod_rowfac.c",
    "CHOLMOD/Cholesky/cholmod_factorize.c",
    "CHOLMOD/Cholesky/cholmod_solve.c",
    "CHOLMOD/Cholesky/cholmod_postorder.c",
    "CHOLMOD/Cholesky/cholmod_spsolve.c",
    "CHOLMOD/MatrixOps/cholmod_drop.c",
    "CHOLMOD/MatrixOps/cholmod_scale.c",
    "CHOLMOD/MatrixOps/cholmod_submatrix.c",
    "CHOLMOD/MatrixOps/cholmod_horzcat.c",
    "CHOLMOD/MatrixOps/cholmod_sdmult.c",
    "CHOLMOD/MatrixOps/cholmod_symmetry.c",
    "CHOLMOD/MatrixOps/cholmod_norm.c",
    "CHOLMOD/MatrixOps/cholmod_ssmult.c",
    "CHOLMOD/MatrixOps/cholmod_vertcat.c",
    "CHOLMOD/Partition/cholmod_camd.c",
    "CHOLMOD/Partition/cholmod_csymamd.c",
    "CHOLMOD/Partition/cholmod_nesdis.c",
    "CHOLMOD/Partition/cholmod_ccolamd.c",
    "CHOLMOD/Partition/cholmod_metis.c",
    "CHOLMOD/Supernodal/cholmod_super_numeric.c",
    "CHOLMOD/Supernodal/cholmod_super_symbolic.c",
    "CHOLMOD/Supernodal/cholmod_super_solve.c",
    "CHOLMOD/Include/cholmod_camd.h",
    "CHOLMOD/Include/cholmod_check.h",
    "CHOLMOD/Include/cholmod_cholesky.h",
    "CHOLMOD/Include/cholmod_complexity.h",
    "CHOLMOD/Include/cholmod_core.h",
    "CHOLMOD/Include/cholmod_internal.h",
    "CHOLMOD/Include/cholmod_matrixops.h",
    "CHOLMOD/Include/cholmod_modify.h",
    "CHOLMOD/Include/cholmod_partition.h",
    "CHOLMOD/Include/cholmod_supernodal.h",
]

CHOLMOD_includes = [
    "AMD/Include",
    "AMD/Source",
    "COLAMD/Include",
    "CCOLAMD/Include",
    "CAMD/Include",
    "CHOLMOD/Include",
    "CHOLMOD/Cholesky",
    "CHOLMOD/Core",
    "CHOLMOD/Include",
    "CHOLMOD/MatrixOps",
    "CHOLMOD/Modify",
    "CHOLMOD/Supernodal",
    "SuiteSparse_config",
]

CHOLMOD_gpl_srcs = [
    "CHOLMOD/Modify/cholmod_rowadd.c",
    "CHOLMOD/Modify/cholmod_updown.c",
    "CHOLMOD/Modify/cholmod_rowdel.c",
]

CHOLMOD_gpl_includes = ["CHOLMOD/Modify"]

CHOLMOD_deps = [
    ":amd",
    ":colamd",
    ":ccolamd",
    ":camd",
    ":suitesparseconfig",
    "//third_party/lapack:lapack",
]

cc_library(
    name = "cholmod_int",
    srcs = CHOLMOD_srcs,
    hdrs = ["CHOLMOD/Include/cholmod.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
        "-DNMODIFY",
        "-DNPARTITION",
    ],
    includes = CHOLMOD_includes,
    nocopts = "-Werror",
    deps = CHOLMOD_deps,
)

cc_library(
    name = "cholmod_long",
    srcs = CHOLMOD_srcs,
    hdrs = ["CHOLMOD/Include/cholmod.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
        "-DNMODIFY",
        "-DNPARTITION",
    ],
    includes = CHOLMOD_includes,
    nocopts = "-Werror",
    deps = CHOLMOD_deps,
)

cc_library(
    name = "cholmod",
    hdrs = [
        "CHOLMOD/Include/cholmod.h",
    ],
    deps = [
        ":cholmod_int",
        ":cholmod_long",
    ],
)

cc_library(
    name = "cholmod_int_gpl",
    srcs = CHOLMOD_gpl_srcs,
    hdrs = ["CHOLMOD/Include/cholmod.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDINT",
        "-DNPARTITION",
    ],
    includes = CHOLMOD_includes + CHOLMOD_gpl_includes,
    nocopts = "-Werror",
    visibility = [":cholmod_gpl_friends"],
    deps = CHOLMOD_deps + [":cholmod_int"],
)

cc_library(
    name = "cholmod_long_gpl",
    srcs = CHOLMOD_gpl_srcs,
    hdrs = ["CHOLMOD/Include/cholmod.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DDLONG",
        "-DNPARTITION",
    ],
    includes = CHOLMOD_includes + CHOLMOD_gpl_includes,
    nocopts = "-Werror",
    visibility = [":cholmod_gpl_friends"],
    deps = CHOLMOD_deps + [":cholmod_long"],
)

# CHOLMOD
cc_library(
    name = "cholmod_gpl",
    hdrs = [
        "CHOLMOD/Include/cholmod.h",
    ],
    visibility = [":cholmod_gpl_friends"],
    deps = [
        ":cholmod_int_gpl",
        ":cholmod_long_gpl",
    ],
)

SPQR_srcs = [
    "SPQR/Source/spqr_rmap.cpp",
    "SPQR/Source/SuiteSparseQR_C.cpp",
    "SPQR/Source/SuiteSparseQR_expert.cpp",
    "SPQR/Source/spqr_parallel.cpp",
    "SPQR/Source/spqr_kernel.cpp",
    "SPQR/Source/spqr_analyze.cpp",
    "SPQR/Source/spqr_assemble.cpp",
    "SPQR/Source/spqr_cpack.cpp",
    "SPQR/Source/spqr_csize.cpp",
    "SPQR/Source/spqr_fcsize.cpp",
    "SPQR/Source/spqr_debug.cpp",
    "SPQR/Source/spqr_front.cpp",
    "SPQR/Source/spqr_factorize.cpp",
    "SPQR/Source/spqr_freenum.cpp",
    "SPQR/Source/spqr_freesym.cpp",
    "SPQR/Source/spqr_freefac.cpp",
    "SPQR/Source/spqr_fsize.cpp",
    "SPQR/Source/spqr_maxcolnorm.cpp",
    "SPQR/Source/spqr_rconvert.cpp",
    "SPQR/Source/spqr_rcount.cpp",
    "SPQR/Source/spqr_rhpack.cpp",
    "SPQR/Source/spqr_rsolve.cpp",
    "SPQR/Source/spqr_stranspose1.cpp",
    "SPQR/Source/spqr_stranspose2.cpp",
    "SPQR/Source/spqr_hpinv.cpp",
    "SPQR/Source/spqr_1fixed.cpp",
    "SPQR/Source/spqr_1colamd.cpp",
    "SPQR/Source/SuiteSparseQR.cpp",
    "SPQR/Source/spqr_1factor.cpp",
    "SPQR/Source/spqr_cumsum.cpp",
    "SPQR/Source/spqr_shift.cpp",
    "SPQR/Source/spqr_happly.cpp",
    "SPQR/Source/spqr_panel.cpp",
    "SPQR/Source/spqr_happly_work.cpp",
    "SPQR/Source/SuiteSparseQR_qmult.cpp",
    "SPQR/Source/spqr_trapezoidal.cpp",
    "SPQR/Source/spqr_larftb.cpp",
    "SPQR/Source/spqr_append.cpp",
    "SPQR/Source/spqr_type.cpp",
    "SPQR/Source/spqr_tol.cpp",
]

# SPQR
# To use this library from a C++ program:
#   #include "third_party/SuiteSparse/SPQR/Include/SuiteSparseQR.hpp"
cc_library(
    name = "spqr",
    srcs = SPQR_srcs + [
        "SPQR/Include/SuiteSparseQR_C.h",
        "SPQR/Include/spqr.hpp",
    ],
    copts = [
        "-w",
        "-DNPARTITION",
    ],
    includes = [
        "CHOLMOD/Include",
        "SPQR/Include",
        "SuiteSparse_config",
    ],
    nocopts = "-Werror",
    deps = [
        ":cholmod",
        ":suitesparseconfig",
        "//third_party/lapack",
    ],
)

CXSPARSE_srcs = [
    "CXSparse/Source/cs_add.c",
    "CXSparse/Source/cs_amd.c",
    "CXSparse/Source/cs_chol.c",
    "CXSparse/Source/cs_cholsol.c",
    "CXSparse/Source/cs_compress.c",
    "CXSparse/Source/cs_convert.c",
    "CXSparse/Source/cs_counts.c",
    "CXSparse/Source/cs_cumsum.c",
    "CXSparse/Source/cs_dfs.c",
    "CXSparse/Source/cs_dmperm.c",
    "CXSparse/Source/cs_droptol.c",
    "CXSparse/Source/cs_dropzeros.c",
    "CXSparse/Source/cs_dupl.c",
    "CXSparse/Source/cs_entry.c",
    "CXSparse/Source/cs_ereach.c",
    "CXSparse/Source/cs_etree.c",
    "CXSparse/Source/cs_fkeep.c",
    "CXSparse/Source/cs_gaxpy.c",
    "CXSparse/Source/cs_happly.c",
    "CXSparse/Source/cs_house.c",
    "CXSparse/Source/cs_ipvec.c",
    "CXSparse/Source/cs_leaf.c",
    "CXSparse/Source/cs_load.c",
    "CXSparse/Source/cs_lsolve.c",
    "CXSparse/Source/cs_ltsolve.c",
    "CXSparse/Source/cs_lu.c",
    "CXSparse/Source/cs_lusol.c",
    "CXSparse/Source/cs_malloc.c",
    "CXSparse/Source/cs_maxtrans.c",
    "CXSparse/Source/cs_multiply.c",
    "CXSparse/Source/cs_norm.c",
    "CXSparse/Source/cs_permute.c",
    "CXSparse/Source/cs_pinv.c",
    "CXSparse/Source/cs_post.c",
    "CXSparse/Source/cs_print.c",
    "CXSparse/Source/cs_pvec.c",
    "CXSparse/Source/cs_qr.c",
    "CXSparse/Source/cs_qrsol.c",
    "CXSparse/Source/cs_randperm.c",
    "CXSparse/Source/cs_reach.c",
    "CXSparse/Source/cs_scatter.c",
    "CXSparse/Source/cs_scc.c",
    "CXSparse/Source/cs_schol.c",
    "CXSparse/Source/cs_spsolve.c",
    "CXSparse/Source/cs_sqr.c",
    "CXSparse/Source/cs_symperm.c",
    "CXSparse/Source/cs_tdfs.c",
    "CXSparse/Source/cs_transpose.c",
    "CXSparse/Source/cs_updown.c",
    "CXSparse/Source/cs_usolve.c",
    "CXSparse/Source/cs_util.c",
    "CXSparse/Source/cs_utsolve.c",
]

cc_library(
    name = "cxsparse_int",
    srcs = CXSPARSE_srcs + ["CXSparse/Include/cs.h"],
    copts = [
        "-w",
        "-fexceptions",
    ],
    includes = [
        "CXSparse/Include",
        "SuiteSparse_config",
    ],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "cxsparse_long",
    srcs = CXSPARSE_srcs + ["CXSparse/Include/cs.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DCS_LONG",
    ],
    includes = [
        "CXSparse/Include",
        "SuiteSparse_config",
    ],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "cxsparse_complex_int",
    srcs = CXSPARSE_srcs + ["CXSparse/Include/cs.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DCS_COMPLEX",
    ],
    includes = [
        "CXSparse/Include",
        "SuiteSparse_config",
    ],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

cc_library(
    name = "cxsparse_complex_long",
    srcs = CXSPARSE_srcs + [],
    hdrs = ["CXSparse/Include/cs.h"],
    copts = [
        "-w",
        "-fexceptions",
        "-DCS_COMPLEX",
        "-DCS_LONG",
    ],
    includes = [
        "CXSparse/Include",
        "SuiteSparse_config",
    ],
    nocopts = "-Werror",
    deps = [
        ":suitesparseconfig",
    ],
)

# CXSparse
cc_library(
    name = "cxsparse",
    srcs = CXSPARSE_srcs + [],
    visibility = ["//visibility:public"],
    hdrs = ["CXSparse/Include/cs.h"],
    deps = [
        ":cxsparse_complex_int",
        ":cxsparse_complex_long",
        ":cxsparse_int",
        ":cxsparse_long",
    ],
)
