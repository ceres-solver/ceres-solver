# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
# http://code.google.com/p/ceres-solver/
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
# Author: settinger@google.com (Scott Ettinger)
#         keir@google.com (Keir Mierle)
#
# Builds Ceres for Android, using the standard toolchain (not standalone). It
# uses STLPort instead of GNU C++. This is useful for anyone wishing to ship
# GPL-free code. This cannot build the tests or other parts of Ceres; only the
# core libraries. If you need a more complete Ceres build, consider using the
# CMake toolchain (noting that the standalone toolchain doesn't work with
# STLPort).
#
# You will have to specify the environment EIGEN_PATH to point to the Eigen
# sources when building. For example:
#
#   EIGEN_PATH=/home/keir/src/eigen-3.0.5 ndk-build -j
#
# It is also possible to specify CERES_EXTRA_DEFINES, in case you need to pass
# more definitions to the C compiler.
#
# IMPORTANT:
#
# The shared library built at the bottom is fake, broken, and empty. It exists
# only to force ndk-build to build the shared library. This shouldn't be
# necessary, but if it is missing, then ndk-build will do nothing when asked to
# build. The produced .so library is NON-FUNCTIONAL since it has no Ceres
# function-level dependencies. Instead, copy the static library:
#
#   ../obj/local/armeabi-v7a/libceres.a
#
# into your own project, then link it into your binary in your Android.mk file.
#
# Reducing binary size:
#
# This build includes the Schur specializations, which cause binary bloat. If
# you don't need them for your application, consider adding:
#
#   -DCERES_RESTRICT_SCHUR_SPECIALIZATION
#
# to the LOCAL_CFLAGS variable below, and commenting out all the
# generated/schur_eliminator_2_2_2.cc-alike files, leaving only the _d_d_d one.

LOCAL_PATH := $(call my-dir)

EIGEN_PATH := $(EIGEN_PATH)

CERES_INCLUDE_PATHS := ../internal
CERES_INCLUDE_PATHS += ../internal/ceres
CERES_INCLUDE_PATHS += ../include
CERES_INCLUDE_PATHS += ../internal/ceres/miniglog
CERES_SRC_PATH := ../internal/ceres

include $(CLEAR_VARS)
LOCAL_C_INCLUDES := $(CERES_INCLUDE_PATHS)
LOCAL_C_INCLUDES += $(EIGEN_PATH)

LOCAL_CPP_EXTENSION := .cc
LOCAL_CFLAGS := $(CERES_EXTRA_DEFINES) \
                -DCERES_NO_PROTOCOL_BUFFERS \
                -DCERES_NO_SUITESPARSE \
                -DCERES_NO_GFLAGS \
                -DCERES_NO_THREADS \
                -DCERES_NO_CXSPARSE \
                -DCERES_NO_TR1

LOCAL_SRC_FILES := $(CERES_SRC_PATH)/array_utils.cc \
                   $(CERES_SRC_PATH)/block_evaluate_preparer.cc \
                   $(CERES_SRC_PATH)/block_jacobian_writer.cc \
                   $(CERES_SRC_PATH)/block_jacobi_preconditioner.cc \
                   $(CERES_SRC_PATH)/block_random_access_dense_matrix.cc \
                   $(CERES_SRC_PATH)/block_random_access_matrix.cc \
                   $(CERES_SRC_PATH)/block_random_access_sparse_matrix.cc \
                   $(CERES_SRC_PATH)/block_sparse_matrix.cc \
                   $(CERES_SRC_PATH)/block_structure.cc \
                   $(CERES_SRC_PATH)/canonical_views_clustering.cc \
                   $(CERES_SRC_PATH)/cgnr_solver.cc \
                   $(CERES_SRC_PATH)/compressed_row_jacobian_writer.cc \
                   $(CERES_SRC_PATH)/compressed_row_sparse_matrix.cc \
                   $(CERES_SRC_PATH)/conditioned_cost_function.cc \
                   $(CERES_SRC_PATH)/conjugate_gradients_solver.cc \
                   $(CERES_SRC_PATH)/corrector.cc \
                   $(CERES_SRC_PATH)/dense_qr_solver.cc \
                   $(CERES_SRC_PATH)/dense_sparse_matrix.cc \
                   $(CERES_SRC_PATH)/detect_structure.cc \
                   $(CERES_SRC_PATH)/dogleg_strategy.cc \
                   $(CERES_SRC_PATH)/evaluator.cc \
                   $(CERES_SRC_PATH)/file.cc \
                   $(CERES_SRC_PATH)/gradient_checking_cost_function.cc \
                   $(CERES_SRC_PATH)/implicit_schur_complement.cc \
                   $(CERES_SRC_PATH)/iterative_schur_complement_solver.cc \
                   $(CERES_SRC_PATH)/levenberg_marquardt_strategy.cc \
                   $(CERES_SRC_PATH)/linear_least_squares_problems.cc \
                   $(CERES_SRC_PATH)/linear_operator.cc \
                   $(CERES_SRC_PATH)/linear_solver.cc \
                   $(CERES_SRC_PATH)/local_parameterization.cc \
                   $(CERES_SRC_PATH)/loss_function.cc \
                   $(CERES_SRC_PATH)/normal_prior.cc \
                   $(CERES_SRC_PATH)/partitioned_matrix_view.cc \
                   $(CERES_SRC_PATH)/polynomial_solver.cc \
                   $(CERES_SRC_PATH)/problem.cc \
                   $(CERES_SRC_PATH)/problem_impl.cc \
                   $(CERES_SRC_PATH)/program.cc \
                   $(CERES_SRC_PATH)/residual_block.cc \
                   $(CERES_SRC_PATH)/residual_block_utils.cc \
                   $(CERES_SRC_PATH)/runtime_numeric_diff_cost_function.cc \
                   $(CERES_SRC_PATH)/schur_complement_solver.cc \
                   $(CERES_SRC_PATH)/schur_eliminator.cc \
                   $(CERES_SRC_PATH)/schur_ordering.cc \
                   $(CERES_SRC_PATH)/scratch_evaluate_preparer.cc \
                   $(CERES_SRC_PATH)/solver.cc \
                   $(CERES_SRC_PATH)/solver_impl.cc \
                   $(CERES_SRC_PATH)/sparse_matrix.cc \
                   $(CERES_SRC_PATH)/sparse_normal_cholesky_solver.cc \
                   $(CERES_SRC_PATH)/split.cc \
                   $(CERES_SRC_PATH)/stringprintf.cc \
                   $(CERES_SRC_PATH)/suitesparse.cc \
                   $(CERES_SRC_PATH)/triplet_sparse_matrix.cc \
                   $(CERES_SRC_PATH)/trust_region_minimizer.cc \
                   $(CERES_SRC_PATH)/trust_region_strategy.cc \
                   $(CERES_SRC_PATH)/types.cc \
                   $(CERES_SRC_PATH)/visibility_based_preconditioner.cc \
                   $(CERES_SRC_PATH)/visibility.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_d_d_d.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_2_2.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_2_3.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_2_4.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_2_d.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_3_3.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_3_4.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_3_9.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_3_d.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_4_3.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_4_4.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_2_4_d.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_4_4_2.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_4_4_3.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_4_4_4.cc \
                   $(CERES_SRC_PATH)/generated/schur_eliminator_4_4_d.cc

LOCAL_MODULE := ceres
include $(BUILD_STATIC_LIBRARY)

# This is a fake library; see the file header comments.
include $(CLEAR_VARS)
LOCAL_C_INCLUDES := $(CERES_INCLUDE_PATHS)
LOCAL_C_INCLUDES += $(EIGEN_PATH)
LOCAL_MODULE := forces_static_ceres_build_do_not_use
LOCAL_STATIC_LIBRARIES := ceres
include $(BUILD_SHARED_LIBRARY)
