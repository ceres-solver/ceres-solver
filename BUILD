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
# Author: mierle@gmail.com (Keir Mierle)
#
# These are Bazel rules to build Ceres. It's currently in Alpha state, and does
# not support parameterization around threading choice or sparse backends.

CERES_SRCS = ['internal/ceres/' + filename for filename in [
    'array_utils.cc',
    'blas.cc',
    'block_evaluate_preparer.cc',
    'block_jacobian_writer.cc',
    'block_jacobi_preconditioner.cc',
    'block_random_access_dense_matrix.cc',
    'block_random_access_diagonal_matrix.cc',
    'block_random_access_matrix.cc',
    'block_random_access_sparse_matrix.cc',
    'block_sparse_matrix.cc',
    'block_structure.cc',
    'c_api.cc',
    'callbacks.cc',
    'canonical_views_clustering.cc',
    'cgnr_solver.cc',
    'compressed_col_sparse_matrix_utils.cc',
    'compressed_row_jacobian_writer.cc',
    'compressed_row_sparse_matrix.cc',
    'conditioned_cost_function.cc',
    'conjugate_gradients_solver.cc',
    'coordinate_descent_minimizer.cc',
    'corrector.cc',
    'covariance.cc',
    'covariance_impl.cc',
    'dense_normal_cholesky_solver.cc',
    'dense_qr_solver.cc',
    'dense_sparse_matrix.cc',
    'detect_structure.cc',
    'dogleg_strategy.cc',
    'dynamic_compressed_row_jacobian_writer.cc',
    'dynamic_compressed_row_sparse_matrix.cc',
    'dynamic_sparse_normal_cholesky_solver.cc',
    'eigensparse.cc',
    'evaluator.cc',
    'file.cc',
    'function_sample.cc',
    'gradient_checker.cc',
    'gradient_checking_cost_function.cc',
    'gradient_problem.cc',
    'gradient_problem_solver.cc',
    'is_close.cc',
    'implicit_schur_complement.cc',
    'inner_product_computer.cc',
    'iterative_schur_complement_solver.cc',
    'lapack.cc',
    'levenberg_marquardt_strategy.cc',
    'line_search.cc',
    'line_search_direction.cc',
    'line_search_minimizer.cc',
    'linear_least_squares_problems.cc',
    'linear_operator.cc',
    'line_search_preprocessor.cc',
    'linear_solver.cc',
    'local_parameterization.cc',
    'loss_function.cc',
    'low_rank_inverse_hessian.cc',
    'minimizer.cc',
    'normal_prior.cc',
    'parameter_block_ordering.cc',
    'partitioned_matrix_view.cc',
    'polynomial.cc',
    'preconditioner.cc',
    'preprocessor.cc',
    'problem.cc',
    'problem_impl.cc',
    'program.cc',
    'reorder_program.cc',
    'residual_block.cc',
    'residual_block_utils.cc',
    'schur_complement_solver.cc',
    'schur_eliminator.cc',
    'schur_jacobi_preconditioner.cc',
    'schur_templates.cc',
    'scratch_evaluate_preparer.cc',
    'single_linkage_clustering.cc',
    'solver.cc',
    'solver_utils.cc',
    'sparse_cholesky.cc',
    'sparse_matrix.cc',
    'sparse_normal_cholesky_solver.cc',
    'split.cc',
    'stringprintf.cc',
    'suitesparse.cc',
    'thread_token_provider.cc',
    'triplet_sparse_matrix.cc',
    'trust_region_minimizer.cc',
    'trust_region_preprocessor.cc',
    'trust_region_step_evaluator.cc',
    'trust_region_strategy.cc',
    'types.cc',
    'visibility_based_preconditioner.cc',
    'visibility.cc',
    'wall_time.cc',
    'generated/schur_eliminator_d_d_d.cc',
    'generated/schur_eliminator_2_2_2.cc',
    'generated/schur_eliminator_2_2_3.cc',
    'generated/schur_eliminator_2_2_4.cc',
    'generated/schur_eliminator_2_2_d.cc',
    'generated/schur_eliminator_2_3_3.cc',
    'generated/schur_eliminator_2_3_4.cc',
    'generated/schur_eliminator_2_3_6.cc',
    'generated/schur_eliminator_2_3_9.cc',
    'generated/schur_eliminator_2_3_d.cc',
    'generated/schur_eliminator_2_4_3.cc',
    'generated/schur_eliminator_2_4_4.cc',
    'generated/schur_eliminator_2_4_6.cc',
    'generated/schur_eliminator_2_4_8.cc',
    'generated/schur_eliminator_2_4_9.cc',
    'generated/schur_eliminator_2_4_d.cc',
    'generated/schur_eliminator_2_d_d.cc',
    'generated/schur_eliminator_4_4_2.cc',
    'generated/schur_eliminator_4_4_3.cc',
    'generated/schur_eliminator_4_4_4.cc',
    'generated/schur_eliminator_4_4_d.cc',
    'generated/partitioned_matrix_view_d_d_d.cc',
    'generated/partitioned_matrix_view_2_2_2.cc',
    'generated/partitioned_matrix_view_2_2_3.cc',
    'generated/partitioned_matrix_view_2_2_4.cc',
    'generated/partitioned_matrix_view_2_2_d.cc',
    'generated/partitioned_matrix_view_2_3_3.cc',
    'generated/partitioned_matrix_view_2_3_4.cc',
    'generated/partitioned_matrix_view_2_3_6.cc',
    'generated/partitioned_matrix_view_2_3_9.cc',
    'generated/partitioned_matrix_view_2_3_d.cc',
    'generated/partitioned_matrix_view_2_4_3.cc',
    'generated/partitioned_matrix_view_2_4_4.cc',
    'generated/partitioned_matrix_view_2_4_6.cc',
    'generated/partitioned_matrix_view_2_4_8.cc',
    'generated/partitioned_matrix_view_2_4_9.cc',
    'generated/partitioned_matrix_view_2_4_d.cc',
    'generated/partitioned_matrix_view_2_d_d.cc',
    'generated/partitioned_matrix_view_4_4_2.cc',
    'generated/partitioned_matrix_view_4_4_3.cc',
    'generated/partitioned_matrix_view_4_4_4.cc',
    'generated/partitioned_matrix_view_4_4_d.cc'
]]

# TODO(keir): This should get converted into a Skylark function that can
# configure Ceres into various permutations, like SuiteSparse or not, threading
# or not, glog or not, and so on.
# See https://github.com/ceres-solver/ceres-solver/issues/335.
cc_library(
    name = 'ceres',

    # These include directories and defines are propagated to other targets
    # depending on Ceres.
    includes = [
        'include',
        'config'
    ],

    # TODO(keir): These defines are placeholders for now to facilitate getting
    # started with a Bazel build. However, these should become configurable as
    # part of a Skylark Ceres target macro.
    defines = [
        'CERES_NO_SUITESPARSE',
        'CERES_NO_CXSPARSE',
        'CERES_NO_THREADS',
        'CERES_NO_LAPACK',
        'CERES_STD_UNORDERED_MAP',
    ],

    # These headers are made available to other targets.
    hdrs =
        glob(['include/ceres/*.h']) +
        glob(['include/ceres/internal/*.h']) +

        # This is an empty config, since the Bazel-based build does not
        # generate a config.h from config.h.in. This is fine, since Bazel
        # properly handles propagating -D defines to dependent targets.
        glob(['config/ceres/internal/config.h']),

    # Internal sources, options, and dependencies.
    srcs = CERES_SRCS +
        glob(['include/ceres/internal/*.h']) +
        glob(['internal/ceres/*.h']),

    copts = [
        '-Iinternal',
        '-Wno-sign-compare',
        ],

    deps = [
        '@com_github_eigen_eigen//:eigen',
        '@com_github_google_glog//:glog',
    ],
    visibility = ['//visibility:public'],
)

cc_library(
    name = 'test_util',
    defines = [
        'CERES_GFLAGS_NAMESPACE=gflags',
        ],

    copts = [
        '-Wno-sign-compare',
        '-DCERES_TEST_SRCDIR_SUFFIX=\\"data/\\"'
        ],

    includes = [
        'internal',
        'internal/ceres',
        ],

    hdrs = [
        'internal/ceres/gtest/gtest.h',
        'internal/ceres/gmock/gmock.h',
        'internal/ceres/gmock/mock-log.h',
        ],

    srcs = ['internal/ceres/' + x for x in [
        'evaluator_test_utils.cc',
        'numeric_diff_test_utils.cc',
        'test_util.cc',
        'gmock_gtest_all.cc',
        'gmock_main.cc',
        ]],

    deps = [
        '//:ceres',
        '@com_github_gflags_gflags//:gflags',
        ]
)

CERES_TESTS = [
  'array_utils',
  'autodiff_cost_function',
  'autodiff_local_parameterization',
  'autodiff',
  'block_jacobi_preconditioner',
  'block_random_access_dense_matrix',
  'block_random_access_diagonal_matrix',
  'block_random_access_sparse_matrix',
  'block_sparse_matrix',
  'bundle_adjustment',
  'canonical_views_clustering',
  'c_api',
  'compressed_col_sparse_matrix_utils',
  'compressed_row_sparse_matrix',
  'conditioned_cost_function',
  'conjugate_gradients_solver',
  'corrector',
  'cost_function_to_functor',
  'covariance',
  'cubic_interpolation',
  'dense_linear_solver',
  'dense_sparse_matrix',
  'detect_structure',
  'dogleg_strategy',
  'dynamic_autodiff_cost_function',
  'dynamic_compressed_row_sparse_matrix',
  'dynamic_numeric_diff_cost_function',
  'dynamic_sparse_normal_cholesky_solver',
  'dynamic_sparsity',
  'evaluator',
  'gradient_checker',
  'gradient_checking_cost_function',
  'gradient_problem_solver',
  'gradient_problem',
  'graph_algorithms',
  'graph',
  'householder_vector',
  'implicit_schur_complement',
  'inner_product_computer',
  'invert_psd_matrix',
  'is_close',
  'iterative_schur_complement_solver',
  'jet',
  'levenberg_marquardt_strategy',
  'line_search_minimizer',
  'line_search_preprocessor',
  'local_parameterization',
  'loss_function',
  'minimizer',
  'normal_prior',
  'numeric_diff_cost_function',
  'ordered_groups',
  'parameter_block_ordering',
  'parameter_block',
  'partitioned_matrix_view',
  'polynomial',
  'problem',
  'program',
  'reorder_program',
  'residual_block',
  'residual_block_utils',
  'rotation',
  'schur_complement_solver',
  'schur_eliminator',
  'single_linkage_clustering',
  'small_blas',
  'solver',
  'sparse_cholesky',
  'sparse_normal_cholesky_solver',
  'system',
  'tiny_solver_autodiff_function',
  'tiny_solver_cost_function_adapter',
  'tiny_solver',
  'triplet_sparse_matrix',
  'trust_region_minimizer',
  'trust_region_preprocessor',
  'visibility_based_preconditioner',
  'visibility',
  ]

# Instantiate all the tests with a template.
# TODO(keir): Use a Skylark macro to support tests having unique settings, like
# big or small without duplicating the common components (deps, etc).
# See https://github.com/ceres-solver/ceres-solver/issues/336.
[cc_test(
    name = test_name + '_test',
    srcs = ['internal/ceres/' + test_name + '_test.cc'],
    copts = [
        '-Wno-sign-compare',

        # These two warnings don't work well in conjunction with GMock, and
        # trigger incorrectly on parts of rotation_test. For now, disable them,
        # but in the future disable these warnings only for rotation_test.
        # TODO(keir): When the tests are macro-ified, apply these selectively.
        '-Wno-nonnull-compare',
        '-Wno-address',
        ],
    deps = [
        '//:test_util',
        '//:ceres',
        '@com_github_eigen_eigen//:eigen',
        '@com_github_gflags_gflags//:gflags',
        ],

    # Needed for bundle_adjustment_test.
    data = [':data/problem-16-22106-pre.txt'],
    timeout = 'long'
 )
 for test_name in CERES_TESTS]
