# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2023 Google Inc. All rights reserved.
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
load("//:bazel/ceres.bzl", "ceres_library")

ceres_library(
    name = "ceres",
    restrict_schur_specializations = False,
)

cc_library(
    name = "test_util",
    srcs = ["internal/ceres/" + x for x in [
        "evaluator_test_utils.cc",
        "numeric_diff_test_utils.cc",
        "test_util.cc",
    ]],
    copts = [
        "-Wno-sign-compare",
        "-DCERES_TEST_SRCDIR_SUFFIX=\\\"_main/data/\\\"",
    ],
    includes = [
        "internal",
        "internal/ceres",
    ],
    deps = [
        "//:ceres",
        "@googletest//:gtest_main",
    ],
)

CERES_TESTS = [
    "array_utils",
    "autodiff_cost_function",
    "autodiff_manifold",
    "autodiff",
    "block_jacobi_preconditioner",
    "block_random_access_dense_matrix",
    "block_random_access_diagonal_matrix",
    "block_random_access_sparse_matrix",
    "block_sparse_matrix",
    "canonical_views_clustering",
    "c_api",
    "compressed_col_sparse_matrix_utils",
    "compressed_row_sparse_matrix",
    "concurrent_queue",
    "conditioned_cost_function",
    "conjugate_gradients_solver",
    "corrector",
    "cost_function_to_functor",
    "covariance",
    "cubic_interpolation",
    "dense_cholesky",
    "dense_linear_solver",
    "dense_sparse_matrix",
    "detect_structure",
    "dogleg_strategy",
    "dynamic_autodiff_cost_function",
    "dynamic_compressed_row_sparse_matrix",
    "dynamic_numeric_diff_cost_function",
    "dynamic_sparse_normal_cholesky_solver",
    "dynamic_sparsity",
    "evaluation_callback",
    "evaluator",
    "gradient_checker",
    "gradient_checking_cost_function",
    "gradient_problem_solver",
    "gradient_problem",
    "graph_algorithms",
    "graph",
    "householder_vector",
    "implicit_schur_complement",
    "inner_product_computer",
    "invert_psd_matrix",
    "is_close",
    "iterative_refiner",
    "iterative_schur_complement_solver",
    "jet",
    "levenberg_marquardt_strategy",
    "line_search_minimizer",
    "line_search_preprocessor",
    "loss_function",
    "minimizer",
    "normal_prior",
    "numeric_diff_cost_function",
    "ordered_groups",
    "parallel_for",
    "parallel_utils",
    "parameter_block_ordering",
    "parameter_block",
    "partitioned_matrix_view",
    "polynomial",
    "problem",
    "program",
    "reorder_program",
    "residual_block",
    "residual_block_utils",
    "rotation",
    "schur_complement_solver",
    "schur_eliminator",
    "single_linkage_clustering",
    "small_blas",
    "solver",
    "sparse_cholesky",
    "sparse_normal_cholesky_solver",
    "subset_preconditioner",
    "system",
    "thread_pool",
    "tiny_solver_autodiff_function",
    "tiny_solver_cost_function_adapter",
    "tiny_solver",
    "triplet_sparse_matrix",
    "trust_region_minimizer",
    "trust_region_preprocessor",
    "visibility_based_preconditioner",
    "visibility",
]

TEST_DEPS = [
    "//:ceres",
    "//:test_util",
    "@eigen//:eigen",
]

# Instantiate all the tests with a template.
[cc_test(
    name = test_name + "_test",
    timeout = "short",
    srcs = ["internal/ceres/" + test_name + "_test.cc"],
    deps = TEST_DEPS,
) for test_name in CERES_TESTS]

# Instantiate all the bundle adjustment tests. These are separate to
# parallelize the execution of the tests; otherwise the tests take a long time.
#
# Note: While it is possible to run the Python script to generate the .cc files
# as part of the build, it introduces an undesirable build-time Python
# dependency that we'd prefer to avoid.
[cc_test(
    name = test_filename.split("/")[-1][:-3],  # Remove .cc.
    timeout = "long",
    srcs = [test_filename],
    # This is the data set that is bundled for the testing.
    data = [":data/problem-16-22106-pre.txt"],
    deps = TEST_DEPS,
) for test_filename in glob([
    "internal/ceres/generated_bundle_adjustment_tests/*_test.cc",
])]

# Build the benchmarks.
[cc_binary(
    name = benchmark_name,
    srcs = ["internal/ceres/" + benchmark_name + ".cc"],
    copts = ["-mllvm -inlinehint-threshold=1000000"],
    deps = TEST_DEPS + ["@google_benchmark//:benchmark"],

) for benchmark_name in [
    "evaluation_benchmark",
    "invert_psd_matrix_benchmark",
    "schur_eliminator_benchmark",
    "jet_operator_benchmark",
    "dense_linear_solver_benchmark",
    "parallel_vector_operations_benchmark",
    "small_blas_gemm_benchmark",
    "small_blas_gemv_benchmark",
]]
