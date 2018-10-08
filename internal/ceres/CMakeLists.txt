# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
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
# Author: keir@google.com (Keir Mierle)

# Avoid 'xxx.cc has no symbols' warnings from source files which are 'empty'
# when their enclosing #ifdefs are disabled.
if (CERES_THREADING_MODEL STREQUAL "CXX11_THREADS")
  set(CERES_PARALLEL_FOR_SRC parallel_for_cxx.cc thread_pool.cc)
elseif (CERES_THREADING_MODEL STREQUAL "OPENMP")
  set(CERES_PARALLEL_FOR_SRC parallel_for_openmp.cc)
  if (CMAKE_COMPILER_IS_GNUCXX)
    # OpenMP in GCC requires the GNU OpenMP library.
    list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES gomp)
  endif()
elseif (CERES_THREADING_MODEL STREQUAL "NO_THREADS")
  set(CERES_PARALLEL_FOR_SRC parallel_for_nothreads.cc)
endif()

set(CERES_INTERNAL_SRC
    ${CERES_PARALLEL_FOR_SRC}
    accelerate_sparse.cc
    array_utils.cc
    blas.cc
    block_evaluate_preparer.cc
    block_jacobi_preconditioner.cc
    block_jacobian_writer.cc
    block_random_access_dense_matrix.cc
    block_random_access_diagonal_matrix.cc
    block_random_access_matrix.cc
    block_random_access_sparse_matrix.cc
    block_sparse_matrix.cc
    block_structure.cc
    c_api.cc
    canonical_views_clustering.cc
    cgnr_solver.cc
    callbacks.cc
    compressed_col_sparse_matrix_utils.cc
    compressed_row_jacobian_writer.cc
    compressed_row_sparse_matrix.cc
    conditioned_cost_function.cc
    conjugate_gradients_solver.cc
    context.cc
    context_impl.cc
    coordinate_descent_minimizer.cc
    corrector.cc
    covariance.cc
    covariance_impl.cc
    cxsparse.cc
    dense_normal_cholesky_solver.cc
    dense_qr_solver.cc
    dense_sparse_matrix.cc
    detect_structure.cc
    dogleg_strategy.cc
    dynamic_compressed_row_jacobian_writer.cc
    dynamic_compressed_row_sparse_matrix.cc
    dynamic_sparse_normal_cholesky_solver.cc
    evaluator.cc
    eigensparse.cc
    file.cc
    float_suitesparse.cc
    float_cxsparse.cc
    function_sample.cc
    gradient_checker.cc
    gradient_checking_cost_function.cc
    gradient_problem.cc
    gradient_problem_solver.cc
    implicit_schur_complement.cc
    inner_product_computer.cc
    is_close.cc
    iterative_refiner.cc
    iterative_schur_complement_solver.cc
    levenberg_marquardt_strategy.cc
    lapack.cc
    line_search.cc
    line_search_direction.cc
    line_search_minimizer.cc
    line_search_preprocessor.cc
    linear_least_squares_problems.cc
    linear_operator.cc
    linear_solver.cc
    local_parameterization.cc
    loss_function.cc
    low_rank_inverse_hessian.cc
    minimizer.cc
    normal_prior.cc
    parallel_utils.cc
    parameter_block_ordering.cc
    partitioned_matrix_view.cc
    polynomial.cc
    preconditioner.cc
    preprocessor.cc
    problem.cc
    problem_impl.cc
    program.cc
    reorder_program.cc
    residual_block.cc
    residual_block_utils.cc
    schur_complement_solver.cc
    schur_eliminator.cc
    schur_jacobi_preconditioner.cc
    schur_templates.cc
    scratch_evaluate_preparer.cc
    single_linkage_clustering.cc
    solver.cc
    solver_utils.cc
    sparse_matrix.cc
    sparse_cholesky.cc
    sparse_normal_cholesky_solver.cc
    subset_preconditioner.cc
    split.cc
    stringprintf.cc
    suitesparse.cc
    thread_token_provider.cc
    triplet_sparse_matrix.cc
    trust_region_preprocessor.cc
    trust_region_minimizer.cc
    trust_region_step_evaluator.cc
    trust_region_strategy.cc
    types.cc
    visibility.cc
    visibility_based_preconditioner.cc
    wall_time.cc
)

# Also depend on the header files so that they appear in IDEs.
file(GLOB CERES_INTERNAL_HDRS *.h)
if (MINIGLOG)
  file(GLOB MINIGLOG_HDRS miniglog/glog/*.h)
  list(APPEND CERES_INTERNAL_HDRS ${MINIGLOG_HDRS})
  if (ANDROID)
    list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES log)
  endif()
endif()

# Depend also on public headers so they appear in IDEs.
file(GLOB CERES_PUBLIC_HDRS ${Ceres_SOURCE_DIR}/include/ceres/*.h)
file(GLOB CERES_PUBLIC_INTERNAL_HDRS ${Ceres_SOURCE_DIR}/include/ceres/internal/*.h)

# Include the specialized schur solvers.
if (SCHUR_SPECIALIZATIONS)
  file(GLOB CERES_INTERNAL_SCHUR_FILES generated/*.cc)
else (SCHUR_SPECIALIZATIONS)
  # Only the fully dynamic solver. The build is much faster this way.
  file(GLOB CERES_INTERNAL_SCHUR_FILES generated/*_d_d_d.cc)
endif (SCHUR_SPECIALIZATIONS)

# Build the list of dependencies for Ceres based on the current configuration.
find_package(Threads QUIET)
list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES Threads::Threads)

if (NOT MINIGLOG AND GLOG_FOUND)
  list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES ${GLOG_LIBRARIES})
  if (GFLAGS_FOUND)
    # If glog & gflags are both found, we assume that glog was built with
    # gflags, as it is awkward to perform a try_compile() to verify this
    # when gflags is an imported target (as it is in newer versions).
    # As glog #includes gflags/gflags.h in glog/logging.h if compiled with
    # gflags, it is thus a public dependency for Ceres in this case.
    list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES ${GFLAGS_LIBRARIES})
  endif()
endif (NOT MINIGLOG AND GLOG_FOUND)

if (SUITESPARSE AND SUITESPARSE_FOUND)
  # Define version information for use in Solver::FullReport.
  add_definitions(-DCERES_SUITESPARSE_VERSION="${SUITESPARSE_VERSION}")
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES ${SUITESPARSE_LIBRARIES})
endif (SUITESPARSE AND SUITESPARSE_FOUND)

if (CXSPARSE AND CXSPARSE_FOUND)
  # Define version information for use in Solver::FullReport.
  add_definitions(-DCERES_CXSPARSE_VERSION="${CXSPARSE_VERSION}")
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES ${CXSPARSE_LIBRARIES})
endif (CXSPARSE AND CXSPARSE_FOUND)

if (ACCELERATESPARSE AND AccelerateSparse_FOUND)
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES ${AccelerateSparse_LIBRARIES})
endif()

if (LAPACK_FOUND)
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES ${LAPACK_LIBRARIES})
endif ()

set(CERES_LIBRARY_SOURCE
    ${CERES_INTERNAL_SRC}
    ${CERES_INTERNAL_HDRS}
    ${CERES_PUBLIC_HDRS}
    ${CERES_PUBLIC_INTERNAL_HDRS}
    ${CERES_INTERNAL_SCHUR_FILES})

# Primarily for Android, but optionally for others, compile the minimal
# glog implementation into Ceres.
if (MINIGLOG)
  list(APPEND CERES_LIBRARY_SOURCE miniglog/glog/logging.cc)
endif (MINIGLOG)

add_library(ceres ${CERES_LIBRARY_SOURCE})
set_target_properties(ceres PROPERTIES
  VERSION ${CERES_VERSION}
  SOVERSION ${CERES_VERSION_MAJOR})

# The ability to specify a minimum language version via cxx_std_[11,14,17]
# requires CMake >= 3.8.  Prior to that we have to specify the compiler features
# we require.
if (CMAKE_VERSION VERSION_LESS 3.8)
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_alignas cxx_alignof cxx_constexpr)
else()
  # Forward whatever C++ version Ceres was compiled with as our requirement
  # for downstream clients.
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_std_${CMAKE_CXX_STANDARD})
endif()
target_compile_features(ceres PUBLIC ${REQUIRED_PUBLIC_CXX_FEATURES})

include(AppendTargetProperty)
# Always build position-independent code (PIC), even when building Ceres as a
# static library so that shared libraries can link against it, not just
# executables (PIC does not apply on Windows).
if (NOT WIN32 AND NOT BUILD_SHARED_LIBS)
  # Use set_target_properties() not append_target_property() here as
  # POSITION_INDEPENDENT_CODE is a binary ON/OFF switch.
  set_target_properties(ceres PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_SHARED_LIBS)
  # When building a shared library, mark all external libraries as
  # PRIVATE so they don't show up as a dependency.
  target_link_libraries(ceres
        LINK_PUBLIC ${CERES_LIBRARY_PUBLIC_DEPENDENCIES}
        LINK_PRIVATE ${CERES_LIBRARY_PRIVATE_DEPENDENCIES})
else (BUILD_SHARED_LIBS)
  # When building a static library, all external libraries are
  # PUBLIC(default) since the user needs to link to them.
  # They will be listed in CeresTargets.cmake.
  set(CERES_LIBRARY_DEPENDENCIES
        ${CERES_LIBRARY_PUBLIC_DEPENDENCIES}
        ${CERES_LIBRARY_PRIVATE_DEPENDENCIES})
  target_link_libraries(ceres ${CERES_LIBRARY_DEPENDENCIES})
endif (BUILD_SHARED_LIBS)

# Add the Ceres headers to its target.
#
# Force the location containing the configured config.h to the front of the
# include_directories list (by default it is appended to the back) to ensure
# that if the user has an installed version of Ceres in the same location as one
# of the dependencies (e.g. /usr/local) that we find the config.h we just
# configured, not the (older) installed config.h.
target_include_directories(ceres BEFORE PUBLIC
  $<BUILD_INTERFACE:${Ceres_BINARY_DIR}/config>)
target_include_directories(ceres PRIVATE ${Ceres_SOURCE_DIR}/internal)
target_include_directories(ceres PUBLIC
  $<BUILD_INTERFACE:${Ceres_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Eigen SparseQR generates various compiler warnings related to unused and
# uninitialised local variables.  To avoid having to individually suppress these
# warnings around the #include statments for Eigen headers across all GCC/Clang
# versions, we tell CMake to treat Eigen headers as system headers.  This
# results in all compiler warnings from them being suppressed.
target_include_directories(ceres SYSTEM PUBLIC ${EIGEN_INCLUDE_DIRS})

# Gather the list of public & private include locations for all enabled optional
# dependencies to be added to the Ceres target.
set(CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS "")
set(CERES_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS "")
if (MINIGLOG)
  # Force the miniglog headers to the front of the public include directories
  # to protect against the case when the user has glog installed in a standard
  # location (specifically the same as the Ceres install location) but compiled
  # Ceres with MINIGLOG anyway.  Otherwise: "glog/logging.h" in the public Ceres
  # headers used in client code would match the installed version of glog, not
  # the miniglog headers, and the client application would fail to link.
  #
  # Note that this is an imperfect fix, as we cannot control the include
  # directories in client projects, and they could easily invert this ordering
  # themselves (intentionally or otherwise) and so break their build.
  target_include_directories(ceres BEFORE PUBLIC
    $<BUILD_INTERFACE:${Ceres_SOURCE_DIR}/internal/ceres/miniglog>
    $<INSTALL_INTERFACE:include/ceres/internal/miniglog>)
elseif (NOT FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION)
  # Only append glog include directories if the glog found was not a CMake
  # exported target that already includes them.
  list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS
    ${GLOG_INCLUDE_DIRS})
endif()
if (SUITESPARSE)
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS
    ${SUITESPARSE_INCLUDE_DIRS})
endif()
if (CXSPARSE)
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS
    ${CXSPARSE_INCLUDE_DIRS})
endif()
if (ACCELERATESPARSE)
  list(APPEND CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS
    ${AccelerateSparse_INCLUDE_DIRS})
endif()
if (GFLAGS AND NOT FOUND_INSTALLED_GFLAGS_CMAKE_CONFIGURATION)
  # Only append gflags include directories if the gflags found was not a CMake
  # exported target that already includes them.
  list(APPEND CERES_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS
    ${GFLAGS_INCLUDE_DIRS})
endif()
# Add include locations for optional dependencies to the Ceres target without
# duplication.
list(REMOVE_DUPLICATES CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(ceres PRIVATE ${INC_DIR})
endforeach()
list(REMOVE_DUPLICATES CERES_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${CERES_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(ceres PUBLIC ${INC_DIR})
endforeach()

install(TARGETS ceres
        EXPORT  CeresExport
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib${LIB_SUFFIX}
        ARCHIVE DESTINATION lib${LIB_SUFFIX})

if (BUILD_TESTING AND GFLAGS)
  add_library(gtest gmock_gtest_all.cc gmock_main.cc)
  target_include_directories(gtest PUBLIC ${Ceres_SOURCE_DIR}/internal/ceres)
  if (BUILD_SHARED_LIBS)
    # Define gtest-specific shared library flags for compilation.
    append_target_property(gtest COMPILE_DEFINITIONS
      GTEST_CREATE_SHARED_LIBRARY)
  endif()

  add_library(test_util
              evaluator_test_utils.cc
              numeric_diff_test_utils.cc
              test_util.cc)
  target_include_directories(test_util PUBLIC ${Ceres_SOURCE_DIR}/internal)

  if (MINIGLOG)
    # When using miniglog, it is compiled into Ceres, thus Ceres becomes
    # the library against which other libraries should link for logging.
    target_link_libraries(gtest ${GFLAGS_LIBRARIES} ceres)
    target_link_libraries(test_util ceres gtest)
  else (MINIGLOG)
    target_link_libraries(gtest ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
    target_link_libraries(test_util ceres gtest ${GLOG_LIBRARIES})
  endif (MINIGLOG)

  macro (CERES_TEST NAME)
    add_executable(${NAME}_test ${NAME}_test.cc)
    # Pull in local headers from the generated test directories when ceres_test()
    # is invoked there, as well as the private headers in this directory which
    # may be referenced without the 'ceres' path prefix and all private
    # dependencies that may be directly referenced.
    target_include_directories(${NAME}_test
      PUBLIC ${CMAKE_CURRENT_LIST_DIR}
             ${Ceres_SOURCE_DIR}/internal/ceres
             ${CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})

    target_link_libraries(${NAME}_test test_util ceres gtest)
    if (BUILD_SHARED_LIBS)
      # Define gtest-specific shared library flags for linking.
      append_target_property(${NAME}_test COMPILE_DEFINITIONS
        GTEST_LINKED_AS_SHARED_LIBRARY)
    endif()
    add_test(NAME ${NAME}_test
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test
             --test_srcdir
             ${Ceres_SOURCE_DIR}/data)
  endmacro (CERES_TEST)

  ceres_test(array_utils)
  ceres_test(autodiff)
  ceres_test(autodiff_cost_function)
  ceres_test(autodiff_local_parameterization)
  ceres_test(block_jacobi_preconditioner)
  ceres_test(block_random_access_dense_matrix)
  ceres_test(block_random_access_diagonal_matrix)
  ceres_test(block_random_access_sparse_matrix)
  ceres_test(block_sparse_matrix)
  ceres_test(c_api)
  ceres_test(canonical_views_clustering)
  ceres_test(compressed_col_sparse_matrix_utils)
  ceres_test(compressed_row_sparse_matrix)
  ceres_test(concurrent_queue)
  ceres_test(conditioned_cost_function)
  ceres_test(conjugate_gradients_solver)
  ceres_test(corrector)
  ceres_test(cost_function_to_functor)
  ceres_test(covariance)
  ceres_test(cubic_interpolation)
  ceres_test(dense_linear_solver)
  ceres_test(dense_sparse_matrix)
  ceres_test(detect_structure)
  ceres_test(dogleg_strategy)
  ceres_test(dynamic_autodiff_cost_function)
  ceres_test(dynamic_compressed_row_sparse_matrix)
  ceres_test(dynamic_numeric_diff_cost_function)
  ceres_test(dynamic_sparse_normal_cholesky_solver)
  ceres_test(dynamic_sparsity)
  ceres_test(evaluation_callback)
  ceres_test(evaluator)
  ceres_test(gradient_checker)
  ceres_test(gradient_checking_cost_function)
  ceres_test(gradient_problem)
  ceres_test(gradient_problem_solver)
  ceres_test(graph)
  ceres_test(graph_algorithms)
  ceres_test(householder_vector)
  ceres_test(implicit_schur_complement)
  ceres_test(inner_product_computer)
  ceres_test(invert_psd_matrix)
  ceres_test(integer_sequence)
  ceres_test(integer_sequence_algorithm)
  ceres_test(is_close)
  ceres_test(iterative_refiner)
  ceres_test(iterative_schur_complement_solver)
  ceres_test(jet)
  ceres_test(levenberg_marquardt_strategy)
  ceres_test(line_search_minimizer)
  ceres_test(line_search_preprocessor)
  ceres_test(local_parameterization)
  ceres_test(loss_function)
  ceres_test(minimizer)
  ceres_test(normal_prior)
  ceres_test(numeric_diff_cost_function)
  ceres_test(ordered_groups)
  ceres_test(parallel_for)
  ceres_test(parallel_utils)
  ceres_test(parameter_block)
  ceres_test(parameter_block_ordering)
  ceres_test(parameter_dims)
  ceres_test(partitioned_matrix_view)
  ceres_test(polynomial)
  ceres_test(problem)
  ceres_test(program)
  ceres_test(reorder_program)
  ceres_test(residual_block)
  ceres_test(residual_block_utils)
  ceres_test(rotation)
  ceres_test(schur_complement_solver)
  ceres_test(schur_eliminator)
  ceres_test(single_linkage_clustering)
  ceres_test(small_blas)
  ceres_test(solver)
  ceres_test(sparse_cholesky)
  ceres_test(sparse_normal_cholesky_solver)
  ceres_test(subset_preconditioner)
  ceres_test(system)
  ceres_test(tiny_solver)
  ceres_test(tiny_solver_autodiff_function)
  ceres_test(tiny_solver_cost_function_adapter)
  ceres_test(thread_pool)
  ceres_test(triplet_sparse_matrix)
  ceres_test(trust_region_minimizer)
  ceres_test(trust_region_preprocessor)
  ceres_test(visibility)
  ceres_test(visibility_based_preconditioner)

  add_subdirectory(generated_bundle_adjustment_tests)

endif (BUILD_TESTING AND GFLAGS)

macro(add_dependencies_to_benchmark BENCHMARK_TARGET)
  target_link_libraries(${BENCHMARK_TARGET} ceres benchmark::benchmark)
  target_include_directories(${BENCHMARK_TARGET} PUBLIC
                             ${Ceres_SOURCE_DIR}/internal
                             ${CERES_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})
endmacro()

if (BUILD_BENCHMARKS)
  add_executable(autodiff_cost_function_benchmark autodiff_cost_function_benchmark.cc)
  add_dependencies_to_benchmark(autodiff_cost_function_benchmark)

  add_executable(small_blas_gemv_benchmark small_blas_gemv_benchmark.cc)
  add_dependencies_to_benchmark(small_blas_gemv_benchmark)

  add_executable(small_blas_gemm_benchmark small_blas_gemm_benchmark.cc)
  add_dependencies_to_benchmark(small_blas_gemm_benchmark)
endif (BUILD_BENCHMARKS)
