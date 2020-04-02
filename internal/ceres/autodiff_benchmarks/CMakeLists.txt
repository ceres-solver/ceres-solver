# TODO: Add support for other compilers
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CERES_BENCHMARK_FLAGS "-mllvm" "-inline-threshold=1000000")
endif()

add_executable(autodiff_benchmarks autodiff_benchmarks.cc)
add_dependencies_to_benchmark(autodiff_benchmarks)
target_compile_options(autodiff_benchmarks PRIVATE ${CERES_BENCHMARK_FLAGS})

# All other flags + fast-math
list(APPEND CERES_BENCHMARK_FAST_MATH_FLAGS ${CERES_BENCHMARK_FLAGS} "-ffast-math")
add_executable(autodiff_benchmarks_fast_math autodiff_benchmarks.cc)
add_dependencies_to_benchmark(autodiff_benchmarks_fast_math)
target_compile_options(autodiff_benchmarks_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})
