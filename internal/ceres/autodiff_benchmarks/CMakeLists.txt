# TODO: Add support for other compilers
if(CODE_GENERATION)
  list(APPEND CERES_BENCHMARK_FLAGS "-DWITH_CODE_GENERATION")
endif()

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

if(CODE_GENERATION)
  macro (generate_benchmark_functor FUNCTOR_NAME FUNCTOR_FILE)
    ceres_generate_cost_function_implementation_for_functor(
      NAME ${FUNCTOR_NAME}
      INPUT_FILE ${FUNCTOR_FILE}
      OUTPUT_DIRECTORY benchmarks
      NAMESPACE ceres
      )
  endmacro()
  include(CeresCodeGeneration)

  generate_benchmark_functor(SnavelyReprojectionError snavely_reprojection_error.h)
  generate_benchmark_functor(Linear1CostFunction linear_cost_functions.h)
  generate_benchmark_functor(Linear10CostFunction linear_cost_functions.h)
  generate_benchmark_functor(Brdf brdf_cost_function.h)

  target_link_libraries(autodiff_benchmarks PUBLIC SnavelyReprojectionError Linear1CostFunction Linear10CostFunction Brdf)
  target_link_libraries(autodiff_benchmarks_fast_math PUBLIC SnavelyReprojectionError Linear1CostFunction Linear10CostFunction Brdf)
endif()
