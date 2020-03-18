
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

  list(APPEND CERES_BENCHMARK_FLAGS "-DWITH_CODE_GENERATION")
endif()


# TODO: Add support for other compilers
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND CERES_BENCHMARK_FLAGS "-mllvm" "-inline-threshold=1000000")

  # All other flags + fast-math
  list(APPEND CERES_BENCHMARK_FAST_MATH_FLAGS ${CERES_BENCHMARK_FLAGS} "-ffast-math")
endif()

add_executable(rat43_benchmark rat43_benchmark.cc)
add_dependencies_to_benchmark(rat43_benchmark)


add_executable(snavely_reprojection_error_benchmark snavely_reprojection_error_benchmark.cc)
add_dependencies_to_benchmark(snavely_reprojection_error_benchmark)
target_compile_options(snavely_reprojection_error_benchmark PRIVATE ${CERES_BENCHMARK_FLAGS})

add_executable(snavely_reprojection_error_benchmark_fast_math snavely_reprojection_error_benchmark.cc)
add_dependencies_to_benchmark(snavely_reprojection_error_benchmark_fast_math)
target_compile_options(snavely_reprojection_error_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})

add_executable(linear_benchmark linear_benchmark.cc)
add_dependencies_to_benchmark(linear_benchmark)
target_compile_options(linear_benchmark PRIVATE ${CERES_BENCHMARK_FLAGS})

add_executable(linear_benchmark_fast_math linear_benchmark.cc)
add_dependencies_to_benchmark(linear_benchmark_fast_math)
target_compile_options(linear_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})

add_executable(brdf_benchmark brdf_benchmark.cc)
add_dependencies_to_benchmark(brdf_benchmark)
target_compile_options(brdf_benchmark PRIVATE ${CERES_BENCHMARK_FLAGS})

add_executable(brdf_benchmark_fast_math brdf_benchmark.cc)
add_dependencies_to_benchmark(brdf_benchmark_fast_math)
target_compile_options(brdf_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})

if(CODE_GENERATION)
  target_link_libraries(snavely_reprojection_error_benchmark PUBLIC SnavelyReprojectionError)
  target_link_libraries(snavely_reprojection_error_benchmark_fast_math PUBLIC SnavelyReprojectionError)
  target_link_libraries(linear_benchmark PUBLIC Linear1CostFunction Linear10CostFunction)
  target_link_libraries(linear_benchmark_fast_math PUBLIC Linear1CostFunction Linear10CostFunction)
  target_link_libraries(brdf_benchmark PUBLIC Brdf)
  target_link_libraries(brdf_benchmark_fast_math PUBLIC Brdf)
endif()

