
if(CODE_GENERATION)
  macro (generate_benchmark_functor FUNCTOR_NAME FUNCTOR_FILE)
    ceres_generate_cost_function_implementation_for_functor(
      NAME ${FUNCTOR_NAME}
      INPUT_FILE ${FUNCTOR_FILE}
      OUTPUT_DIRECTORY benchmarks
      NAMESPACE benchmark
      )
  endmacro()
  include(CeresCodeGeneration)

  generate_benchmark_functor(SnavelyReprojectionError snavely_reprojection_error.h)
  generate_benchmark_functor(Linear1CostFunction trivial_cost_functions.h)
  generate_benchmark_functor(Linear10CostFunction trivial_cost_functions.h)

  list(APPEND CERES_BENCHMARK_FLAGS "-DWITH_CODE_GENERATION")
endif()

list(APPEND CERES_BENCHMARK_FLAGS "-mllvm" "-inline-threshold=1000000")

# All other flags + fast-math
list(APPEND CERES_BENCHMARK_FAST_MATH_FLAGS ${CERES_BENCHMARK_FLAGS} "-ffast-math")


add_executable(autodiff_ba_benchmark autodiff_ba_benchmark.cc)
add_dependencies_to_benchmark(autodiff_ba_benchmark)
target_compile_options(autodiff_ba_benchmark PRIVATE ${CERES_BENCHMARK_FLAGS})

add_executable(autodiff_ba_benchmark_fast_math autodiff_ba_benchmark.cc)
add_dependencies_to_benchmark(autodiff_ba_benchmark_fast_math)
target_compile_options(autodiff_ba_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})

add_executable(autodiff_trivial_benchmark autodiff_trivial_benchmark.cc)
add_dependencies_to_benchmark(autodiff_trivial_benchmark)
target_compile_options(autodiff_trivial_benchmark PRIVATE ${CERES_BENCHMARK_FLAGS})

add_executable(autodiff_trivial_benchmark_fast_math autodiff_trivial_benchmark.cc)
add_dependencies_to_benchmark(autodiff_trivial_benchmark_fast_math)
target_compile_options(autodiff_trivial_benchmark_fast_math PRIVATE ${CERES_BENCHMARK_FAST_MATH_FLAGS})

if(CODE_GENERATION)
  target_link_libraries(autodiff_ba_benchmark PUBLIC SnavelyReprojectionError)
  target_link_libraries(autodiff_ba_benchmark_fast_math PUBLIC SnavelyReprojectionError)
  target_link_libraries(autodiff_trivial_benchmark PUBLIC Linear1CostFunction Linear10CostFunction)
  target_link_libraries(autodiff_trivial_benchmark_fast_math PUBLIC Linear1CostFunction Linear10CostFunction)
endif()

