# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2019 Google Inc. All rights reserved.
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
# Author: darius.rueckert@fau.de (Darius Rueckert)

# The directory containing the following files
#   - codegen_include.h.in
#   - generate_code_from_functor.cc.in
set(CODEGEN_CMAKE_SCRIPT_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Generates C-code implementation of Ceres' CostFunction::Evaluate() API from a
# templated C++ cost functor derived from ceres::CodegenCostFunction using
# autodiff. The resulting implementation replaces the direct instantiation of
# autodiff in client code, typically resulting in improved performance.
#
# Parameters:
#
# NAME
#     The name of the cost functor. The name must exactly match the C++ type
#     name of the functor. This is also the name of the CMake output target.
# NAMESPACE [optional]
#     The C++ namespace of the cost functor type. For example, if the full
#     type name is ceres::BundleAdjust, then NAME should be "BundleAdjust"
#     and NAMESPACE should be "ceres".
# INPUT_FILE
#     The path to the header defining the cost functor <NAME>.
# OUTPUT_DIRECTORY [default = "generated"]
#     The relative output directory of the generated header file. This is the
#     prefix that has to be added to the #include of the generated files, i.e:
#     #include "<OUTPUT_DIRECTORY>/<GENERATED_FILE.h>"

#
# Example Usage:
#   ceres_generate_cost_function_implementation_for_functor(
#       NAME SquareFunctor
#       INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/square_functor.h
#       )
#   add_executable(helloworld_codegen helloworld_codegen.cc )
#   target_link_libraries(helloworld_codegen ceres SquareFunctor)
function(ceres_generate_cost_function_implementation_for_functor)
  # Define and parse arguments
  set(OPTIONAL_ARGS)
  set(ONE_VALUE_ARGS NAME INPUT_FILE OUTPUT_DIRECTORY NAMESPACE)
  set(MULTI_VALUE_ARGS)
  cmake_parse_arguments(
    COST_FUNCTOR "${OPTIONAL_ARGS}" "${ONE_VALUE_ARGS}"
    "${MULTI_VALUE_ARGS}" ${ARGN} )

  # Default value of the output directory
  set(OUTPUT_DIRECTORY "generated")
  if(COST_FUNCTOR_OUTPUT_DIRECTORY)
    set(OUTPUT_DIRECTORY "${COST_FUNCTOR_OUTPUT_DIRECTORY}")
  endif()

  set(CALLER_CODEGEN_BUILD_DIR "${PROJECT_BINARY_DIR}/codegen")
  set(CALLER_CODEGEN_INCLUDE_DIR "${CALLER_CODEGEN_BUILD_DIR}/include/")

  file(MAKE_DIRECTORY
    "${CALLER_CODEGEN_BUILD_DIR}")
  file(MAKE_DIRECTORY
    "${CALLER_CODEGEN_INCLUDE_DIR}/${OUTPUT_DIRECTORY}")
  file(MAKE_DIRECTORY
    "${CALLER_CODEGEN_BUILD_DIR}/src")

  # Convert the input file to an absolute path and check if it exists
  get_filename_component(
    COST_FUNCTOR_INPUT_FILE "${COST_FUNCTOR_INPUT_FILE}" REALPATH)
  if(NOT EXISTS "${COST_FUNCTOR_INPUT_FILE}")
    message(FATAL_ERROR
      "Could not find codegen input file ${COST_FUNCTOR_INPUT_FILE}")
  endif()


  # The full C++ type name of the cost functor. This is used inside the
  # generator to create an object of it.
  set(FULL_CXX_FUNCTOR_TYPE_NAME "${COST_FUNCTOR_NAME}")
  if(COST_FUNCTOR_NAMESPACE)
    set(FULL_CXX_FUNCTOR_TYPE_NAME
      "${COST_FUNCTOR_NAMESPACE}::${FULL_CXX_FUNCTOR_TYPE_NAME}")
  endif()

  # 1. Generate a wrapper include file which is included by the user.
  #    This is required, because
  #      - It must exist during compiliation of the code generator (otherwise
  #        the #include will fail)
  #      - We don't want to have the users add macros to their code
  string(TOLOWER "${COST_FUNCTOR_NAME}" LOWER_CASE_FUNCTOR_NAME)
  set(INCLUDE_FILE
    "${CALLER_CODEGEN_INCLUDE_DIR}/${OUTPUT_DIRECTORY}/${LOWER_CASE_FUNCTOR_NAME}.h")
  configure_file("${CODEGEN_CMAKE_SCRIPT_DIR}/codegen_include.inc.in" "${INCLUDE_FILE}")

  # 2. Generate the source file for the code generator
  set(GENERATOR_SOURCE
    "${CALLER_CODEGEN_BUILD_DIR}/src/${LOWER_CASE_FUNCTOR_NAME}_code_generator.cc")
  set(GENERATED_EVALUATION_IMPL_FILE
    "${CALLER_CODEGEN_INCLUDE_DIR}/${OUTPUT_DIRECTORY}/${LOWER_CASE_FUNCTOR_NAME}_evaluate_impl.inc")
  configure_file(
    "${CODEGEN_CMAKE_SCRIPT_DIR}/generate_code_for_functor.cc.in" "${GENERATOR_SOURCE}")

  # 3. Build the executable that generates the autodiff code
  set(GENERATOR_TARGET ${COST_FUNCTOR_NAME}_generator)
  add_executable(${GENERATOR_TARGET} "${GENERATOR_SOURCE}")
  target_link_libraries(${GENERATOR_TARGET} ceres)
  set_target_properties(${GENERATOR_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CALLER_CODEGEN_BUILD_DIR}/bin")
  target_compile_definitions(${GENERATOR_TARGET} PRIVATE -DCERES_CODEGEN)
  target_include_directories(${GENERATOR_TARGET}
    PRIVATE "${CALLER_CODEGEN_INCLUDE_DIR}")

  # 4. Execute the program from (3.) using a custom command
  add_custom_command(OUTPUT "${GENERATED_EVALUATION_IMPL_FILE}"
    COMMAND ${GENERATOR_TARGET}
    DEPENDS "${COST_FUNCTOR_INPUT_FILE}"
    VERBATIM
    )
  set(GENERATE_TARGET ${COST_FUNCTOR_NAME}_generate)
  add_custom_target(${GENERATE_TARGET} DEPENDS "${GENERATED_EVALUATION_IMPL_FILE}" VERBATIM)

  # 5. Create an output target which can be used by the client. This is required,
  #    because custom targets can't have include directories.
  set(OUTPUT_TARGET ${COST_FUNCTOR_NAME})
  add_library(${OUTPUT_TARGET} INTERFACE)
  target_include_directories(
    ${OUTPUT_TARGET} INTERFACE "${CALLER_CODEGEN_INCLUDE_DIR}")
  target_sources(
    ${OUTPUT_TARGET} INTERFACE "${INCLUDE_FILE}" "${GENERATED_EVALUATION_IMPL_FILE}")
  add_dependencies(${OUTPUT_TARGET} ${GENERATE_TARGET})
endfunction()
