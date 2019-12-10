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

set(CALLER_CODEGEN_BUILD_DIR ${PROJECT_BINARY_DIR}/codegen)
set(CALLER_CODEGEN_INCLUDE_DIR ${CALLER_CODEGEN_BUILD_DIR}/include/)

file(MAKE_DIRECTORY ${CALLER_CODEGEN_BUILD_DIR})
file(MAKE_DIRECTORY ${CALLER_CODEGEN_INCLUDE_DIR}/codegen)

# The directory containing the following files
#   - codegen_include.h.in
#   - generate_code_from_functor.cc.in
set(CODEGEN_CMAKE_SCRIPT_DIR ${CMAKE_CURRENT_LIST_DIR})

# Generates autodiff C-code for a given C++ cost functor. Parameters:
#
# NAME
#     The name of the cost functor. The name must exactly match the C++ type name of the
#     functor. This is also the name of the CMake output target.
# INPUT_FILE
#     The absolute file name of the header defining the cost functor <NAME>.
#
#
# Example Usage:
#   CERES_COST_FUNCTOR(
#       NAME SquareFunctor
#       INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR}/square_functor.h
#       )
#   add_executable(helloworld_codegen helloworld_codegen.cc )
#   target_link_libraries(helloworld_codegen ceres SquareFunctor)
function(ceres_cost_functor)
    # Define and parse arguments
    set(OPTIONAL_ARGS)
    set(ONE_VALUE_ARGS NAME INPUT_FILE)
    set(MULTI_VALUE_ARGS)
    cmake_parse_arguments(COST_FUNCTOR "${OPTIONAL_ARGS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN} )


    # 1. Generate a wrapper include file which is included by the application. This is required, because
    #      - It must exist during compiliation of the code generator (otherwise the #include will fail)
    #      - We don't want to have the users add macros to their code
    set(INCLUDE_FILE "${CALLER_CODEGEN_INCLUDE_DIR}/codegen/${COST_FUNCTOR_NAME}.h")
    configure_file("${CODEGEN_CMAKE_SCRIPT_DIR}/codegen_include.h.in" "${INCLUDE_FILE}")

    # 2. Generate the source file for the code generator
    set(GENERATOR_SOURCE "${CALLER_CODEGEN_BUILD_DIR}/src/${COST_FUNCTOR_NAME}.cc")
    set(FULL_OUTPUT_FILE "${CALLER_CODEGEN_INCLUDE_DIR}/codegen/${COST_FUNCTOR_NAME}_evaluate_impl.h")
    configure_file("${CODEGEN_CMAKE_SCRIPT_DIR}/generate_code_from_functor.cc.in" "${GENERATOR_SOURCE}")


    # 3. Build the executable that generates the autodiff code
    set(GENERATOR_TARGET ${COST_FUNCTOR_NAME}_generator)
    add_executable(${GENERATOR_TARGET} "${GENERATOR_SOURCE}")
    target_link_libraries(${GENERATOR_TARGET} ceres)
    set_target_properties(${GENERATOR_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CALLER_CODEGEN_BUILD_DIR}/bin")
    target_compile_definitions(${GENERATOR_TARGET} PRIVATE -DCERES_CODEGEN)
    target_include_directories(${GENERATOR_TARGET} PRIVATE "${CALLER_CODEGEN_INCLUDE_DIR}")


    # 4. Execute the program from (2.) using a custom command
    add_custom_command(OUTPUT "${FULL_OUTPUT_FILE}"
        COMMAND ${GENERATOR_TARGET} "${FULL_OUTPUT_FILE}"
        DEPENDS "${COST_FUNCTOR_INPUT_FILE}" ceres
        VERBATIM
        )
    set(GENERATE_TARGET ${COST_FUNCTOR_NAME}_generate)
    add_custom_target(${GENERATE_TARGET} ALL DEPENDS "${FULL_OUTPUT_FILE}" ceres VERBATIM)

    # 5. Create an output target which can be used by the application. This is required,
    #    because custom targets can't have include directories.
    set(OUTPUT_TARGET ${COST_FUNCTOR_NAME})
    add_library(${OUTPUT_TARGET} INTERFACE)
    target_include_directories(${OUTPUT_TARGET} INTERFACE "${CALLER_CODEGEN_INCLUDE_DIR}")
    target_sources(${OUTPUT_TARGET} INTERFACE "${INCLUDE_FILE}" "${FULL_OUTPUT_FILE}" )
    add_dependencies(${OUTPUT_TARGET} ${GENERATE_TARGET})
endfunction()
