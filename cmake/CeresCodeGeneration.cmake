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

set(CERES_CODEGEN_BUILD_DIR ${PROJECT_BINARY_DIR}/codegen)
set(CERES_CODEGEN_INCLUDE_DIR ${CERES_CODEGEN_BUILD_DIR}/include)

file(MAKE_DIRECTORY ${CERES_CODEGEN_BUILD_DIR})
file(MAKE_DIRECTORY ${CERES_CODEGEN_INCLUDE_DIR})

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
function(CERES_COST_FUNCTOR)
    # Define and parse arguments
    set(options)
    set(oneValueArgs NAME INPUT_FILE)
    set(multiValueArgs)
    cmake_parse_arguments(COST_FUNCTOR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # This function creates 3 targets
    #  1. The target which compiles the executable
    #  2. The target which executes it
    #  3. A combined target with the correct dependencies and libaries
    set(EXE_TARGET ${COST_FUNCTOR_NAME}_exe)
    set(GENERATE_TARGET ${COST_FUNCTOR_NAME}_generate)
    set(OUTPUT_TARGET ${COST_FUNCTOR_NAME})

    # Full path names of the files
    set(FULL_INPUT_FILE "${COST_FUNCTOR_INPUT_FILE}")
    set(FULL_OUTPUT_FILE "${CERES_CODEGEN_INCLUDE_DIR}/${COST_FUNCTOR_NAME}_codegen.h")
    set(STUB_FILE ${CERES_CODEGEN_BUILD_DIR}/stubs/${COST_FUNCTOR_NAME}_stub.cc)
    set(INCLUDE_FILE ${CERES_CODEGEN_BUILD_DIR}/include/${COST_FUNCTOR_NAME}_generated.h)

    # Generate a stub file which calls the autodiff code generation
    configure_file(${CMAKE_SOURCE_DIR}/cmake/codegen_functor.cc.in ${STUB_FILE})
    configure_file(${CMAKE_SOURCE_DIR}/cmake/codegen_include.h.in ${INCLUDE_FILE})

    # Build the executable
    add_executable(${EXE_TARGET} ${STUB_FILE})
    target_link_libraries(${EXE_TARGET} ceres)
    target_compile_definitions(${EXE_TARGET} PRIVATE -DCERES_CODEGEN)
    target_include_directories(${EXE_TARGET} PRIVATE ${CERES_CODEGEN_INCLUDE_DIR})

    # Setup a custom command that executes the stub
    add_custom_command(OUTPUT ${FULL_OUTPUT_FILE}
        COMMAND ${EXE_TARGET} ${FULL_OUTPUT_FILE}
        DEPENDS ${FULL_INPUT_FILE} ceres
        )
    # Wrap the custom command into a target
    add_custom_target(${GENERATE_TARGET} ALL DEPENDS ${FULL_OUTPUT_FILE})

    # Combine everything into an interface library
    add_library(${OUTPUT_TARGET} INTERFACE)
    target_include_directories(${OUTPUT_TARGET} INTERFACE ${CERES_CODEGEN_INCLUDE_DIR})
    add_dependencies(${OUTPUT_TARGET} ${GENERATE_TARGET})
endfunction()
