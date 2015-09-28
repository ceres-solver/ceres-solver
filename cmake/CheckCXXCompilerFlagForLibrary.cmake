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
# Author: alexs.mac@gmail.com (Alex Stewart)

# NOTE: It is very important that we save this variable OUTSIDE of
#       the definition of check_cxx_compiler_flag_for_library() as the
#       value of CMAKE_CURRENT_LIST_DIR is different.  Outside it is the
#       path to the directory containing *this* file.  Inside the function
#       it would be the path to the directory containing the CMakeLists.txt
#       that called check_cxx_compiler_flag_for_library().
set(CHECK_CXX_COMPILER_FLAG_FOR_LIBRARY_PROJECT_DIR
  "${CMAKE_CURRENT_LIST_DIR}/CheckCXXCompilerFlagForLibrary")

# Custom version of check_cxx_compiler_flag shipped with CMake that
# verifies that a sample library, not executable can be built using
# the specified compiler flag: ${FLAG}.  Sets ${VAR} to TRUE iff the
# compilation of the test library with ${FLAG} was successful.
function(check_cxx_compiler_flag_for_library FLAG VAR)
  if (NOT IS_DIRECTORY "${CHECK_CXX_COMPILER_FLAG_FOR_LIBRARY_PROJECT_DIR}")
    message(FATAL_ERROR "Missing CheckCXXCompilerFlagForLibrary test project, "
      "it is not here: ${CHECK_CXX_COMPILER_FLAG_FOR_LIBRARY_PROJECT_DIR}")
  endif()

  # Ensure that the test output directory is always unique.
  string(RANDOM RANDOM_TEXT)
  set(OUTPUT_DIR
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckCXXCompilerFlagForLibrary-${RANDOM_TEXT}")

  # Try to compile the test library project with the specified compiler
  # flag, using the same shared/static library configuration as the caller.
  try_compile(${VAR} ${OUTPUT_DIR}
    ${CHECK_CXX_COMPILER_FLAG_FOR_LIBRARY_PROJECT_DIR}
    CheckCXXCompilerFlagForLibrary
    CMAKE_FLAGS -DTEST_CXX_FLAGS:STRING=${FLAG}
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    OUTPUT_VARIABLE OUTPUT)

  # Taken from CheckCXXSourceCompiles.cmake shipped with CMake to ensure
  # that we report the same format output as the built-in functions and
  # also write our result to the CMake log/error file.
  if(${VAR})
    set(${VAR} 1 CACHE INTERNAL "Test ${VAR}")
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Performing Test ${VAR} - Success")
    endif()
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Performing CheckCXXCompilerFlagForLibrary Test ${VAR} succeded with the following output:\n"
      "${OUTPUT}\n"
      "Test flag was: ${FLAG}\n")
  else()
    if(NOT CMAKE_REQUIRED_QUIET)
      message(STATUS "Performing Test ${VAR} - Failed")
    endif()
    set(${VAR} "" CACHE INTERNAL "Test ${VAR}")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Performing CheckCXXCompilerFlagForLibrary Test ${VAR} failed with the following output:\n"
      "${OUTPUT}\n"
      "Test flag was: ${FLAG}\n")
  endif()

  set(${VAR} "${${VAR}}" PARENT_SCOPE)
endfunction()
