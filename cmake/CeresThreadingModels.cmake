# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2018 Google Inc. All rights reserved.
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

# Ordered by expected preference.
set(CERES_THREADING_MODELS "CXX_THREADS;OPENMP;NO_THREADS")

function(find_available_ceres_threading_models CERES_THREADING_MODELS_AVAILABLE_VAR)
  set(CERES_THREADING_MODELS_AVAILABLE ${CERES_THREADING_MODELS})
  # Remove any threading models for which the dependencies are not available.
  find_package(OpenMP QUIET)
  if (NOT OPENMP_FOUND)
    list(REMOVE_ITEM CERES_THREADING_MODELS_AVAILABLE "OPENMP")
  endif()
  if (NOT CERES_THREADING_MODELS_AVAILABLE)
    # At least NO_THREADS should never be removed.  This check is purely
    # protective against future threading model updates.
    message(FATAL_ERROR "Ceres bug: Removed all threading models.")
  endif()
  set(${CERES_THREADING_MODELS_AVAILABLE_VAR}
    ${CERES_THREADING_MODELS_AVAILABLE} PARENT_SCOPE)
endfunction()

macro(set_ceres_threading_model_to_cxx11_threads)
  list(APPEND CERES_COMPILE_OPTIONS CERES_USE_CXX_THREADS)
endmacro()

macro(set_ceres_threading_model_to_openmp)
  find_package(OpenMP REQUIRED)
  list(APPEND CERES_COMPILE_OPTIONS CERES_USE_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endmacro()

macro(set_ceres_threading_model_to_no_threads)
  list(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
endmacro()

macro(set_ceres_threading_model CERES_THREADING_MODEL_TO_SET)
  if ("${CERES_THREADING_MODEL_TO_SET}" STREQUAL "CXX_THREADS")
    set_ceres_threading_model_to_cxx11_threads()
  elseif ("${CERES_THREADING_MODEL_TO_SET}" STREQUAL "OPENMP")
    set_ceres_threading_model_to_openmp()
  elseif ("${CERES_THREADING_MODEL_TO_SET}" STREQUAL "NO_THREADS")
    set_ceres_threading_model_to_no_threads()
  else()
    include(PrettyPrintCMakeList)
    find_available_ceres_threading_models(_AVAILABLE_THREADING_MODELS)
    pretty_print_cmake_list(
      _AVAILABLE_THREADING_MODELS ${_AVAILABLE_THREADING_MODELS})
    message(FATAL_ERROR "Unknown threading model specified: "
      "'${CERES_THREADING_MODEL_TO_SET}'. Available threading models for "
      "this platform are: ${_AVAILABLE_THREADING_MODELS}")
  endif()
  message("-- Using Ceres threading model: ${CERES_THREADING_MODEL_TO_SET}")
endmacro()
