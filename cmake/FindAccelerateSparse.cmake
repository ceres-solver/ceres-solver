# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2026 Google Inc. All rights reserved.
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
#

# FindAccelerateSparse.cmake - Find the sparse solvers in Apple's Accelerate
#                              framework, introduced in Xcode 9.0 (2017).
#                              Note that this is distinct from the Accelerate
#                              framework on its own, which existed in previous
#                              versions but without the sparse solvers.
#
# This module defines the following variables.
#
# AccelerateSparse_FOUND: TRUE iff an Accelerate framework including the sparse
#                         solvers, and all dependencies, has been found.
#
# The following imported target is also defined when the package is found:
#
# AccelerateSparse::Accelerate: Accelerate framework with sparse solvers.
#
# The following variables affect the behavior of the module via
# FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# AccelerateSparse_INCLUDE_DIR: Include directory for Accelerate framework, not
#                               including the include directory of any
#                               dependencies.
# AccelerateSparse_LIBRARY: Accelerate framework, not including the libraries of
#                           any dependencies.

# Called if we failed to find the Accelerate framework with the sparse solvers.
# Lets the standard package helper report the failure.
macro(accelerate_sparse_report_not_found REASON_MSG)
  unset(AccelerateSparse_FOUND)
  # Make results of search visible in the CMake GUI if Accelerate has not
  # been found so that user does not have to toggle to advanced view.
  mark_as_advanced(CLEAR AccelerateSparse_INCLUDE_DIR
                         AccelerateSparse_LIBRARY)

  if (ARGN)
    string(JOIN " " ACCELERATE_SPARSE_FAILURE_REASON
      "${REASON_MSG}" ${ARGN})
  else()
    set(ACCELERATE_SPARSE_FAILURE_REASON "${REASON_MSG}")
  endif()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(AccelerateSparse
    REQUIRED_VARS AccelerateSparse_INCLUDE_DIR AccelerateSparse_LIBRARY
    REASON_FAILURE_MESSAGE "${ACCELERATE_SPARSE_FAILURE_REASON}")
  return()
endmacro()

unset(AccelerateSparse_FOUND)

find_path(AccelerateSparse_INCLUDE_DIR NAMES Accelerate.h)
if (NOT AccelerateSparse_INCLUDE_DIR OR
    NOT EXISTS ${AccelerateSparse_INCLUDE_DIR})
  accelerate_sparse_report_not_found(
    "Could not find AccelerateSparse because the Accelerate framework headers were not found")
endif()

find_library(AccelerateSparse_LIBRARY NAMES Accelerate)
if (NOT AccelerateSparse_LIBRARY OR
    NOT EXISTS ${AccelerateSparse_LIBRARY})
  accelerate_sparse_report_not_found(
    "Could not find AccelerateSparse because the Accelerate framework was not found. Set AccelerateSparse_LIBRARY to the Accelerate.framework directory")
endif()

set(AccelerateSparse_FOUND TRUE)

# Determine if the Accelerate framework detected includes the sparse solvers.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${AccelerateSparse_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${AccelerateSparse_LIBRARY})
check_cxx_source_compiles(
  "#include <Accelerate.h>
   int main() {
     SparseMatrix_Double A;
     SparseFactor(SparseFactorizationCholesky, A);
     return 0;
   }"
   ACCELERATE_FRAMEWORK_HAS_SPARSE_SOLVER)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)
if (NOT ACCELERATE_FRAMEWORK_HAS_SPARSE_SOLVER)
  accelerate_sparse_report_not_found(
    "Could not find AccelerateSparse because the Accelerate framework ${AccelerateSparse_LIBRARY} does not include the sparse solvers")
endif()

# Handle REQUIRED / QUIET optional arguments and version.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AccelerateSparse
  REQUIRED_VARS AccelerateSparse_INCLUDE_DIR AccelerateSparse_LIBRARY)
if (AccelerateSparse_FOUND)
  if (NOT TARGET AccelerateSparse::Accelerate)
    add_library(AccelerateSparse::Accelerate UNKNOWN IMPORTED)
    set_target_properties(AccelerateSparse::Accelerate PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${AccelerateSparse_INCLUDE_DIR}"
      IMPORTED_LOCATION "${AccelerateSparse_LIBRARY}")
  endif()
  mark_as_advanced(FORCE AccelerateSparse_INCLUDE_DIR
                         AccelerateSparse_LIBRARY)
endif()
