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
#

# FindAccelerate.cmake - Find Apple's Accelerate framework, including as
#                        required components to report found the sparse solvers.
#
# This module defines the following variables which should be referenced
# by the caller to use the library.
#
# Accelerate_FOUND: TRUE iff the Accelerate framework (including the sparse
#                   solvers) and all dependencies have been found.
# Accelerate_INCLUDE_DIRS: Include directories for Accelerate framework.
# Accelerate_LIBRARIES: Libraries for Accelerate framework and all dependencies.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# Accelerate_INCLUDE_DIR: Include directory for Accelerate framework, not
#                         including the include directory of any dependencies.
# Accelerate_LIBRARY: Accelerate framework, not including the libraries of any
#                     dependencies.

# Called if we failed to find the Accelerate framework with the sparse solvers.
# Unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(accelerate_report_not_found REASON_MSG)
  unset(Accelerate_FOUND)
  unset(Accelerate_INCLUDE_DIRS)
  unset(Accelerate_LIBRARIES)
  # Make results of search visible in the CMake GUI if Accelerate has not
  # been found so that user does not have to toggle to advanced view.
  mark_as_advanced(CLEAR Accelerate_INCLUDE_DIR
                         Accelerate_LIBRARY)

  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  if (Accelerate_FIND_QUIETLY)
    message(STATUS "Failed to find Accelerate - " ${REASON_MSG} ${ARGN})
  elseif (Accelerate_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find Accelerate - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    message("-- Failed to find Accelerate - " ${REASON_MSG} ${ARGN})
  endif()
  return()
endmacro(accelerate_report_not_found)

unset(Accelerate_FOUND)

find_path(Accelerate_INCLUDE_DIR NAMES Accelerate.h)
if (NOT Accelerate_INCLUDE_DIR OR
    NOT EXISTS ${Accelerate_INCLUDE_DIR})
  accelerate_report_not_found(
    "Could not find Accelerate framework headers. Set Accelerate_INCLUDE_DIR "
    "to the directory containing Accelerate.h")
endif()

find_library(Accelerate_LIBRARY NAMES Accelerate)
if (NOT Accelerate_LIBRARY OR
    NOT EXISTS ${Accelerate_LIBRARY})
  accelerate_report_not_found(
    "Could not find Accelerate framework. Set Accelerate_LIBRARY "
    "to the Accelerate.framework directory")
endif()

set(Accelerate_FOUND TRUE)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${Accelerate_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${Accelerate_LIBRARY})
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
  accelerate_report_not_found(
    "Detected Accelerate framework: ${Accelerate_LIBRARY} does not include "
    "the sparse solvers.  Reporting Accelerate as not found.")
endif()

if (Accelerate_FOUND)
  set(Accelerate_INCLUDE_DIRS ${Accelerate_INCLUDE_DIR})
  set(Accelerate_LIBRARIES ${Accelerate_LIBRARY})
endif()

# Handle REQUIRED / QUIET optional arguments and version.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Accelerate
  REQUIRED_VARS Accelerate_INCLUDE_DIRS Accelerate_LIBRARIES)
if (Accelerate_FOUND)
  mark_as_advanced(FORCE Accelerate_INCLUDE_DIR
                         Accelerate_LIBRARY)
endif()
