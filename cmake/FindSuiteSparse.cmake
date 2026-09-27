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

#[=======================================================================[.rst:
FindSuiteSparse
===============

Module for locating SuiteSparse libraries and its dependencies.

SuiteSparse 7 and later provide a unified CMake package configuration. This
module retains a component-based fallback for older supported releases such as
SuiteSparse 5.10.1 in Ubuntu 22.04.

This module defines the following variables:

``SuiteSparse_FOUND``
   ``TRUE`` iff SuiteSparse and all dependencies have been found.

``SuiteSparse_VERSION``
   Extracted from ``SuiteSparse_config.h`` (>= v4).

``SuiteSparse_VERSION_MAJOR``
    Equal to 4 if ``SuiteSparse_VERSION`` = 4.2.1

``SuiteSparse_VERSION_MINOR``
    Equal to 2 if ``SuiteSparse_VERSION`` = 4.2.1

``SuiteSparse_VERSION_PATCH``
    Equal to 1 if ``SuiteSparse_VERSION`` = 4.2.1

The following variables control the behaviour of this module:

``SuiteSparse_NO_CMAKE``
  Do not attempt to use the native SuiteSparse CMake package configuration.


Targets
-------

The following targets define the SuiteSparse components searched for.

``SuiteSparse::AMD``
    Symmetric Approximate Minimum Degree (AMD)

``SuiteSparse::CAMD``
    Constrained Approximate Minimum Degree (CAMD)

``SuiteSparse::COLAMD``
    Column Approximate Minimum Degree (COLAMD)

``SuiteSparse::CCOLAMD``
    Constrained Column Approximate Minimum Degree (CCOLAMD)

``SuiteSparse::CHOLMOD``
    Sparse Supernodal Cholesky Factorization and Update/Downdate (CHOLMOD)

``SuiteSparse::Partition``
    CHOLMOD with METIS support

``SuiteSparse::SPQR``
    Multifrontal Sparse QR (SuiteSparseQR)

``SuiteSparse::Config``
    Common configuration for all but CSparse (SuiteSparse version >= 4).

Optional SuiteSparse dependencies:

``METIS::METIS``
    Serial Graph Partitioning and Fill-reducing Matrix Ordering (METIS)
]=======================================================================]

# Keep native package-config diagnostics deterministic.
if (SuiteSparse_FIND_COMPONENTS)
  list(SORT SuiteSparse_FIND_COMPONENTS COMPARE STRING CASE INSENSITIVE)
endif ()

include(CMakePushCheckState)

function(suitesparse_check_blas_interface OUTPUT_VARIABLE REASON_VARIABLE)
  if (NOT DEFINED MKL_INTERFACE_FULL OR MKL_INTERFACE_FULL STREQUAL "")
    set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
    return()
  endif()

  string(TOLOWER "${MKL_INTERFACE_FULL}" _mkl_interface)
  set(_lp64_integer_size 4)
  set(_lp64_integer_bits 32)
  set(_ilp64_integer_size 8)
  set(_ilp64_integer_bits 64)
  if (_mkl_interface MATCHES "ilp64")
    set(_expected_integer_size ${_ilp64_integer_size})
    set(_expected_integer_bits ${_ilp64_integer_bits})
  elseif (_mkl_interface MATCHES "lp64")
    set(_expected_integer_size ${_lp64_integer_size})
    set(_expected_integer_bits ${_lp64_integer_bits})
  else()
    message(WARNING
      "Cannot determine the BLAS/LAPACK integer size for MKL_INTERFACE_FULL="
      "${MKL_INTERFACE_FULL}. The SuiteSparse compatibility check will not "
      "verify this interface.")
    set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
    return()
  endif()

  include(CheckCXXSourceCompiles)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_QUIET TRUE)
  set(CMAKE_REQUIRED_LIBRARIES SuiteSparse::CHOLMOD)
  set(_blas_integer_prelude [=[
#include <SuiteSparse_config.h>

#ifndef SUITESPARSE_BLAS_INT
#error SUITESPARSE_BLAS_INT is unavailable
#endif
]=])
  set(_main_source "int main() { return 0; }\n")
  unset(SuiteSparse_HAS_BLAS_INTEGER CACHE)
  unset(SuiteSparse_HAS_BLAS_INTEGER)
  check_cxx_source_compiles("${_blas_integer_prelude}${_main_source}"
    SuiteSparse_HAS_BLAS_INTEGER)

  if (SuiteSparse_HAS_BLAS_INTEGER)
    set(_supported_integer_sizes
      ${_lp64_integer_size} ${_ilp64_integer_size})
    foreach(_integer_size IN LISTS _supported_integer_sizes)
      string(CONCAT _matching_blas_integer_source
        "${_blas_integer_prelude}"
        "static_assert(sizeof(SUITESPARSE_BLAS_INT) == ${_integer_size},\n"
        "              \"SuiteSparse BLAS/LAPACK integer size does not "
        "match MKL\");\n"
        "${_main_source}")
      set(_result_variable SuiteSparse_BLAS_INTEGER_IS_${_integer_size}_BYTES)
      unset(${_result_variable} CACHE)
      unset(${_result_variable})
      check_cxx_source_compiles("${_matching_blas_integer_source}"
        ${_result_variable})
      if (${_result_variable})
        set(_actual_integer_size ${_integer_size})
      endif()
    endforeach()
  endif()
  cmake_pop_check_state()

  if (NOT SuiteSparse_HAS_BLAS_INTEGER)
    string(CONCAT _reason
      "SuiteSparse does not expose SUITESPARSE_BLAS_INT. The BLAS/LAPACK "
      "integer interface cannot be checked against MKL_INTERFACE_FULL="
      "${MKL_INTERFACE_FULL}.")
    set(${REASON_VARIABLE} "${_reason}" PARENT_SCOPE)
    set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
  elseif (_actual_integer_size EQUAL _expected_integer_size)
    set(${OUTPUT_VARIABLE} TRUE PARENT_SCOPE)
  elseif (_actual_integer_size GREATER 0)
    if (_actual_integer_size EQUAL _lp64_integer_size)
      set(_actual_integer_bits ${_lp64_integer_bits})
      set(_suggested_mkl_interface intel_lp64)
    else()
      set(_actual_integer_bits ${_ilp64_integer_bits})
      set(_suggested_mkl_interface intel_ilp64)
    endif()
    string(CONCAT _reason
      "SuiteSparse BLAS/LAPACK integer interface mismatch: SuiteSparse "
      "uses ${_actual_integer_bits}-bit integers, but "
      "MKL_INTERFACE_FULL=${MKL_INTERFACE_FULL} requires "
      "${_expected_integer_bits}-bit integers. Reconfigure Ceres with "
      "-DMKL_INTERFACE_FULL=${_suggested_mkl_interface}, or rebuild "
      "SuiteSparse with a matching BLAS/LAPACK interface.")
    set(${REASON_VARIABLE} "${_reason}" PARENT_SCOPE)
    set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
  else()
    string(CONCAT _reason
      "Could not determine the SuiteSparse BLAS/LAPACK integer interface. "
      "MKL_INTERFACE_FULL=${MKL_INTERFACE_FULL} requires "
      "${_expected_integer_bits}-bit integers.")
    set(${REASON_VARIABLE} "${_reason}" PARENT_SCOPE)
    set(${OUTPUT_VARIABLE} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(suitesparse_check_cholmod_compatibility)
  if (NOT TARGET SuiteSparse::CHOLMOD)
    set(SuiteSparse_CHOLMOD_COMPATIBLE FALSE PARENT_SCOPE)
    return()
  endif()
  # Without oneMKL, Ceres does not change the BLAS and LAPACK libraries
  # SuiteSparse uses, so there is nothing to check.
  if (NOT TARGET MKL::MKL)
    set(SuiteSparse_CHOLMOD_COMPATIBLE TRUE PARENT_SCOPE)
    return()
  endif()

  suitesparse_check_blas_interface(_blas_interface_compatible
    _blas_interface_reason)
  if (NOT _blas_interface_compatible)
    set(SuiteSparse_CHOLMOD_COMPATIBLE FALSE PARENT_SCOPE)
    set(SuiteSparse_CHOLMOD_INCOMPATIBILITY_REASON
      "${_blas_interface_reason}" PARENT_SCOPE)
    return()
  endif()

  if (CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
    message(WARNING
      "Cannot run the SuiteSparse CHOLMOD compatibility check while "
      "cross compiling. Set CMAKE_CROSSCOMPILING_EMULATOR to enable it.")
    set(SuiteSparse_CHOLMOD_COMPATIBLE TRUE PARENT_SCOPE)
    return()
  endif()


  include(CheckCXXSourceRuns)
  set(_source [=[
#include <cholmod.h>

int main() {
  cholmod_common common;
  if (!cholmod_start(&common)) {
    return 1;
  }
  common.print = 0;
  // A simplicial factorization of this small matrix calls no BLAS or LAPACK
  // routine, so it would pass with a mismatched integer interface.
  common.supernodal = CHOLMOD_SUPERNODAL;

  cholmod_sparse* matrix = cholmod_allocate_sparse(
      2, 2, 3, 1, 1, 1, CHOLMOD_REAL, &common);
  if (matrix == nullptr) {
    cholmod_finish(&common);
    return 1;
  }

  // The upper triangle of [2 1; 1 2] in compressed columns. CHOLMOD ignores
  // entries below the diagonal of a matrix with stype > 0.
  auto* column_pointers = static_cast<int*>(matrix->p);
  auto* row_indices = static_cast<int*>(matrix->i);
  auto* values = static_cast<double*>(matrix->x);
  column_pointers[0] = 0;
  column_pointers[1] = 1;
  column_pointers[2] = 3;
  row_indices[0] = 0;
  row_indices[1] = 0;
  row_indices[2] = 1;
  values[0] = 2.0;
  values[1] = 1.0;
  values[2] = 2.0;

  cholmod_factor* factor = cholmod_analyze(matrix, &common);
  const bool success = factor != nullptr &&
                       cholmod_factorize(matrix, factor, &common) != 0 &&
                       common.status == CHOLMOD_OK;
  if (factor != nullptr) {
    cholmod_free_factor(&factor, &common);
  }
  cholmod_free_sparse(&matrix, &common);
  cholmod_finish(&common);
  return success ? 0 : 1;
}
]=])

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_QUIET TRUE)
  set(CMAKE_REQUIRED_LIBRARIES SuiteSparse::CHOLMOD)
  unset(SuiteSparse_CHOLMOD_FACTORIZATION_WORKS CACHE)
  unset(SuiteSparse_CHOLMOD_FACTORIZATION_WORKS)
  check_cxx_source_runs("${_source}" SuiteSparse_CHOLMOD_FACTORIZATION_WORKS)
  cmake_pop_check_state()

  if (SuiteSparse_CHOLMOD_FACTORIZATION_WORKS)
    set(SuiteSparse_CHOLMOD_COMPATIBLE TRUE PARENT_SCOPE)
  else()
    set(SuiteSparse_CHOLMOD_COMPATIBLE FALSE PARENT_SCOPE)
    string(CONCAT _reason
      "CHOLMOD factorization check failed. SuiteSparse may have been "
      "compiled against a BLAS/LAPACK interface incompatible with the "
      "selected libraries.")
    set(SuiteSparse_CHOLMOD_INCOMPATIBILITY_REASON "${_reason}" PARENT_SCOPE)
  endif()
endfunction()

# Marks CHOLMOD and SuiteSparseQR, which depends on CHOLMOD, as not found if
# CHOLMOD is incompatible with the selected BLAS and LAPACK libraries.
macro(suitesparse_require_compatible_cholmod)
  suitesparse_check_cholmod_compatibility()
  if (NOT SuiteSparse_CHOLMOD_COMPATIBLE)
    set (SuiteSparse_CHOLMOD_FOUND FALSE)
    set (SuiteSparse_SPQR_FOUND FALSE)
    list (APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_CHOLMOD_COMPATIBLE)
  endif ()
endmacro()

if (NOT SuiteSparse_NO_CMAKE)
  # CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse requires the search this module
  # performs, but it would also turn this optional search for a package
  # configuration into a required one. Distributions such as Debian ship
  # configurations only for the individual SuiteSparse libraries, so that this
  # search must be allowed to fail.
  if (DEFINED CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse)
    set (_SuiteSparse_REQUIRE_FIND_PACKAGE
      "${CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse}")
  endif ()
  set (CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse FALSE)
  find_package (SuiteSparse NO_MODULE QUIET)
  unset (CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse)
  if (DEFINED _SuiteSparse_REQUIRE_FIND_PACKAGE)
    set (CMAKE_REQUIRE_FIND_PACKAGE_SuiteSparse
      "${_SuiteSparse_REQUIRE_FIND_PACKAGE}")
    unset (_SuiteSparse_REQUIRE_FIND_PACKAGE)
  endif ()
  if (SuiteSparse_FOUND)
    # Report the main include directory instead of the package configuration
    # file path in FindPackageHandleStandardArgs' standard success message.
    get_target_property(SuiteSparse_INCLUDE_DIR SuiteSparse::Config
      INTERFACE_INCLUDE_DIRECTORIES)
    if (SuiteSparse_INCLUDE_DIR)
      list(GET SuiteSparse_INCLUDE_DIR -1 SuiteSparse_INCLUDE_DIR)
    endif ()
    set (SuiteSparse_REQUIRED_VARS SuiteSparse_INCLUDE_DIR)
    set (CMAKE_FIND_PACKAGE_REASON)
    suitesparse_require_compatible_cholmod()
    if (NOT SuiteSparse_CHOLMOD_COMPATIBLE)
      set (CMAKE_FIND_PACKAGE_REASON
        "${SuiteSparse_CHOLMOD_INCOMPATIBILITY_REASON}")
    endif ()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(SuiteSparse
      REQUIRED_VARS ${SuiteSparse_REQUIRED_VARS}
      VERSION_VAR SuiteSparse_VERSION
      REASON_FAILURE_MESSAGE "${CMAKE_FIND_PACKAGE_REASON}"
      HANDLE_COMPONENTS)
    return ()
  endif (SuiteSparse_FOUND)
endif (NOT SuiteSparse_NO_CMAKE)

# Push CMP0057 to enable support for IN_LIST, when cmake_minimum_required is
# set to <3.3.
cmake_policy (PUSH)
cmake_policy (SET CMP0057 NEW)

if (NOT SuiteSparse_FIND_COMPONENTS)
  set (SuiteSparse_FIND_COMPONENTS
    AMD
    CAMD
    CCOLAMD
    CHOLMOD
    COLAMD
    SPQR
  )

  foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
    set (SuiteSparse_FIND_REQUIRED_${component} TRUE)
  endforeach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
endif (NOT SuiteSparse_FIND_COMPONENTS)

# Assume SuiteSparse was found and set it to false when a component or a
# third-party dependency could not be located. SuiteSparse component failures
# are reported by FindPackageHandleStandardArgs HANDLE_COMPONENTS.
set (SuiteSparse_FOUND TRUE)
set (CMAKE_FIND_PACKAGE_REASON)

# Keep nested dependency failures available for SuiteSparse's final reason.
macro (suitesparse_find_dependency DEPENDENCY)
  find_package(${DEPENDENCY} ${ARGN} QUIET)
  if (NOT ${DEPENDENCY}_FOUND)
    set (SuiteSparse_${DEPENDENCY}_REASON
      "${DEPENDENCY}: Could not find ${DEPENDENCY}.")
    if (${DEPENDENCY}_NOT_FOUND_MESSAGE)
      set (SuiteSparse_${DEPENDENCY}_REASON
        "${DEPENDENCY}: ${${DEPENDENCY}_NOT_FOUND_MESSAGE}")
    endif ()
  endif ()
endmacro (suitesparse_find_dependency)

# SuiteSparseQR optionally depends on TBB. Find it here so that it is treated
# as a SuiteSparse dependency rather than as a Ceres dependency.
suitesparse_find_dependency(TBB NO_MODULE)

include (CheckLibraryExists)
include (CheckSymbolExists)
include (CMakePushCheckState)

# Config is a base component and thus always required
set (SuiteSparse_IMPLICIT_COMPONENTS Config)

# CHOLMOD depends on AMD, CAMD, CCOLAMD, and COLAMD.
if (CHOLMOD IN_LIST SuiteSparse_FIND_COMPONENTS)
  list (APPEND SuiteSparse_IMPLICIT_COMPONENTS AMD CAMD CCOLAMD COLAMD)
endif (CHOLMOD IN_LIST SuiteSparse_FIND_COMPONENTS)

# SPQR depends on CHOLMOD.
if (SPQR IN_LIST SuiteSparse_FIND_COMPONENTS)
  list (APPEND SuiteSparse_IMPLICIT_COMPONENTS CHOLMOD)
endif (SPQR IN_LIST SuiteSparse_FIND_COMPONENTS)

# Implicit components are always required
foreach (component IN LISTS SuiteSparse_IMPLICIT_COMPONENTS)
  set (SuiteSparse_FIND_REQUIRED_${component} TRUE)
endforeach (component IN LISTS SuiteSparse_IMPLICIT_COMPONENTS)

list (APPEND SuiteSparse_FIND_COMPONENTS ${SuiteSparse_IMPLICIT_COMPONENTS})

# Do not list components multiple times.
list (REMOVE_DUPLICATES SuiteSparse_FIND_COMPONENTS)
list (SORT SuiteSparse_FIND_COMPONENTS COMPARE STRING CASE INSENSITIVE)

# Reset CALLERS_CMAKE_FIND_LIBRARY_PREFIXES to its value when
# FindSuiteSparse was invoked.
macro(SuiteSparse_RESET_FIND_LIBRARY_PREFIX)
  if (MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CALLERS_CMAKE_FIND_LIBRARY_PREFIXES}")
  endif (MSVC)
endmacro(SuiteSparse_RESET_FIND_LIBRARY_PREFIX)

# Called if we failed to find SuiteSparse or any of its required dependencies.
# The standard package helper reports the failure after all components have
# been checked.
macro(SuiteSparse_REPORT_NOT_FOUND REASON_MSG)
  set (SuiteSparse_FOUND FALSE)
  list (APPEND CMAKE_FIND_PACKAGE_REASON "${REASON_MSG}")

  # Do NOT unset SuiteSparse_REQUIRED_VARS here, as it is used by
  # FindPackageHandleStandardArgs() to generate the automatic error message on
  # failure which highlights which components are missing.

  suitesparse_reset_find_library_prefix()

  # Do not return so all components can be checked before standard reporting.
endmacro(SuiteSparse_REPORT_NOT_FOUND)

# Handle possible presence of lib prefix for libraries on MSVC, see
# also SuiteSparse_RESET_FIND_LIBRARY_PREFIX().
if (MSVC)
  # Preserve the caller's original values for CMAKE_FIND_LIBRARY_PREFIXES
  # s/t we can set it back before returning.
  set(CALLERS_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  # The empty string in this list is important, it represents the case when
  # the libraries have no prefix (shared libraries / DLLs).
  set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "" "${CMAKE_FIND_LIBRARY_PREFIXES}")
endif (MSVC)

# Additional suffixes to try appending to each search path.
list(APPEND SuiteSparse_CHECK_PATH_SUFFIXES
  suitesparse) # Windows/Ubuntu

# Wrappers to find_path/library that pass the SuiteSparse search hints/paths.
#
# suitesparse_find_component(<component> [FILES name1 [name2 ...]]
#                                        [LIBRARIES name1 [name2 ...]])
macro(suitesparse_find_component COMPONENT)
  include(CMakeParseArguments)
  set(MULTI_VALUE_ARGS FILES LIBRARIES)
  cmake_parse_arguments(SuiteSparse_FIND_COMPONENT_${COMPONENT}
    "" "" "${MULTI_VALUE_ARGS}" ${ARGN})

  set(SuiteSparse_${COMPONENT}_FOUND TRUE)
  if (SuiteSparse_FIND_COMPONENT_${COMPONENT}_FILES)
    find_path(SuiteSparse_${COMPONENT}_INCLUDE_DIR
      NAMES ${SuiteSparse_FIND_COMPONENT_${COMPONENT}_FILES}
      PATH_SUFFIXES ${SuiteSparse_CHECK_PATH_SUFFIXES})
    if (SuiteSparse_${COMPONENT}_INCLUDE_DIR)
      mark_as_advanced(SuiteSparse_${COMPONENT}_INCLUDE_DIR)
    else()
      # Specified headers not found.
      set(SuiteSparse_${COMPONENT}_FOUND FALSE)
      if (SuiteSparse_FIND_REQUIRED_${COMPONENT})
        set(SuiteSparse_FOUND FALSE)
      else()
        # Hide optional vars from CMake GUI even if not found.
        mark_as_advanced(SuiteSparse_${COMPONENT}_INCLUDE_DIR)
      endif()
    endif()
  endif()

  if (SuiteSparse_FIND_COMPONENT_${COMPONENT}_LIBRARIES)
    find_library(SuiteSparse_${COMPONENT}_LIBRARY
      NAMES ${SuiteSparse_FIND_COMPONENT_${COMPONENT}_LIBRARIES}
      PATH_SUFFIXES ${SuiteSparse_CHECK_PATH_SUFFIXES})
    if (SuiteSparse_${COMPONENT}_LIBRARY)
      mark_as_advanced(SuiteSparse_${COMPONENT}_LIBRARY)
    else ()
      # Specified libraries not found.
      set(SuiteSparse_${COMPONENT}_FOUND FALSE)
      if (SuiteSparse_FIND_REQUIRED_${COMPONENT})
        set(SuiteSparse_FOUND FALSE)
      else()
        # Hide optional vars from CMake GUI even if not found.
        mark_as_advanced(SuiteSparse_${COMPONENT}_LIBRARY)
      endif()
    endif()
  endif()

  # A component can be optional (given to OPTIONAL_COMPONENTS). However, if the
  # component is implicit (must be always present, such as the Config component)
  # assume it be required as well.
  if (SuiteSparse_FIND_REQUIRED_${COMPONENT})
    list (APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_${COMPONENT}_INCLUDE_DIR)
    list (APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_${COMPONENT}_LIBRARY)
  endif (SuiteSparse_FIND_REQUIRED_${COMPONENT})

  # Define the target only if the include directory and the library were found
  if (SuiteSparse_${COMPONENT}_INCLUDE_DIR AND SuiteSparse_${COMPONENT}_LIBRARY)
    if (NOT TARGET SuiteSparse::${COMPONENT})
      add_library(SuiteSparse::${COMPONENT} IMPORTED UNKNOWN)
    endif (NOT TARGET SuiteSparse::${COMPONENT})

    set_property(TARGET SuiteSparse::${COMPONENT} PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${SuiteSparse_${COMPONENT}_INCLUDE_DIR})
    set_property(TARGET SuiteSparse::${COMPONENT} PROPERTY
      IMPORTED_LOCATION ${SuiteSparse_${COMPONENT}_LIBRARY})
  endif (SuiteSparse_${COMPONENT}_INCLUDE_DIR AND SuiteSparse_${COMPONENT}_LIBRARY)
endmacro()

# Given the number of components of SuiteSparse, and to ensure that the
# automatic failure message generated by FindPackageHandleStandardArgs()
# when not all required components are found is helpful, we maintain a list
# of all variables that must be defined for SuiteSparse to be considered found.
unset(SuiteSparse_REQUIRED_VARS)

# BLAS.
if (NOT DEFINED BLAS_FOUND)
  suitesparse_find_dependency(BLAS)
endif()

# LAPACK.
if (NOT DEFINED LAPACK_FOUND)
  suitesparse_find_dependency(LAPACK)
endif()

foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
  if (component STREQUAL Partition)
    # Partition is a meta component that neither provides additional headers nor
    # a separate library. It is strictly part of CHOLMOD.
    continue ()
  endif (component STREQUAL Partition)
  string (TOLOWER ${component} component_library)

  if (component STREQUAL "Config")
    set (component_header SuiteSparse_config.h)
    set (component_library suitesparseconfig)
  elseif (component STREQUAL "SPQR")
    set (component_header SuiteSparseQR.hpp)
  else (component STREQUAL "SPQR")
    set (component_header ${component_library}.h)
  endif (component STREQUAL "Config")

  suitesparse_find_component(${component}
    FILES ${component_header}
    LIBRARIES ${component_library})
endforeach (component IN LISTS SuiteSparse_FIND_COMPONENTS)

check_library_exists(rt shm_open "" HAVE_LIBRT)

if (TARGET SuiteSparse::Config)
  # SuiteSparse version >= 4.
  set(SuiteSparse_VERSION_FILE
    ${SuiteSparse_Config_INCLUDE_DIR}/SuiteSparse_config.h)
  if (NOT EXISTS ${SuiteSparse_VERSION_FILE})
    list(APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_VERSION)
    suitesparse_report_not_found(
      "Could not find file: ${SuiteSparse_VERSION_FILE} containing version "
      "information for >= v4 SuiteSparse installs, but SuiteSparse_config was "
      "found (only present in >= v4 installs).")
  else (NOT EXISTS ${SuiteSparse_VERSION_FILE})
    file(READ ${SuiteSparse_VERSION_FILE} Config_CONTENTS)

    string(REGEX MATCH "#define SUITESPARSE_MAIN_VERSION[ \t]+([0-9]+)"
      SuiteSparse_VERSION_LINE "${Config_CONTENTS}")
    set (SuiteSparse_VERSION_MAJOR ${CMAKE_MATCH_1})

    string(REGEX MATCH "#define SUITESPARSE_SUB_VERSION[ \t]+([0-9]+)"
      SuiteSparse_VERSION_LINE "${Config_CONTENTS}")
    set (SuiteSparse_VERSION_MINOR ${CMAKE_MATCH_1})

    string(REGEX MATCH "#define SUITESPARSE_SUBSUB_VERSION[ \t]+([0-9]+)"
      SuiteSparse_VERSION_LINE "${Config_CONTENTS}")
    set (SuiteSparse_VERSION_PATCH ${CMAKE_MATCH_1})

    unset (SuiteSparse_VERSION_LINE)

    # This is on a single line s/t CMake does not interpret it as a list of
    # elements and insert ';' separators which would result in 4.;2.;1 nonsense.
    set(SuiteSparse_VERSION
      "${SuiteSparse_VERSION_MAJOR}.${SuiteSparse_VERSION_MINOR}.${SuiteSparse_VERSION_PATCH}")

    if (SuiteSparse_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
      set(SuiteSparse_VERSION_COMPONENTS 3)
    else (SuiteSparse_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
      message (WARNING "Could not parse SuiteSparse_config.h: SuiteSparse "
        "version will not be available")

      unset (SuiteSparse_VERSION)
      unset (SuiteSparse_VERSION_MAJOR)
      unset (SuiteSparse_VERSION_MINOR)
      unset (SuiteSparse_VERSION_PATCH)
      list(APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_VERSION)
    endif (SuiteSparse_VERSION MATCHES "[0-9]+\\.[0-9]+\\.[0-9]+")
  endif (NOT EXISTS ${SuiteSparse_VERSION_FILE})
endif (TARGET SuiteSparse::Config)

# CHOLMOD requires AMD CAMD CCOLAMD COLAMD
if (TARGET SuiteSparse::CHOLMOD)
  foreach (component IN ITEMS AMD CAMD CCOLAMD COLAMD)
    if (TARGET SuiteSparse::${component})
      set_property (TARGET SuiteSparse::CHOLMOD APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES SuiteSparse::${component})
    else (TARGET SuiteSparse::${component})
      # Consider CHOLMOD not found if COLAMD cannot be found
      set (SuiteSparse_CHOLMOD_FOUND FALSE)
      set (SuiteSparse_FOUND FALSE)
    endif (TARGET SuiteSparse::${component})
  endforeach (component IN ITEMS AMD CAMD CCOLAMD COLAMD)
endif (TARGET SuiteSparse::CHOLMOD)

# SPQR requires CHOLMOD
if (TARGET SuiteSparse::SPQR)
  if (TARGET SuiteSparse::CHOLMOD)
    set_property (TARGET SuiteSparse::SPQR APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES SuiteSparse::CHOLMOD)
  else (TARGET SuiteSparse::CHOLMOD)
    # Consider SPQR not found if CHOLMOD cannot be found
    set (SuiteSparse_SPQR_FOUND FALSE)
    set (SuiteSparse_FOUND FALSE)
  endif (TARGET SuiteSparse::CHOLMOD)
endif (TARGET SuiteSparse::SPQR)

# Add SuiteSparse::Config as dependency to all components
if (TARGET SuiteSparse::Config)
  foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
    if (component STREQUAL Config)
      continue ()
    endif (component STREQUAL Config)

    if (TARGET SuiteSparse::${component})
      set_property (TARGET SuiteSparse::${component} APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES SuiteSparse::Config)
    endif (TARGET SuiteSparse::${component})
  endforeach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
endif (TARGET SuiteSparse::Config)

# Check whether the SuiteSparse libraries need their optional dependencies.
# This avoids adding libraries that are available but are not required by the
# installed SuiteSparse build.
function (suitesparse_check_link RESULT SYMBOL)
  set (SuiteSparse_LINK_CHECK_SOURCE
    "${CMAKE_BINARY_DIR}/CMakeFiles/SuiteSparseLinkCheck.cxx")
  file (WRITE "${SuiteSparse_LINK_CHECK_SOURCE}"
    "extern \"C\" void ${SYMBOL}(void);\n"
    "int main(void) { ${SYMBOL}(); return 0; }\n")

  unset (SuiteSparse_LINK_CHECK_RESULT CACHE)
  unset (SuiteSparse_LINK_CHECK_RESULT)
  try_compile (SuiteSparse_LINK_CHECK_RESULT
    "${CMAKE_BINARY_DIR}/CMakeFiles/SuiteSparseLinkCheck"
    "${SuiteSparse_LINK_CHECK_SOURCE}"
    LINK_LIBRARIES ${ARGN}
    OUTPUT_VARIABLE SuiteSparse_LINK_CHECK_OUTPUT)
  set (${RESULT} ${SuiteSparse_LINK_CHECK_RESULT} PARENT_SCOPE)
  unset (SuiteSparse_LINK_CHECK_RESULT CACHE)
  unset (SuiteSparse_LINK_CHECK_RESULT)
endfunction (suitesparse_check_link)

set (SuiteSparse_LINK_TARGET)
set (SuiteSparse_LINK_SYMBOL)
if (TARGET SuiteSparse::SPQR)
  set (SuiteSparse_LINK_TARGET SuiteSparse::SPQR)
  set (SuiteSparse_LINK_SYMBOL SuiteSparseQR_C_symbolic)
elseif (TARGET SuiteSparse::CHOLMOD)
  set (SuiteSparse_LINK_TARGET SuiteSparse::CHOLMOD)
  set (SuiteSparse_LINK_SYMBOL cholmod_start)
endif ()

set (SuiteSparse_BLAS_LINK)
if (TARGET BLAS::BLAS)
  set (SuiteSparse_BLAS_LINK BLAS::BLAS)
elseif (BLAS_LIBRARIES)
  set (SuiteSparse_BLAS_LINK ${BLAS_LIBRARIES})
endif ()

set (SuiteSparse_LAPACK_LINK)
if (TARGET LAPACK::LAPACK)
  set (SuiteSparse_LAPACK_LINK LAPACK::LAPACK)
elseif (LAPACK_LIBRARIES)
  set (SuiteSparse_LAPACK_LINK ${LAPACK_LIBRARIES})
endif ()

set (SuiteSparse_TBB_LINK)
if (TARGET TBB::tbb)
  set (SuiteSparse_TBB_LINK TBB::tbb)
elseif (TBB_LIBRARIES)
  set (SuiteSparse_TBB_LINK ${TBB_LIBRARIES})
endif ()

set (SuiteSparse_RT_LINK)
if (HAVE_LIBRT)
  set (SuiteSparse_RT_LINK rt)
endif ()

set (SuiteSparse_OPTIONAL_DEPENDENCIES)
if (SuiteSparse_BLAS_LINK)
  list (APPEND SuiteSparse_OPTIONAL_DEPENDENCIES BLAS)
endif ()
if (SuiteSparse_LAPACK_LINK)
  list (APPEND SuiteSparse_OPTIONAL_DEPENDENCIES LAPACK)
endif ()
if (TARGET SuiteSparse::SPQR AND SuiteSparse_TBB_LINK)
  list (APPEND SuiteSparse_OPTIONAL_DEPENDENCIES TBB)
endif ()
if (SuiteSparse_RT_LINK)
  list (APPEND SuiteSparse_OPTIONAL_DEPENDENCIES RT)
endif ()

if (SuiteSparse_FOUND AND SuiteSparse_LINK_TARGET)
  get_target_property(SuiteSparse_ORIGINAL_CONFIG_LINK
    SuiteSparse::Config INTERFACE_LINK_LIBRARIES)
  if (SuiteSparse_ORIGINAL_CONFIG_LINK MATCHES "-NOTFOUND$")
    set (SuiteSparse_ORIGINAL_CONFIG_LINK)
  endif ()
  if (TARGET SuiteSparse::SPQR)
    get_target_property(SuiteSparse_ORIGINAL_SPQR_LINK
      SuiteSparse::SPQR INTERFACE_LINK_LIBRARIES)
    if (SuiteSparse_ORIGINAL_SPQR_LINK MATCHES "-NOTFOUND$")
      set (SuiteSparse_ORIGINAL_SPQR_LINK)
    endif ()
  endif ()

  function (suitesparse_set_link_dependencies)
    set (SuiteSparse_CONFIG_LINK ${SuiteSparse_ORIGINAL_CONFIG_LINK})
    if (BLAS IN_LIST ARGN)
      list (APPEND SuiteSparse_CONFIG_LINK ${SuiteSparse_BLAS_LINK})
    endif ()
    if (LAPACK IN_LIST ARGN)
      list (APPEND SuiteSparse_CONFIG_LINK ${SuiteSparse_LAPACK_LINK})
    endif ()
    if (RT IN_LIST ARGN)
      list (APPEND SuiteSparse_CONFIG_LINK ${SuiteSparse_RT_LINK})
    endif ()
    set_property (TARGET SuiteSparse::Config PROPERTY
      INTERFACE_LINK_LIBRARIES "${SuiteSparse_CONFIG_LINK}")

    if (TARGET SuiteSparse::SPQR)
      set (SuiteSparse_SPQR_LINK ${SuiteSparse_ORIGINAL_SPQR_LINK})
      if (TBB IN_LIST ARGN)
        list (APPEND SuiteSparse_SPQR_LINK ${SuiteSparse_TBB_LINK})
      endif ()
      set_property (TARGET SuiteSparse::SPQR PROPERTY
        INTERFACE_LINK_LIBRARIES "${SuiteSparse_SPQR_LINK}")
    endif ()
  endfunction (suitesparse_set_link_dependencies)

  suitesparse_set_link_dependencies(${SuiteSparse_OPTIONAL_DEPENDENCIES})
  suitesparse_check_link(SuiteSparse_LINKS
    ${SuiteSparse_LINK_SYMBOL} ${SuiteSparse_LINK_TARGET})
  if (SuiteSparse_LINKS)
    set (SuiteSparse_REQUIRED_DEPENDENCIES)
    foreach (dependency IN LISTS SuiteSparse_OPTIONAL_DEPENDENCIES)
      set (SuiteSparse_DEPENDENCIES_WITHOUT
        ${SuiteSparse_OPTIONAL_DEPENDENCIES})
      list (REMOVE_ITEM SuiteSparse_DEPENDENCIES_WITHOUT ${dependency})
      suitesparse_set_link_dependencies(${SuiteSparse_DEPENDENCIES_WITHOUT})
      suitesparse_check_link(SuiteSparse_LINKS_WITHOUT_DEPENDENCY
        ${SuiteSparse_LINK_SYMBOL} ${SuiteSparse_LINK_TARGET})
      if (NOT SuiteSparse_LINKS_WITHOUT_DEPENDENCY)
        list (APPEND SuiteSparse_REQUIRED_DEPENDENCIES ${dependency})
      endif ()
    endforeach ()

    suitesparse_set_link_dependencies(${SuiteSparse_REQUIRED_DEPENDENCIES})
  else ()
    suitesparse_set_link_dependencies()
    list (APPEND SuiteSparse_REQUIRED_VARS SuiteSparse_LINKS)
    set (SuiteSparse_MISSING_DEPENDENCIES)
    if (NOT SuiteSparse_BLAS_LINK)
      list (APPEND SuiteSparse_MISSING_DEPENDENCIES BLAS)
    endif ()
    if (NOT SuiteSparse_LAPACK_LINK)
      list (APPEND SuiteSparse_MISSING_DEPENDENCIES LAPACK)
    endif ()
    if (TARGET SuiteSparse::SPQR AND NOT SuiteSparse_TBB_LINK)
      list (APPEND SuiteSparse_MISSING_DEPENDENCIES TBB)
    endif ()
    if (NOT SuiteSparse_RT_LINK)
      list (APPEND SuiteSparse_MISSING_DEPENDENCIES RT)
    endif ()
    if (SuiteSparse_MISSING_DEPENDENCIES)
      foreach (dependency IN LISTS SuiteSparse_MISSING_DEPENDENCIES)
        if (SuiteSparse_${dependency}_REASON)
          list (APPEND CMAKE_FIND_PACKAGE_REASON
            "${SuiteSparse_${dependency}_REASON}")
        endif ()
      endforeach ()
      set (SuiteSparse_LINK_FAILURE_REASON
        "SuiteSparse libraries could not be linked with the detected "
        "dependencies. The following dependencies were not found, so their "
        "necessity could not be determined: ${SuiteSparse_MISSING_DEPENDENCIES}.")
      suitesparse_report_not_found("${SuiteSparse_LINK_FAILURE_REASON}")
    else ()
      set (SuiteSparse_LINK_FAILURE_REASON
        "SuiteSparse libraries could not be linked with the detected "
        "dependencies.")
      suitesparse_report_not_found("${SuiteSparse_LINK_FAILURE_REASON}")
    endif ()
  endif ()
endif ()

# Check whether CHOLMOD was compiled with METIS support. The check can be
# performed only after the main components have been set up.
if (TARGET SuiteSparse::CHOLMOD)
  # NOTE If SuiteSparse was compiled as a static library we'll need to link
  # against METIS already during the check. Otherwise, the check can fail due to
  # undefined references even though SuiteSparse was compiled with METIS.
  #
  # Other METIS find modules may define METIS_FOUND without providing the
  # METIS::METIS target required for linking.
  if (NOT TARGET METIS::METIS)
    find_package (METIS)
  endif (NOT TARGET METIS::METIS)

  if (TARGET METIS::METIS)
    cmake_push_check_state (RESET)
    set (CMAKE_REQUIRED_LIBRARIES SuiteSparse::CHOLMOD METIS::METIS)
    check_symbol_exists (cholmod_metis cholmod.h SuiteSparse_CHOLMOD_USES_METIS)
    cmake_pop_check_state ()

    if (NOT SuiteSparse_CHOLMOD_USES_METIS AND SuiteSparse_FIND_REQUIRED_Partition)
      list (APPEND CMAKE_FIND_PACKAGE_REASON
        "Partition: CHOLMOD was not compiled with METIS support.")
    endif (NOT SuiteSparse_CHOLMOD_USES_METIS AND SuiteSparse_FIND_REQUIRED_Partition)

    if (SuiteSparse_CHOLMOD_USES_METIS)
      set_property (TARGET SuiteSparse::CHOLMOD APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES $<LINK_ONLY:METIS::METIS>)

      # Provide the SuiteSparse::Partition component whose availability indicates
      # that CHOLMOD was compiled with the Partition module.
      if (NOT TARGET SuiteSparse::Partition)
        add_library (SuiteSparse::Partition IMPORTED INTERFACE)
      endif (NOT TARGET SuiteSparse::Partition)

      set_property (TARGET SuiteSparse::Partition APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES SuiteSparse::CHOLMOD)
    endif (SuiteSparse_CHOLMOD_USES_METIS)
  elseif (SuiteSparse_FIND_REQUIRED_Partition)
    list (APPEND CMAKE_FIND_PACKAGE_REASON
      "Partition: METIS could not be found.")
  endif (TARGET METIS::METIS)
endif (TARGET SuiteSparse::CHOLMOD)

# We do not use suitesparse_find_component to find Partition and therefore must
# handle the availability in an extra step.
if (TARGET SuiteSparse::Partition)
  set (SuiteSparse_Partition_FOUND TRUE)
else (TARGET SuiteSparse::Partition)
  set (SuiteSparse_Partition_FOUND FALSE)
endif (TARGET SuiteSparse::Partition)

suitesparse_reset_find_library_prefix()

if (SuiteSparse_FOUND AND TARGET SuiteSparse::CHOLMOD)
  suitesparse_require_compatible_cholmod()
  if (NOT SuiteSparse_CHOLMOD_COMPATIBLE)
    suitesparse_report_not_found(
      "${SuiteSparse_CHOLMOD_INCOMPATIBILITY_REASON}")
  endif ()
endif ()

list(REMOVE_DUPLICATES SuiteSparse_REQUIRED_VARS)

# Handle REQUIRED and QUIET arguments to FIND_PACKAGE.
include(FindPackageHandleStandardArgs)
list(REMOVE_DUPLICATES CMAKE_FIND_PACKAGE_REASON)
string(JOIN "\n    " CMAKE_FIND_PACKAGE_REASON
  ${CMAKE_FIND_PACKAGE_REASON})
find_package_handle_standard_args(SuiteSparse
  REQUIRED_VARS ${SuiteSparse_REQUIRED_VARS}
  VERSION_VAR SuiteSparse_VERSION
  REASON_FAILURE_MESSAGE "${CMAKE_FIND_PACKAGE_REASON}"
  HANDLE_COMPONENTS)

# Pop CMP0057.
cmake_policy (POP)
