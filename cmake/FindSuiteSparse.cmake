# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2022 Google Inc. All rights reserved.
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

``SuiteSparse_INCLUDE_DIR_HINTS``
    List of additional directories in which to search for SuiteSparse includes,
    e.g.: ``/timbuktu/include``.

``SuiteSparse_LIBRARY_DIR_HINTS``
    List of additional directories in which to search for SuiteSparse libraries,
    e.g: ``/timbuktu/lib``.


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

``SuiteSparse::SPQR``
    Multifrontal Sparse QR (SuiteSparseQR)

``SuiteSparse::Config``
    Common configuration for all but CSparse (SuiteSparse version >= 4).

Optional SuiteSparse dependencies:

``METIS::METIS``
    Serial Graph Partitioning and Fill-reducing Matrix Ordering (METIS)
]=======================================================================]

if (NOT SuiteSparse_NO_CMAKE)
  find_package (SuiteSparse NO_MODULE QUIET)
endif (NOT SuiteSparse_NO_CMAKE)

if (SuiteSparse_FOUND)
  return ()
endif (SuiteSparse_FOUND)

include (CheckLibraryExists)

# CHOLMOD depends on AMD, CAMD, CCOLAMD, and COLAMD
if (CHOLMOD IN_LIST SuiteSparse_FIND_COMPONENTS)
  list (APPEND SuiteSparse_FIND_COMPONENTS AMD CAMD CCOLAMD COLAMD)
endif (CHOLMOD IN_LIST SuiteSparse_FIND_COMPONENTS)

# CHOLMOD depends on CHOLMOD
if (SPQR IN_LIST SuiteSparse_FIND_COMPONENTS)
  list (APPEND SuiteSparse_FIND_COMPONENTS CHOLMOD)
endif (SPQR IN_LIST SuiteSparse_FIND_COMPONENTS)

# Do note list components multiple times
list (REMOVE_DUPLICATES SuiteSparse_FIND_COMPONENTS)

# Reset CALLERS_CMAKE_FIND_LIBRARY_PREFIXES to its value when
# FindSuiteSparse was invoked.
macro(SuiteSparse_RESET_FIND_LIBRARY_PREFIX)
  if (MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CALLERS_CMAKE_FIND_LIBRARY_PREFIXES}")
  endif (MSVC)
endmacro(SuiteSparse_RESET_FIND_LIBRARY_PREFIX)

# Called if we failed to find SuiteSparse or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(SuiteSparse_REPORT_NOT_FOUND REASON_MSG)
  # Do NOT unset SuiteSparse_FOUND_REQUIRED_VARS here, as it is used by
  # FindPackageHandleStandardArgs() to generate the automatic error message on
  # failure which highlights which components are missing.

  suitesparse_reset_find_library_prefix()

  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  if (SuiteSparse_FIND_QUIETLY)
    message(STATUS "Failed to find SuiteSparse - " ${REASON_MSG} ${ARGN})
  elseif (SuiteSparse_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find SuiteSparse - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    message("-- Failed to find SuiteSparse - " ${REASON_MSG} ${ARGN})
  endif (SuiteSparse_FIND_QUIETLY)

  # Do not call return(), s/t we keep processing if not called with REQUIRED
  # and report all missing components, rather than bailing after failing to find
  # the first.
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

# On macOS, add the Homebrew prefix (with appropriate suffixes) to the
# respective HINTS directories (after any user-specified locations).  This
# handles Homebrew installations into non-standard locations (not /usr/local).
# We do not use CMAKE_PREFIX_PATH for this as given the search ordering of
# find_xxx(), doing so would override any user-specified HINTS locations with
# the Homebrew version if it exists.
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  find_program(HOMEBREW_EXECUTABLE brew)
  mark_as_advanced(FORCE HOMEBREW_EXECUTABLE)
  if (HOMEBREW_EXECUTABLE)
    # Detected a Homebrew install, query for its install prefix.
    execute_process(COMMAND ${HOMEBREW_EXECUTABLE} --prefix
      OUTPUT_VARIABLE HOMEBREW_INSTALL_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Detected Homebrew with install prefix: "
      "${HOMEBREW_INSTALL_PREFIX}, adding to CMake search paths.")
    list(APPEND SuiteSparse_INCLUDE_DIR_HINTS "${HOMEBREW_INSTALL_PREFIX}/include")
    list(APPEND SuiteSparse_LIBRARY_DIR_HINTS "${HOMEBREW_INSTALL_PREFIX}/lib")
  endif()
endif()

# Specify search directories for include files and libraries (this is the union
# of the search directories for all OSs).  Search user-specified hint
# directories first if supplied, and search user-installed locations first
# so that we prefer user installs to system installs where both exist.
list(APPEND SuiteSparse_CHECK_INCLUDE_DIRS
  /opt/local/include
  /opt/local/include/ufsparse # Mac OS X
  /usr/local/homebrew/include # Mac OS X
  /usr/local/include
  /usr/include)
list(APPEND SuiteSparse_CHECK_LIBRARY_DIRS
  /opt/local/lib
  /opt/local/lib/ufsparse # Mac OS X
  /usr/local/homebrew/lib # Mac OS X
  /usr/local/lib
  /usr/lib)
# Additional suffixes to try appending to each search path.
list(APPEND SuiteSparse_CHECK_PATH_SUFFIXES
  suitesparse) # Windows/Ubuntu

# Wrappers to find_path/library that pass the SuiteSparse search hints/paths.
#
# suitesparse_find_component(<component> [FILES name1 [name2 ...]]
#                                        [LIBRARIES name1 [name2 ...]]
#                                        [REQUIRED])
macro(suitesparse_find_component COMPONENT)
  include(CMakeParseArguments)
  set(OPTIONS REQUIRED)
  set(MULTI_VALUE_ARGS FILES LIBRARIES)
  cmake_parse_arguments(SuiteSparse_FIND_${COMPONENT}
    "${OPTIONS}" "" "${MULTI_VALUE_ARGS}" ${ARGN})

  if (SuiteSparse_FIND_${COMPONENT}_REQUIRED)
    list(APPEND SuiteSparse_FOUND_REQUIRED_VARS SuiteSparse_${COMPONENT}_FOUND)
  endif()

  set(SuiteSparse_${COMPONENT}_FOUND TRUE)
  if (SuiteSparse_FIND_${COMPONENT}_FILES)
    find_path(SuiteSparse_${COMPONENT}_INCLUDE_DIR
      NAMES ${SuiteSparse_FIND_${COMPONENT}_FILES}
      HINTS ${SuiteSparse_INCLUDE_DIR_HINTS}
      PATHS ${SuiteSparse_CHECK_INCLUDE_DIRS}
      PATH_SUFFIXES ${SuiteSparse_CHECK_PATH_SUFFIXES})
    if (SuiteSparse_${COMPONENT}_INCLUDE_DIR)
      message(STATUS "Found ${COMPONENT} headers in: "
        "${SuiteSparse_${COMPONENT}_INCLUDE_DIR}")
      mark_as_advanced(SuiteSparse_${COMPONENT}_INCLUDE_DIR)
    else()
      # Specified headers not found.
      set(SuiteSparse_${COMPONENT}_FOUND FALSE)
      if (SuiteSparse_FIND_${COMPONENT}_REQUIRED)
        suitesparse_report_not_found(
          "Did not find ${COMPONENT} header (required SuiteSparse component).")
      else()
        message(STATUS "Did not find ${COMPONENT} header (optional "
          "SuiteSparse component).")
        # Hide optional vars from CMake GUI even if not found.
        mark_as_advanced(SuiteSparse_${COMPONENT}_INCLUDE_DIR)
      endif()
    endif()
  endif()

  if (SuiteSparse_FIND_${COMPONENT}_LIBRARIES)
    find_library(SuiteSparse_${COMPONENT}_LIBRARY
      NAMES ${SuiteSparse_FIND_${COMPONENT}_LIBRARIES}
      HINTS ${SuiteSparse_LIBRARY_DIR_HINTS}
      PATHS ${SuiteSparse_CHECK_LIBRARY_DIRS}
      PATH_SUFFIXES ${SuiteSparse_CHECK_PATH_SUFFIXES})
    if (SuiteSparse_${COMPONENT}_LIBRARY)
      message(STATUS "Found ${COMPONENT} library: ${SuiteSparse_${COMPONENT}_LIBRARY}")
      mark_as_advanced(SuiteSparse_${COMPONENT}_LIBRARY)

      if(NOT TARGET SuiteSparse::${COMPONENT})
        add_library(SuiteSparse::${COMPONENT} IMPORTED UNKNOWN)
      endif()
      set_property(TARGET SuiteSparse::${COMPONENT} PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${SuiteSparse_${COMPONENT}_INCLUDE_DIR})
      set_property(TARGET SuiteSparse::${COMPONENT} PROPERTY
        IMPORTED_LOCATION_RELEASE ${SuiteSparse_${COMPONENT}_LIBRARY})
      set_property(TARGET SuiteSparse::${COMPONENT} APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
    else ()
      # Specified libraries not found.
      set(SuiteSparse_${COMPONENT}_FOUND FALSE)
      if (SuiteSparse_FIND_${COMPONENT}_REQUIRED)
        suitesparse_report_not_found(
          "Did not find ${COMPONENT} library (required SuiteSparse component).")
      else()
        message(STATUS "Did not find ${COMPONENT} library (optional SuiteSparse "
          "dependency)")
        # Hide optional vars from CMake GUI even if not found.
        mark_as_advanced(SuiteSparse_${COMPONENT}_LIBRARY)
      endif()
    endif()
  endif()
endmacro()

# Given the number of components of SuiteSparse, and to ensure that the
# automatic failure message generated by FindPackageHandleStandardArgs()
# when not all required components are found is helpful, we maintain a list
# of all variables that must be defined for SuiteSparse to be considered found.
unset(SuiteSparse_FOUND_REQUIRED_VARS)

# BLAS.
find_package(BLAS QUIET)
if (NOT BLAS_FOUND)
  suitesparse_report_not_found(
    "Did not find BLAS library (required for SuiteSparse).")
endif (NOT BLAS_FOUND)
list(APPEND SuiteSparse_FOUND_REQUIRED_VARS BLAS_FOUND)

# LAPACK.
find_package(LAPACK QUIET)
if (NOT LAPACK_FOUND)
  suitesparse_report_not_found(
    "Did not find LAPACK library (required for SuiteSparse).")
endif (NOT LAPACK_FOUND)
list(APPEND SuiteSparse_FOUND_REQUIRED_VARS LAPACK_FOUND)

foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
  string (TOLOWER ${component} component_lower)

  if (component STREQUAL "SPQR")
    set (component_header SuiteSparseQR.hpp)
  else (component STREQUAL "SPQR")
    set (component_header ${component_lower}.h)
  endif (component STREQUAL "SPQR")

  suitesparse_find_component(${component} REQUIRED FILES ${component_header}
    LIBRARIES ${component_lower})
endforeach (component IN LISTS SuiteSparse_FIND_COMPONENTS)

if (SuiteSparse_SPQR_FOUND)
  # SuiteSparseQR may be compiled with Intel Threading Building Blocks,
  # we assume that if TBB is installed, SuiteSparseQR was compiled with
  # support for it, this will do no harm if it wasn't.
  find_package(TBB QUIET)
  if (TBB_FOUND)
    message(STATUS "Found Intel Thread Building Blocks (TBB) library "
      "(${TBB_VERSION_MAJOR}.${TBB_VERSION_MINOR} / ${TBB_INTERFACE_VERSION}) "
      "include location: ${TBB_INCLUDE_DIRS}. Assuming SuiteSparseQR was "
      "compiled with TBB.")
    # Add the TBB libraries to the SuiteSparseQR libraries (the only
    # libraries to optionally depend on TBB).
    if (TARGET TBB::tbb)
      # Native TBB package configuration provides an imported target. Use it if
      # available.
      set_property (TARGET SuiteSparse::SPQR APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES TBB::tbb)
    else (TARGET TBB::tbb)
      set_property (TARGET SuiteSparse::SPQR APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS})
      set_property (TARGET SuiteSparse::SPQR APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${TBB_LIBRARIES})
    endif (TARGET TBB::tbb)
  else()
    message(STATUS "Did not find Intel TBB library, assuming SuiteSparseQR was "
      "not compiled with TBB.")
  endif()
endif(SuiteSparse_SPQR_FOUND)

# SuiteSparse_config.
#
# If SuiteSparse version is >= 4 then SuiteSparse_config is required.
suitesparse_find_component(
  Config FILES SuiteSparse_config.h LIBRARIES suitesparseconfig)

check_library_exists(rt shm_open "" HAVE_LIBRT)

if (TARGET SuiteSparse::Config)
  # SuiteSparse_config (SuiteSparse version >= 4) requires librt library for
  # timing by default when compiled on Linux or Unix, but not on OSX (which
  # does not have librt).
  if (HAVE_LIBRT)
    message(STATUS "Adding librt to "
      "SuiteSparse_config libraries (required on Linux & Unix [not OSX] if "
      "SuiteSparse is compiled with timing).")
    set_property (TARGET SuiteSparse::Config APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES $<LINK_ONLY:rt>)
  else (HAVE_LIBRT)
    message(STATUS "Could not find librt, but found SuiteSparse_config, "
      "assuming that SuiteSparse was compiled without timing.")
  endif (HAVE_LIBRT)

  if (BLAS_FOUND)
    if (TARGET BLAS::BLAS)
      set_property (TARGET SuiteSparse::Config APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES $<LINK_ONLY:BLAS::BLAS>)
    else (TARGET BLAS::BLAS)
      set_property (TARGET SuiteSparse::Config APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${BLAS_LIBRARIES})
    endif (TARGET BLAS::BLAS)
  endif (BLAS_FOUND)

  if (LAPACK_FOUND)
    if (TARGET LAPACK::LAPACK)
      set_property (TARGET SuiteSparse::Config APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES $<LINK_ONLY:LAPACK::LAPACK>)
    else (TARGET LAPACK::LAPACK)
      set_property (TARGET SuiteSparse::Config APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${LAPACK_LIBRARIES})
    endif (TARGET LAPACK::LAPACK)
  endif (LAPACK_FOUND)

  # SuiteSparse version >= 4.
  set(SuiteSparse_VERSION_FILE
    ${SuiteSparse_Config_INCLUDE_DIR}/SuiteSparse_config.h)
  if (NOT EXISTS ${SuiteSparse_VERSION_FILE})
    suitesparse_report_not_found(
      "Could not find file: ${SuiteSparse_VERSION_FILE} containing version "
      "information for >= v4 SuiteSparse installs, but SuiteSparse_config was "
      "found (only present in >= v4 installs).")
  else (NOT EXISTS ${SuiteSparse_VERSION_FILE})
    file(READ ${SuiteSparse_VERSION_FILE} Config_CONTENTS)

    string(REGEX MATCH "#define SUITESPARSE_MAIN_VERSION [0-9]+"
      SuiteSparse_VERSION_MAJOR "${Config_CONTENTS}")
    string(REGEX REPLACE "#define SUITESPARSE_MAIN_VERSION ([0-9]+)" "\\1"
      SuiteSparse_VERSION_MAJOR "${SuiteSparse_VERSION_MAJOR}")

    string(REGEX MATCH "#define SUITESPARSE_SUB_VERSION [0-9]+"
      SuiteSparse_VERSION_MINOR "${Config_CONTENTS}")
    string(REGEX REPLACE "#define SUITESPARSE_SUB_VERSION ([0-9]+)" "\\1"
      SuiteSparse_VERSION_MINOR "${SuiteSparse_VERSION_MINOR}")

    string(REGEX MATCH "#define SUITESPARSE_SUBSUB_VERSION [0-9]+"
      SuiteSparse_VERSION_PATCH "${Config_CONTENTS}")
    string(REGEX REPLACE "#define SUITESPARSE_SUBSUB_VERSION ([0-9]+)" "\\1"
      SuiteSparse_VERSION_PATCH "${SuiteSparse_VERSION_PATCH}")

    # This is on a single line s/t CMake does not interpret it as a list of
    # elements and insert ';' separators which would result in 4.;2.;1 nonsense.
    set(SuiteSparse_VERSION
      "${SuiteSparse_VERSION_MAJOR}.${SuiteSparse_VERSION_MINOR}.${SuiteSparse_VERSION_PATCH}")
    set(SuiteSparse_VERSION_COMPONENTS 3)
  endif (NOT EXISTS ${SuiteSparse_VERSION_FILE})
endif (TARGET SuiteSparse::Config)

# METIS (Optional dependency).
find_package (METIS)

# CHOLMOD requires AMD CAMD CCOLAMD COLAMD
if (TARGET SuiteSparse::CHOLMOD)
  # METIS is optional
  if (TARGET METIS::METIS)
    set_property (TARGET SuiteSparse::CHOLMOD APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES METIS::METIS)
  endif (TARGET METIS::METIS)

  foreach (component IN ITEMS AMD CAMD CCOLAMD COLAMD)
    if (TARGET SuiteSparse::${component})
      set_property (TARGET SuiteSparse::CHOLMOD APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES SuiteSparse::${component})
    else (TARGET SuiteSparse::${component})
      # Consider CHOLMOD not found if COLAMD cannot be found
      set (SuiteSparse_CHOLMOD_FOUND FALSE)
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
    set (SuiteSparse_SQPR_FOUND FALSE)
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

suitesparse_reset_find_library_prefix()

# Handle REQUIRED and QUIET arguments to FIND_PACKAGE
include(FindPackageHandleStandardArgs)
if (SuiteSparse_FOUND)
  find_package_handle_standard_args(SuiteSparse
    REQUIRED_VARS ${SuiteSparse_FOUND_REQUIRED_VARS}
    VERSION_VAR SuiteSparse_VERSION
    FAIL_MESSAGE "Failed to find some/all required components of SuiteSparse."
    HANDLE_COMPONENTS)
else (SuiteSparse_FOUND)
  # Do not pass VERSION_VAR to FindPackageHandleStandardArgs() if we failed to
  # find SuiteSparse to avoid a confusing autogenerated failure message
  # that states 'not found (missing: FOO) (found version: x.y.z)'.
  find_package_handle_standard_args(SuiteSparse
    REQUIRED_VARS ${SuiteSparse_FOUND_REQUIRED_VARS}
    FAIL_MESSAGE "Failed to find some/all required components of SuiteSparse."
    HANDLE_COMPONENTS)
endif (SuiteSparse_FOUND)
