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

if (NOT SuiteSparse_NO_CMAKE)
  find_package (SuiteSparse NO_MODULE QUIET)
  if (SuiteSparse_FOUND)
    # Report the main include directory instead of the package configuration
    # file path in FindPackageHandleStandardArgs' standard success message.
    get_target_property(SuiteSparse_INCLUDE_DIR SuiteSparse::Config
      INTERFACE_INCLUDE_DIRECTORIES)
    if (SuiteSparse_INCLUDE_DIR)
      list(GET SuiteSparse_INCLUDE_DIR -1 SuiteSparse_INCLUDE_DIR)
    endif ()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(SuiteSparse
      REQUIRED_VARS SuiteSparse_INCLUDE_DIR
      VERSION_VAR SuiteSparse_VERSION
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
  if (NOT DEFINED METIS_FOUND)
    find_package (METIS)
  endif()

  if (TARGET METIS::METIS)
    cmake_push_check_state (RESET)
    set (CMAKE_REQUIRED_LIBRARIES SuiteSparse::CHOLMOD METIS::METIS)
    check_symbol_exists (cholmod_metis cholmod.h SuiteSparse_CHOLMOD_USES_METIS)
    cmake_pop_check_state ()

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
