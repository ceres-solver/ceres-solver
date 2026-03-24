# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2023 Google Inc. All rights reserved.
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

This module requires SuiteSparse version 7.0 or later, which provides
native CMake configuration files.

This module defines the following variables:

``SuiteSparse_FOUND``
   ``TRUE`` iff SuiteSparse and all required components have been found.

``SuiteSparse_VERSION``
   Extracted from the SuiteSparse configuration or headers.

Targets
-------

The following targets define the SuiteSparse components searched for.

``SuiteSparse::AMD``
    Symmetric Approximate Minimum Degree (AMD)

``SuiteSparse::CAMD``
    Constrained Approximate Minimum Degree (AMD)

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
    Common configuration (SuiteSparse_config).

Optional SuiteSparse dependencies:

``METIS::METIS``
    Serial Graph Partitioning and Fill-reducing Matrix Ordering (METIS)
]=======================================================================]

# On macOS, SuiteSparse often depends on OpenMP which might not be found
# automatically by AppleClang.
if (APPLE)
  find_package(OpenMP QUIET)
endif()

# List of components SuiteSparse 7.0+ exports as individual CMake packages.
set(SuiteSparse_EXPORTED_COMPONENTS
  AMD
  BTF
  CAMD
  CCOLAMD
  CHOLMOD
  COLAMD
  CXSparse
  KLU
  LDL
  RBio
  SPQR
  SuiteSparse_config
  UMFPACK
)

# Initialize component found variables.
if (SuiteSparse_FIND_COMPONENTS)
  foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
    set(SuiteSparse_${component}_FOUND FALSE)
  endforeach()
endif()

# We always need SuiteSparse_config for versioning and as a base dependency.
set(SuiteSparse_config_FOUND FALSE)

# 1. Try finding SuiteSparse using CONFIG mode (preferred for 7.0+).
# Some installations might provide a top-level SuiteSparseConfig.cmake.
find_package(SuiteSparse 7.0 CONFIG QUIET)

if (NOT SuiteSparse_FOUND)
  # Try finding individual components using CONFIG mode.
  foreach (component IN LISTS SuiteSparse_EXPORTED_COMPONENTS)
    find_package(${component} CONFIG QUIET)
    if (${component}_FOUND)
      # Map to our namespace if not already done by the config file.
      if (NOT TARGET SuiteSparse::${component})
        # Some components might export SuiteSparse::<component> directly.
        # If they export <component> or <component>_static, we wrap them.
        if (TARGET ${component})
          add_library(SuiteSparse::${component} INTERFACE IMPORTED)
          set_target_properties(SuiteSparse::${component} PROPERTIES
            INTERFACE_LINK_LIBRARIES ${component})
        elseif (TARGET ${component}_static)
           add_library(SuiteSparse::${component} INTERFACE IMPORTED)
           set_target_properties(SuiteSparse::${component} PROPERTIES
             INTERFACE_LINK_LIBRARIES ${component}_static)
        elseif (TARGET SuiteSparse::${component}_static)
           add_library(SuiteSparse::${component} INTERFACE IMPORTED)
           set_target_properties(SuiteSparse::${component} PROPERTIES
             INTERFACE_LINK_LIBRARIES SuiteSparse::${component}_static)
        endif()
      endif()

      # Special handling for SuiteSparse_config -> SuiteSparse::Config
      if (component STREQUAL "SuiteSparse_config")
        set(SuiteSparse_config_FOUND TRUE)
        if (TARGET SuiteSparse::SuiteSparse_config AND NOT TARGET SuiteSparse::Config)
          add_library(SuiteSparse::Config INTERFACE IMPORTED)
          set_target_properties(SuiteSparse::Config PROPERTIES
            INTERFACE_LINK_LIBRARIES SuiteSparse::SuiteSparse_config)
        endif()
        if (NOT SuiteSparse_VERSION)
          set(SuiteSparse_VERSION ${SuiteSparse_config_VERSION})
        endif()
      endif()
      
      # Mark requested components as found.
      if (SuiteSparse_FIND_COMPONENTS)
        list(FIND SuiteSparse_FIND_COMPONENTS ${component} _index)
        if (_index GREATER -1)
          set(SuiteSparse_${component}_FOUND TRUE)
        endif()
      endif()
    endif()
  endforeach()
endif()

# 2. Manual fallback if CONFIG mode failed to find required components.
# This is useful if dependencies (like OpenMP) are missing from the CONFIG search
# or if the installation doesn't provide CONFIG files.
if (NOT SuiteSparse_config_FOUND)
  find_path(SuiteSparse_config_INCLUDE_DIR NAMES SuiteSparse_config.h
    PATH_SUFFIXES suitesparse include/suitesparse .)
  find_library(SuiteSparse_config_LIBRARY NAMES suitesparseconfig)

  if (SuiteSparse_config_INCLUDE_DIR AND SuiteSparse_config_LIBRARY)
    set(SuiteSparse_config_FOUND TRUE)
    if (NOT TARGET SuiteSparse::Config)
      add_library(SuiteSparse::Config UNKNOWN IMPORTED)
      set_target_properties(SuiteSparse::Config PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparse_config_INCLUDE_DIR}"
        IMPORTED_LOCATION "${SuiteSparse_config_LIBRARY}")
    endif()

    # If we found config, try finding other requested components manually.
    if (SuiteSparse_FIND_COMPONENTS)
      foreach (component IN LISTS SuiteSparse_FIND_COMPONENTS)
        if (NOT SuiteSparse_${component}_FOUND AND NOT component STREQUAL "Partition")
          string(TOLOWER ${component} component_lower)
          set(header_names ${component_lower}.h)
          # Special header names for some components
          if (component STREQUAL "SPQR")
            set(header_names SuiteSparseQR.hpp SuiteSparseQR_C.h)
          endif()
          
          find_path(SuiteSparse_${component}_INCLUDE_DIR NAMES ${header_names}
            PATH_SUFFIXES suitesparse include/suitesparse .)
          
          # Note: SPQR library is often named libspqr, but some might name it libsuitesparseqr.
          if (component STREQUAL "SPQR")
            find_library(SuiteSparse_${component}_LIBRARY NAMES spqr suitesparseqr)
          else()
            find_library(SuiteSparse_${component}_LIBRARY NAMES ${component_lower})
          endif()

          if (SuiteSparse_${component}_INCLUDE_DIR AND SuiteSparse_${component}_LIBRARY)
            set(SuiteSparse_${component}_FOUND TRUE)
            if (NOT TARGET SuiteSparse::${component})
              add_library(SuiteSparse::${component} UNKNOWN IMPORTED)
              set_target_properties(SuiteSparse::${component} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparse_${component}_INCLUDE_DIR}"
                IMPORTED_LOCATION "${SuiteSparse_${component}_LIBRARY}"
                INTERFACE_LINK_LIBRARIES SuiteSparse::Config)
            endif()
          endif()
        endif()
      endforeach()
    endif()
  endif()
endif()

# Extract version from SuiteSparse_config.h if possible.
if (SuiteSparse_config_FOUND AND NOT SuiteSparse_VERSION)
  if (TARGET SuiteSparse::Config)
    get_target_property(_inc_dirs SuiteSparse::Config INTERFACE_INCLUDE_DIRECTORIES)
    # Also check IMPORTED_LOCATION for library dir, maybe headers are nearby
    get_target_property(_lib_file SuiteSparse::Config IMPORTED_LOCATION)
    if (_lib_file)
      get_filename_component(_lib_dir "${_lib_file}" DIRECTORY)
      list(APPEND _inc_dirs "${_lib_dir}/../include" "${_lib_dir}/../include/suitesparse")
    endif()
    
    foreach (_dir IN LISTS _inc_dirs)
      if (EXISTS "${_dir}/SuiteSparse_config.h")
        file(READ "${_dir}/SuiteSparse_config.h" _config_content)
        string(REGEX MATCH "#define SUITESPARSE_MAIN_VERSION[ \t]+([0-9]+)" _major_match "${_config_content}")
        set(_major ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define SUITESPARSE_SUB_VERSION[ \t]+([0-9]+)" _minor_match "${_config_content}")
        set(_minor ${CMAKE_MATCH_1})
        string(REGEX MATCH "#define SUITESPARSE_SUBSUB_VERSION[ \t]+([0-9]+)" _patch_match "${_config_content}")
        set(_patch ${CMAKE_MATCH_1})
        if (_major)
          set(SuiteSparse_VERSION "${_major}.${_minor}.${_patch}")
        endif()
        break()
      endif()
    endforeach()
  endif()
endif()

# 3. Handle Partition component (CHOLMOD with METIS support).
if (SuiteSparse_CHOLMOD_FOUND OR TARGET SuiteSparse::CHOLMOD)
  set(SuiteSparse_Partition_FOUND FALSE)
  # Check if CHOLMOD target links against METIS.
  get_target_property(_cholmod_deps SuiteSparse::CHOLMOD INTERFACE_LINK_LIBRARIES)
  if (_cholmod_deps)
    if (_cholmod_deps MATCHES "METIS" OR _cholmod_deps MATCHES "metis")
      set(SuiteSparse_Partition_FOUND TRUE)
    endif()
  endif()
  
  if (NOT SuiteSparse_Partition_FOUND)
     # Some installations might have METIS linked but not explicitly in INTERFACE_LINK_LIBRARIES
     # or it might be in a different property.
     get_target_property(_cholmod_loc SuiteSparse::CHOLMOD IMPORTED_LOCATION)
     if (_cholmod_loc)
       # We could try to use 'nm' or 'objdump' but that's overkill.
       # Let's check if METIS is found and maybe it's just not linked.
     endif()
  endif()

  if (SuiteSparse_Partition_FOUND)
    if (NOT TARGET SuiteSparse::Partition)
      add_library(SuiteSparse::Partition INTERFACE IMPORTED)
      set_target_properties(SuiteSparse::Partition PROPERTIES
        INTERFACE_LINK_LIBRARIES SuiteSparse::CHOLMOD)
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuiteSparse
  REQUIRED_VARS SuiteSparse_config_FOUND
  VERSION_VAR SuiteSparse_VERSION
  HANDLE_COMPONENTS)


