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
FindCXSparse
============

Find CXSparse and its dependencies.

This module defines the following variables which should be referenced by the
caller to use the library.

``CXSparse_FOUND``
    ``TRUE`` iff CXSparse and all dependencies have been found.

``CXSparse_VERSION``
    Extracted from ``cs.h``.

``CXSparse_VERSION_MAJOR``
    Equal to 3 if ``CXSparse_VERSION`` = 3.1.2

``CXSparse_VERSION_MINOR``
    Equal to 1 if ``CXSparse_VERSION`` = 3.1.2

``CXSparse_VERSION_PATCH``
    Equal to 2 if ``CXSparse_VERSION`` = 3.1.2

The following variables control the behaviour of this module:

``CXSparse_NO_CMAKE``
  Do not attempt to use the native CXSparse CMake package configuration.

Targets
-------

The following target defines CXSparse.

``CXSparse::CXSparse``
    The main CXSparse to be linked against.

The following variables are also defined by this module, but in line with CMake
recommended ``find_package`` module style should NOT be referenced directly by
callers (use the plural variables detailed above instead).  These variables do
however affect the behaviour of the module via ``find_[path/library]()`` which
are NOT re-called (i.e., search for library is not repeated) if these variables
are set with valid values *in the CMake cache*. This means that if these
variables are set directly in the cache, either by the user in the CMake GUI, or
by the user passing ``-DVAR=VALUE`` directives to CMake when called (which
explicitly defines a cache variable), then they will be used verbatim, bypassing
the ``HINTS`` variables and other hard-coded search locations.

``CXSparse_INCLUDE_DIR``
    Include directory for CXSparse, not including the include directory of any
    dependencies.

``CXSparse_LIBRARY``
    CXSparse library, not including the libraries of any dependencies.
]=======================================================================]

if (NOT CXSparse_NO_CMAKE)
  find_package (CXSparse NO_MODULE QUIET)
endif (NOT CXSparse_NO_CMAKE)

if (CXSparse_FOUND)
  return ()
endif (CXSparse_FOUND)

# Reset CALLERS_CMAKE_FIND_LIBRARY_PREFIXES to its value when
# FindCXSparse was invoked.
macro(CXSparse_RESET_FIND_LIBRARY_PREFIX)
  if (MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CALLERS_CMAKE_FIND_LIBRARY_PREFIXES}")
  endif (MSVC)
endmacro(CXSparse_RESET_FIND_LIBRARY_PREFIX)

# Called if we failed to find CXSparse or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(CXSparse_REPORT_NOT_FOUND REASON_MSG)
  # Make results of search visible in the CMake GUI if CXSparse has not
  # been found so that user does not have to toggle to advanced view.
  mark_as_advanced(CLEAR CXSparse_INCLUDE_DIR
                         CXSparse_LIBRARY)

  cxsparse_reset_find_library_prefix()

  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  if (CXSparse_FIND_QUIETLY)
    message(STATUS "Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
  elseif (CXSparse_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    message("-- Failed to find CXSparse - " ${REASON_MSG} ${ARGN})
  endif ()
  return()
endmacro(CXSparse_REPORT_NOT_FOUND)

# Handle possible presence of lib prefix for libraries on MSVC, see
# also CXSparse_RESET_FIND_LIBRARY_PREFIX().
if (MSVC)
  # Preserve the caller's original values for CMAKE_FIND_LIBRARY_PREFIXES
  # s/t we can set it back before returning.
  set(CALLERS_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  # The empty string in this list is important, it represents the case when
  # the libraries have no prefix (shared libraries / DLLs).
  set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "" "${CMAKE_FIND_LIBRARY_PREFIXES}")
endif (MSVC)

# Additional suffixes to try appending to each search path.
list(APPEND CXSparse_CHECK_PATH_SUFFIXES
  suitesparse) # Linux/Windows

# Search supplied hint directories first if supplied.
find_path(CXSparse_INCLUDE_DIR
  NAMES cs.h
  PATH_SUFFIXES ${CXSparse_CHECK_PATH_SUFFIXES})
if (NOT CXSparse_INCLUDE_DIR OR
    NOT EXISTS ${CXSparse_INCLUDE_DIR})
  cxsparse_report_not_found(
    "Could not find CXSparse include directory, set CXSparse_INCLUDE_DIR "
    "to directory containing cs.h")
endif (NOT CXSparse_INCLUDE_DIR OR
       NOT EXISTS ${CXSparse_INCLUDE_DIR})

find_library(CXSparse_LIBRARY NAMES cxsparse
  PATH_SUFFIXES ${CXSparse_CHECK_PATH_SUFFIXES})

if (NOT CXSparse_LIBRARY OR
    NOT EXISTS ${CXSparse_LIBRARY})
  cxsparse_report_not_found(
    "Could not find CXSparse library, set CXSparse_LIBRARY "
    "to full path to libcxsparse.")
endif (NOT CXSparse_LIBRARY OR
       NOT EXISTS ${CXSparse_LIBRARY})

# Mark internally as found, then verify. CXSparse_REPORT_NOT_FOUND() unsets
# if called.
set(CXSparse_FOUND TRUE)

# Extract CXSparse version from cs.h
if (CXSparse_INCLUDE_DIR)
  set(CXSparse_VERSION_FILE ${CXSparse_INCLUDE_DIR}/cs.h)
  if (NOT EXISTS ${CXSparse_VERSION_FILE})
    cxsparse_report_not_found(
      "Could not find file: ${CXSparse_VERSION_FILE} "
      "containing version information in CXSparse install located at: "
      "${CXSparse_INCLUDE_DIR}.")
  else (NOT EXISTS ${CXSparse_VERSION_FILE})
    file(READ ${CXSparse_INCLUDE_DIR}/cs.h CXSparse_VERSION_FILE_CONTENTS)

    string(REGEX MATCH "#define CS_VER [0-9]+"
      CXSparse_VERSION_MAJOR "${CXSparse_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define CS_VER ([0-9]+)" "\\1"
      CXSparse_VERSION_MAJOR "${CXSparse_VERSION_MAJOR}")

    string(REGEX MATCH "#define CS_SUBVER [0-9]+"
      CXSparse_VERSION_MINOR "${CXSparse_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define CS_SUBVER ([0-9]+)" "\\1"
      CXSparse_VERSION_MINOR "${CXSparse_VERSION_MINOR}")

    string(REGEX MATCH "#define CS_SUBSUB [0-9]+"
      CXSparse_VERSION_PATCH "${CXSparse_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "#define CS_SUBSUB ([0-9]+)" "\\1"
      CXSparse_VERSION_PATCH "${CXSparse_VERSION_PATCH}")

    # This is on a single line s/t CMake does not interpret it as a list of
    # elements and insert ';' separators which would result in 3.;1.;2 nonsense.
    set(CXSparse_VERSION "${CXSparse_VERSION_MAJOR}.${CXSparse_VERSION_MINOR}.${CXSparse_VERSION_PATCH}")
    set(CXSparse_VERSION_COMPONENTS 3)
  endif (NOT EXISTS ${CXSparse_VERSION_FILE})
endif (CXSparse_INCLUDE_DIR)

# Catch the case when the caller has set CXSparse_LIBRARY in the cache / GUI and
# thus FIND_LIBRARY was not called, but specified library is invalid, otherwise
# we would report CXSparse as found.
# TODO: This regex for CXSparse library is pretty primitive, we use lowercase
#       for comparison to handle Windows using CamelCase library names, could
#       this check be better?
string(TOLOWER "${CXSparse_LIBRARY}" LOWERCASE_CXSparse_LIBRARY)
if (CXSparse_LIBRARY AND
    EXISTS ${CXSparse_LIBRARY} AND
    NOT "${LOWERCASE_CXSparse_LIBRARY}" MATCHES ".*cxsparse[^/]*")
  cxsparse_report_not_found(
    "Caller defined CXSparse_LIBRARY: "
    "${CXSparse_LIBRARY} does not match CXSparse.")
endif (CXSparse_LIBRARY AND
       EXISTS ${CXSparse_LIBRARY} AND
       NOT "${LOWERCASE_CXSparse_LIBRARY}" MATCHES ".*cxsparse[^/]*")

cxsparse_reset_find_library_prefix()

mark_as_advanced(CXSparse_INCLUDE_DIR CXSparse_LIBRARY)

# Handle REQUIRED / QUIET optional arguments and version.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CXSparse
  REQUIRED_VARS CXSparse_INCLUDE_DIR CXSparse_LIBRARY
  VERSION_VAR CXSparse_VERSION)

if (CXSparse_INCLUDE_DIR AND CXSparse_LIBRARY)
  if (NOT TARGET CXSparse::CXSparse)
    add_library (CXSparse::CXSparse IMPORTED UNKNOWN)
  endif (NOT TARGET CXSparse::CXSparse)

  set_property (TARGET CXSparse::CXSparse PROPERTY
    IMPORTED_LOCATION ${CXSparse_LIBRARY})
  set_property (TARGET CXSparse::CXSparse PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${CXSparse_INCLUDE_DIR})
endif (CXSparse_INCLUDE_DIR AND CXSparse_LIBRARY)
