# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2013 Google Inc. All rights reserved.
# http://code.google.com/p/ceres-solver/
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

# FindCXSparse.cmake - Find CXSparse libraries & dependencies.
#
# This module defines the following variables:
#
# CXSPARSE_FOUND: TRUE iff CXSparse and all dependencies have been found.
# CXSPARSE_INCLUDE_DIRS: Include directories for CXSparse.
# CXSPARSE_LIBRARIES: Libraries for CXSparse and all dependencies.
#
# CXSPARSE_VERSION: Extracted from cs.h.
# CXSPARSE_MAIN_VERSION: Equal to 3 if CXSPARSE_VERSION = 3.1.2
# CXSPARSE_SUB_VERSION: Equal to 1 if CXSPARSE_VERSION = 3.1.2
# CXSPARSE_SUBSUB_VERSION: Equal to 2 if CXSPARSE_VERSION = 3.1.2
# 
# The following variables control the behaviour of this module:
#
# CXSPARSE_INCLUDE_DIR_HINTS: List of additional directories in which to
#                             search for CXSparse includes,
#                             e.g: /timbuktu/include.
# CXSPARSE_LIBRARY_DIR_HINTS: List of additional directories in which to
#                             search for CXSparse libraries, e.g: /timbuktu/lib.

# TODO: Add standard Windows search locations for CXSparse.
LIST(APPEND CXSPARSE_CHECK_INCLUDE_DIRS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include)
LIST(APPEND CXSPARSE_CHECK_LIBRARY_DIRS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib)

FIND_PATH(CXSPARSE_INCLUDE_DIR
  NAMES cs.h
  PATHS ${CXSPARSE_CHECK_INCLUDE_DIRS}
  ${CXSPARSE_INCLUDE_DIR_HINTS})
FIND_LIBRARY(CXSPARSE_LIBRARY NAMES cxsparse
  PATHS ${CXSPARSE_CHECK_LIBRARY_DIRS}
  ${CXSPARSE_LIBRARY_DIR_HINTS})

# Only mark include directory & library as advanced if we found them, otherwise
# leave it in the standard GUI for the user to set manually.
IF (CXSPARSE_INCLUDE_DIR)
  MARK_AS_ADVANCED(CXSPARSE_INCLUDE_DIR)
ENDIF (CXSPARSE_INCLUDE_DIR)
IF (CXSPARSE_LIBRARY)
  MARK_AS_ADVANCED(CXSPARSE_LIBRARY)
ENDIF (CXSPARSE_LIBRARY)

# Extract CXSparse version from cs.h
IF (CXSPARSE_INCLUDE_DIR)
  FILE(READ ${CXSPARSE_INCLUDE_DIR}/cs.h CXSPARSE_VERSION_FILE_CONTENTS)

  STRING(REGEX MATCH "#define CS_VER [0-9]+"
    CXSPARSE_MAIN_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define CS_VER ([0-9]+)" "\\1"
    CXSPARSE_MAIN_VERSION "${CXSPARSE_MAIN_VERSION}")

  STRING(REGEX MATCH "#define CS_SUBVER [0-9]+"
    CXSPARSE_SUB_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define CS_SUBVER ([0-9]+)" "\\1"
    CXSPARSE_SUB_VERSION "${CXSPARSE_SUB_VERSION}")

  STRING(REGEX MATCH "#define CS_SUBSUB [0-9]+"
    CXSPARSE_SUBSUB_VERSION "${CXSPARSE_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define CS_SUBSUB ([0-9]+)" "\\1"
    CXSPARSE_SUBSUB_VERSION "${CXSPARSE_SUBSUB_VERSION}")

  # This is on a single line s/t CMake does not interpret it as a list of
  # elements and insert ';' separators which would result in 3.;1.;2 nonsense.
  SET(CXSPARSE_VERSION "${CXSPARSE_MAIN_VERSION}.${CXSPARSE_SUB_VERSION}.${CXSPARSE_SUBSUB_VERSION}")
ENDIF (CXSPARSE_INCLUDE_DIR)

# Set standard CMake FindPackage variables.
SET(CXSPARSE_INCLUDE_DIRS ${CXSPARSE_INCLUDE_DIR})
SET(CXSPARSE_LIBRARIES ${CXSPARSE_LIBRARY})

# Handle REQUIRED / QUIET optional arguments and version.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CXSparse
  REQUIRED_VARS CXSPARSE_INCLUDE_DIRS CXSPARSE_LIBRARIES
  VERSION_VAR CXSPARSE_VERSION)
