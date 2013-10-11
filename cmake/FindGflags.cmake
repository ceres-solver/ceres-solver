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

# FindGflags.cmake - Find Google gflags logging library.
#
# This module defines the following variables:
#
# GFLAGS_FOUND: TRUE iff gflags is found.
# GFLAGS_INCLUDE_DIRS: Include directories for gflags.
# GFLAGS_LIBRARIES: Libraries required to link gflags.
#
# The following variables control the behaviour of this module:
#
# GFLAGS_INCLUDE_DIR_HINTS: List of additional directories in which to
#                           search for gflags includes, e.g: /timbuktu/include.
# GFLAGS_LIBRARY_DIR_HINTS: List of additional directories in which to
#                           search for gflags libraries, e.g: /timbuktu/lib.

# TODO: Add standard Windows search locations for gflags.
LIST(APPEND GFLAGS_CHECK_INCLUDE_DIRS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include)
LIST(APPEND GFLAGS_CHECK_LIBRARY_DIRS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib)

FIND_PATH(GFLAGS_INCLUDE_DIR
  NAMES gflags/gflags.h
  PATHS ${GFLAGS_CHECK_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_HINTS})
FIND_LIBRARY(GFLAGS_LIBRARY NAMES gflags
  PATHS ${GFLAGS_CHECK_LIBRARY_DIRS}
  ${GFLAGS_LIBRARY_HINTS})

# Only mark include directory & library as advanced if we found them, otherwise
# leave it in the standard GUI for the user to set manually.
IF (GFLAGS_INCLUDE_DIR)
  MARK_AS_ADVANCED(GFLAGS_INCLUDE_DIR)
ENDIF (GFLAGS_INCLUDE_DIR)
IF (GFLAGS_LIBRARY)
  MARK_AS_ADVANCED(GFLAGS_LIBRARY)
ENDIF (GFLAGS_LIBRARY)

# gflags does not seem to provide any record of the version in its
# source tree, thus cannot extract version.

# Set standard CMake FindPackage variables.
SET(GFLAGS_INCLUDE_DIRS ${GFLAGS_INCLUDE_DIR})
SET(GFLAGS_LIBRARIES ${GFLAGS_LIBRARY})

# Handle REQUIRED / QUIET optional arguments.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Gflags DEFAULT_MSG
  GFLAGS_INCLUDE_DIRS GFLAGS_LIBRARIES)
