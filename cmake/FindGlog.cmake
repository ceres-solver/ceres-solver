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

# FindGlog.cmake - Find Google glog logging library.
#
# This module defines the following variables:
#
# GLOG_FOUND: TRUE iff glog is found.
# GLOG_INCLUDE_DIRS: Include directories for glog.
# GLOG_LIBRARIES: Libraries required to link glog.
#
# The following variables control the behaviour of this module:
#
# GLOG_INCLUDE_DIRS_HINTS: List of additional directories in which to
#                          search for glog includes, e.g: /timbuktu/include.
# GLOG_LIBRARY_DIRS_HINTS: List of additional directories in which to
#                          search for glog libraries, e.g: /timbuktu/lib.

# TODO: Add standard Windows search locations for glog.
LIST(APPEND GLOG_CHECK_INCLUDE_DIRS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include)
LIST(APPEND GLOG_CHECK_LIBRARY_DIRS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib)

FIND_PATH(GLOG_INCLUDE_DIR
  NAMES glog/logging.h
  PATHS ${GLOG_CHECK_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIR_HINTS})
FIND_LIBRARY(GLOG_LIBRARY NAMES glog
  PATHS ${GLOG_CHECK_LIBRARY_DIRS}
  ${GLOG_LIBRARY_DIR_HINTS})

# Only mark include directory & library as advanced if we found them, otherwise
# leave it in the standard GUI for the user to set manually.
IF (GLOG_INCLUDE_DIR)
  MARK_AS_ADVANCED(GLOG_INCLUDE_DIR)
ENDIF (GLOG_INCLUDE_DIR)
IF (GLOG_LIBRARY)
  MARK_AS_ADVANCED(GLOG_LIBRARY)
ENDIF (GLOG_LIBRARY)

# Glog does not seem to provide any record of the version in its
# source tree, thus cannot extract version.

# Set standard CMake FindPackage variables.
SET(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
SET(GLOG_LIBRARIES ${GLOG_LIBRARY})

# Handle REQUIRED / QUIET optional arguments.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Glog DEFAULT_MSG
  GLOG_INCLUDE_DIRS GLOG_LIBRARIES)
