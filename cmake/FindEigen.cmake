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

# FindEigen.cmake - Find Eigen library, version >= 3.
#
# This module defines the following variables:
#
# EIGEN_FOUND: TRUE iff Eigen is found.
# EIGEN_INCLUDE_DIRS: Include directories for Eigen.
#
# EIGEN_VERSION: Extracted from Eigen/src/Core/util/Macros.h
# EIGEN_WORLD_VERSION: Equal to 3 if EIGEN_VERSION = 3.2.0
# EIGEN_MAJOR_VERSION: Equal to 2 if EIGEN_VERSION = 3.2.0
# EIGEN_MINOR_VERSION: Equal to 0 if EIGEN_VERSION = 3.2.0
#
# The following variables control the behaviour of this module:
#
# EIGEN_INCLUDE_DIR_HINTS: List of additional directories in which to
#                          search for eigen includes, e.g: /timbuktu/eigen3.

# TODO: Add standard Windows search locations for glog.
LIST(APPEND EIGEN_CHECK_INCLUDE_DIRS
  /usr/include/eigen3
  /usr/local/include/eigen3
  /usr/local/homebrew/include/eigen3 # Mac OS X
  /opt/local/var/macports/software/eigen3 # Mac OS X.
  /opt/local/include/eigen3)

# TODO: Add standard Windows search locations for Eigen.
FIND_PATH(EIGEN_INCLUDE_DIR
  NAMES Eigen/Core
  PATHS ${EIGEN_CHECK_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR_HINTS})

# Only mark include directory as advanced if we found it, otherwise
# leave it in the standard GUI for the user to set manually.
IF (EIGEN_INCLUDE_DIR)
  MARK_AS_ADVANCED(EIGEN_INCLUDE_DIR)
ENDIF (EIGEN_INCLUDE_DIR)

# Extract Eigen version from Eigen/src/Core/util/Macros.h
IF (EIGEN_INCLUDE_DIR)
  SET(EIGEN_VERSION_FILE ${EIGEN_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h)
  IF (NOT EXISTS ${EIGEN_VERSION_FILE})
    MESSAGE(FATAL_ERROR "Failed to find Eigen - Could not find file: "
      "${EIGEN_VERSION_FILE} containing version information in Eigen "
      "install located at: ${EIGEN_INCLUDE_DIR}.")
  ENDIF (NOT EXISTS ${EIGEN_VERSION_FILE})
  FILE(READ ${EIGEN_VERSION_FILE} EIGEN_VERSION_FILE_CONTENTS)

  STRING(REGEX MATCH "#define EIGEN_WORLD_VERSION [0-9]+"
    EIGEN_WORLD_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define EIGEN_WORLD_VERSION ([0-9]+)" "\\1"
    EIGEN_WORLD_VERSION "${EIGEN_WORLD_VERSION}")

  STRING(REGEX MATCH "#define EIGEN_MAJOR_VERSION [0-9]+"
    EIGEN_MAJOR_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define EIGEN_MAJOR_VERSION ([0-9]+)" "\\1"
    EIGEN_MAJOR_VERSION "${EIGEN_MAJOR_VERSION}")

  STRING(REGEX MATCH "#define EIGEN_MINOR_VERSION [0-9]+"
    EIGEN_MINOR_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
  STRING(REGEX REPLACE "#define EIGEN_MINOR_VERSION ([0-9]+)" "\\1"
    EIGEN_MINOR_VERSION "${EIGEN_MINOR_VERSION}")

  # This is on a single line s/t CMake does not interpret it as a list of
  # elements and insert ';' separators which would result in 3.;2.;0 nonsense.
  SET(EIGEN_VERSION "${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
ENDIF (EIGEN_INCLUDE_DIR)

# Set standard CMake FindPackage variables.
SET(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})

# Handle REQUIRED / QUIET optional arguments and version.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Eigen
  REQUIRED_VARS EIGEN_INCLUDE_DIRS
  VERSION_VAR EIGEN_VERSION)
