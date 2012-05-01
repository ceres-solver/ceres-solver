# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
# Author: keir@google.com (Keir Mierle)


CMAKE_MINIMUM_REQUIRED(VERSION 2.2)

IF (COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

PROJECT(CERES C CXX)

ENABLE_TESTING()

# SuiteSparse
OPTION(SUITESPARSE
       "Enable SuiteSparse. Needed for efficient solutions of large problems."
       ON)

# To get a more static build, try the following line on Mac and Linux:
# SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

IF (SUITESPARSE)
  SET(SUITESPARSE_SEARCH_LIBS
      ${SEARCH_LIBS}
      /usr/lib
      /usr/lib/suitesparse             # Ubuntu
      /usr/local/lib/
      /usr/local/lib/suitesparse
      /opt/local/lib/ufsparse          # Mac OS X
      )

  SET(SUITESPARSE_SEARCH_HEADERS
      ${SEARCH_HEADERS}
      /usr/include
      /usr/include/suitesparse         # Ubuntu
      /usr/local/include
      /usr/local/include/suitesparse,
      /opt/local/include/ufsparse      # Mac os X
      )

  FIND_LIBRARY(AMD_LIB NAMES amd PATHS ${SUITESPARSE_SEARCH_LIBS})
  FIND_PATH(AMD_INCLUDE NAMES amd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
  IF (NOT EXISTS ${AMD_LIB} OR NOT EXISTS ${AMD_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find AMD (part of suitesparse.) "
            "Please specify -DAMD_INCLUDE=... and -DAMD_LIB=...")
  ENDIF (NOT EXISTS ${AMD_LIB} OR NOT EXISTS ${AMD_INCLUDE})
  MESSAGE("-- Found AMD library: ${AMD_LIB}")
  MESSAGE("-- Found AMD header in: ${AMD_INCLUDE}")

  MESSAGE("-- Check for CAMD")
  FIND_LIBRARY(CAMD_LIB NAMES camd PATHS ${SUITESPARSE_SEARCH_LIBS})
  FIND_PATH(CAMD_INCLUDE NAMES camd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
  IF (NOT EXISTS ${CAMD_LIB} OR NOT EXISTS ${CAMD_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find CAMD (part of suitesparse.) "
            "Please specify -DCAMD_INCLUDE=... and -DCAMD_LIB=...")
  ENDIF (NOT EXISTS ${CAMD_LIB} OR NOT EXISTS ${CAMD_INCLUDE})
  MESSAGE("-- Found CAMD library: ${CAMD_LIB}")
  MESSAGE("-- Found CAMD header in: ${CAMD_INCLUDE}")

  # Note: Even though Ceres doesn't directly depend on COLAMD,
  # some symbols from COLAMD that are needed for linking.
  MESSAGE("-- Check for COLAMD")
  FIND_LIBRARY(COLAMD_LIB NAMES colamd PATHS ${SUITESPARSE_SEARCH_LIBS})
  FIND_PATH(COLAMD_INCLUDE NAMES colamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
  IF (NOT EXISTS ${COLAMD_LIB} OR NOT EXISTS ${COLAMD_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find COLAMD (part of suitesparse.)"
            "Please specify -DCOLAMD_INCLUDE=... and -DCOLAMD_LIB=...")
  ENDIF (NOT EXISTS ${COLAMD_LIB} OR NOT EXISTS ${COLAMD_INCLUDE})
  MESSAGE("-- Found COLAMD library: ${COLAMD_LIB}")
  MESSAGE("-- Found COLAMD header in: ${COLAMD_INCLUDE}")

  MESSAGE("-- Check for CCOLAMD")
  FIND_LIBRARY(CCOLAMD_LIB NAMES ccolamd PATHS ${SUITESPARSE_SEARCH_LIBS})
  FIND_PATH(CCOLAMD_INCLUDE NAMES ccolamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
  IF (NOT EXISTS ${CCOLAMD_LIB} OR NOT EXISTS ${CCOLAMD_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find CCOLAMD (part of suitesparse.)"
            "Please specify -DCOLAMD_INCLUDE=... and -DCOLAMD_LIB=...")
  ENDIF (NOT EXISTS ${CCOLAMD_LIB} OR NOT EXISTS ${CCOLAMD_INCLUDE})
  MESSAGE("-- Found CCOLAMD library: ${CCOLAMD_LIB}")
  MESSAGE("-- Found CCOLAMD header in: ${CCOLAMD_INCLUDE}")

  MESSAGE("-- Check for CHOLMOD")
  FIND_LIBRARY(CHOLMOD_LIB NAMES cholmod PATHS ${SUITESPARSE_SEARCH_LIBS})
  FIND_PATH(CHOLMOD_INCLUDE NAMES cholmod.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
  IF (NOT EXISTS ${CHOLMOD_LIB} OR NOT EXISTS ${CHOLMOD_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find CHOLMOD (part of suitesparse.)"
            "Please specify -DCHOLMOD_INCLUDE=... and -DCHOLMOD_LIB=...")
  ENDIF (NOT EXISTS ${CHOLMOD_LIB} OR NOT EXISTS ${CHOLMOD_INCLUDE})
  MESSAGE("-- Found CHOLMOD library: ${CHOLMOD_LIB}")
  MESSAGE("-- Found CHOLMOD header in: ${CHOLMOD_INCLUDE}")

  MESSAGE("-- Check for METIS (optional)")
  FIND_LIBRARY(METIS_LIB NAMES metis PATHS ${SUITESPARSE_SEARCH_LIBS})
  IF (NOT EXISTS ${METIS_LIB})
    MESSAGE("   Can't find METIS; disabling. (part of suitesparse.)")
  ELSE (NOT EXISTS ${METIS_LIB})
    MESSAGE("-- Found METIS library: ${METIS_LIB}")
  ENDIF (NOT EXISTS ${METIS_LIB})

  MESSAGE("-- Check for LAPACK")
  IF (APPLE)
    # Mac OS X has LAPACK/BLAS bundled in a framework called "vecLib". Search for
    # that instead of for the normal "lapack" library.
    FIND_LIBRARY(LAPACK_LIB NAMES vecLib)
  ELSE (APPLE)
    FIND_LIBRARY(LAPACK_LIB NAMES lapack)
  ENDIF (APPLE)

  IF (NOT EXISTS ${LAPACK_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find LAPACK "
            "Please specify -DLAPACK_LIB=...")
  ENDIF (NOT EXISTS ${LAPACK_LIB})
  MESSAGE ("-- Found LAPACK library: ${LAPACK_LIB}")

ELSE (SUITESPARSE)
  ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
ENDIF (SUITESPARSE)

# Google Flags
OPTION(GFLAGS
       "Enable Google Flags."
       ON)

IF (GFLAGS)
  MESSAGE("-- Check for Google Flags")
  FIND_LIBRARY(GFLAGS_LIB NAMES gflags)
  FIND_PATH(GFLAGS_INCLUDE NAMES gflags/gflags.h)
  IF (NOT EXISTS ${GFLAGS_LIB} OR NOT EXISTS ${GFLAGS_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_INCLUDE=... and -DGFLAGS_LIB=...")
  ENDIF (NOT EXISTS ${GFLAGS_LIB} OR NOT EXISTS ${GFLAGS_INCLUDE})
  MESSAGE("-- Found Google Flags library: ${GFLAGS_LIB}")
  MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE}")
ELSE (GFLAGS)
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  ADD_DEFINITIONS(-DCERES_NO_GFLAGS)
ENDIF (GFLAGS)

# Google Logging
MESSAGE("-- Check for Google Log")
FIND_LIBRARY(GLOG_LIB NAMES glog)
FIND_PATH(GLOG_INCLUDE NAMES glog/logging.h)
IF (NOT EXISTS ${GLOG_LIB} OR NOT EXISTS ${GLOG_INCLUDE})
  MESSAGE(FATAL_ERROR
          "Can't find Google Log. Please specify: "
          "-DGLOG_INCLUDE=... and -DGLOG_LIB=...")
ENDIF (NOT EXISTS ${GLOG_LIB} OR NOT EXISTS ${GLOG_INCLUDE})
MESSAGE("-- Found Google Log library: ${GLOG_LIB}")
MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE}")

# Eigen
MESSAGE("-- Check for Eigen 3.0")
SET(EIGEN_SEARCH_HEADERS
    ${SEARCH_HEADERS}
    /usr/include
    /usr/include/eigen3  # Ubuntu 10.04's default location.
    /usr/local/include
    /usr/local/include/eigen3,
    /opt/local/var/macports/software/eigen3/opt/local/include/eigen3/)
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core PATHS
          /usr/include/eigen2
          )
IF (NOT EXISTS ${EIGEN_INCLUDE})
 MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("-- Found Eigen 3.0: ${EIGEN_INCLUDE}")

OPTION(SCHUR_SPECIALIZATIONS
       "Enable fixed-size schur specializations."
       ON)

IF (NOT SCHUR_SPECIALIZATIONS)
  # Disable fixed-size specializations of the schur complement solver, which
  # requires multiple gigabytes of memory to compile.
  ADD_DEFINITIONS(-DCERES_RESTRICT_SCHUR_SPECIALIZATION)
ENDIF (NOT SCHUR_SPECIALIZATIONS)

OPTION(OPENMP
       "Enable threaded solving in Ceres (requires OpenMP)"
       ON)

IF (OPENMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    MESSAGE("-- Found OpenMP.")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ADD_DEFINITIONS(-DCERES_USE_OPENMP)
  ELSE(OPENMP_FOUND)
    MESSAGE("-- Can't find OpenMP. Continuing without it.")
  ENDIF(OPENMP_FOUND)
ENDIF (OPENMP)

OPTION(PROTOBUF
       "Enable protocol buffers support."
       ON)

# Protocol buffers support.
IF (PROTOBUF)
  FIND_PACKAGE(Protobuf)
  IF (PROTOBUF_FOUND)
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/internal)
  ELSE (PROTOBUF_FOUND)
    ADD_DEFINITIONS(-DCERES_DONT_HAVE_PROTOCOL_BUFFERS)
  ENDIF (PROTOBUF_FOUND)
ELSE (PROTOBUF)
  ADD_DEFINITIONS(-DCERES_DONT_HAVE_PROTOCOL_BUFFERS)
ENDIF (PROTOBUF)

IF (UNIX)
  # Atleast on linux, we need pthreads to be enabled for mutex to
  # compile. This may not work on windows or android.
  FIND_PACKAGE(Threads REQUIRED)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
  ADD_DEFINITIONS(-DCERES_HAVE_PTHREAD)
  ADD_DEFINITIONS(-DCERES_HAVE_RWLOCK)
ENDIF (UNIX)


# Use the std namespace for the hash<> and related templates. This may vary by
# system.
ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std { namespace tr1 {\"")
ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}}\"")

INCLUDE_DIRECTORIES(
  include
  internal
  internal/ceres
  ${AMD_INCLUDE}
  ${COLAMD_INCLUDE}
  ${CHOLMOD_INCLUDE}
  ${GLOG_INCLUDE}
  ${GFLAGS_INCLUDE}
  ${EIGEN_INCLUDE}
)

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
IF (NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_BUILD_TYPE)

ADD_SUBDIRECTORY(internal/ceres)
ADD_SUBDIRECTORY(examples)

# TODO(sameeragarwal): The following flags break the old gcc that
# ships on Mac OS X. Make this conditional on operating system and
# compiler.
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native -march=native")
