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

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

IF (COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

PROJECT(CERES C CXX)

ENABLE_TESTING()

OPTION(BUILD_TESTING
       "Enable tests"
       ON)

# To get a more static build, try the following line on Mac and Linux:
# SET(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/homebrew/include) # Mac OS X
LIST(APPEND SEARCH_HEADERS /opt/local/include)

# Locations to search for Eigen
SET(EIGEN_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/include/eigen3) # Ubuntu 10.04's default location.
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/include/eigen3)
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/homebrew/include/eigen3)  # Mac OS X
LIST(APPEND EIGEN_SEARCH_HEADERS /opt/local/var/macports/software/eigen3/opt/local/include/eigen3) # Mac OS X

# Locations to search for SuiteSparse
SET(SUITESPARSE_SEARCH_LIBS ${SEARCH_LIBS})
LIST(APPEND SUITESPARSE_SEARCH_LIBS /usr/lib/suitesparse) # Ubuntu
LIST(APPEND SUITESPARSE_SEARCH_LIBS /usr/local/lib/suitesparse)
LIST(APPEND SUITESPARSE_SEARCH_LIBS /opt/local/lib/ufsparse) # Mac OS X

SET(SUITESPARSE_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /usr/include/suitesparse) # Ubuntu
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /usr/local/include/suitesparse)
LIST(APPEND SUITESPARSE_SEARCH_HEADERS /opt/local/include/ufsparse) # Mac OS X

SET(CXSPARSE_SEARCH_LIBS ${SEARCH_LIBS})
SET(CXSPARSE_SEARCH_HEADERS ${SEARCH_HEADERS})

# Check for SuiteSparse dependencies
MESSAGE("-- Check for AMD")
SET(AMD_FOUND TRUE)

FIND_LIBRARY(AMD_LIB NAMES amd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${AMD_LIB})
  MESSAGE("-- Found AMD library: ${AMD_LIB}")
ELSE (EXISTS ${AMD_LIB})
  MESSAGE("-- Did not find AMD library")
  SET(AMD_FOUND FALSE)
ENDIF (EXISTS ${AMD_LIB})

FIND_PATH(AMD_INCLUDE NAMES amd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${AMD_INCLUDE})
  MESSAGE("-- Found AMD header in: ${AMD_INCLUDE}")
ELSE (EXISTS ${AMD_INCLUDE})
  MESSAGE("-- Did not find AMD header")
  SET(AMD_FOUND FALSE)
ENDIF (EXISTS ${AMD_INCLUDE})

MESSAGE("-- Check for CAMD")
SET(CAMD_FOUND TRUE)

FIND_LIBRARY(CAMD_LIB NAMES camd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CAMD_LIB})
  MESSAGE("-- Found CAMD library: ${CAMD_LIB}")
ELSE (EXISTS ${CAMD_LIB})
  MESSAGE("-- Did not find CAMD library")
  SET(CAMD_FOUND FALSE)
ENDIF (EXISTS ${CAMD_LIB})

FIND_PATH(CAMD_INCLUDE NAMES camd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CAMD_INCLUDE})
  MESSAGE("-- Found CAMD header in: ${CAMD_INCLUDE}")
ELSE (EXISTS ${CAMD_INCLUDE})
  MESSAGE("-- Did not find CAMD header")
  SET(CAMD_FOUND FALSE)
ENDIF (EXISTS ${CAMD_INCLUDE})

MESSAGE("-- Check for COLAMD")
SET(COLAMD_FOUND TRUE)

FIND_LIBRARY(COLAMD_LIB NAMES colamd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${COLAMD_LIB})
  MESSAGE("-- Found COLAMD library: ${COLAMD_LIB}")
ELSE (EXISTS ${COLAMD_LIB})
  MESSAGE("-- Did not find COLAMD library")
  SET(COLAMD_FOUND FALSE)
ENDIF (EXISTS ${COLAMD_LIB})

FIND_PATH(COLAMD_INCLUDE NAMES colamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${COLAMD_INCLUDE})
  MESSAGE("-- Found COLAMD header in: ${COLAMD_INCLUDE}")
ELSE (EXISTS ${COLAMD_INCLUDE})
  MESSAGE("-- Did not find COLAMD header")
  SET(COLAMD_FOUND FALSE)
ENDIF (EXISTS ${COLAMD_INCLUDE})

MESSAGE("-- Check for CCOLAMD")
SET(CCOLAMD_FOUND TRUE)

FIND_LIBRARY(CCOLAMD_LIB NAMES ccolamd PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CCOLAMD_LIB})
  MESSAGE("-- Found CCOLAMD library: ${CCOLAMD_LIB}")
ELSE (EXISTS ${CCOLAMD_LIB})
  MESSAGE("-- Did not find CCOLAMD library")
  SET(CCOLAMD_FOUND FALSE)
ENDIF (EXISTS ${CCOLAMD_LIB})

FIND_PATH(CCOLAMD_INCLUDE NAMES ccolamd.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CCOLAMD_INCLUDE})
  MESSAGE("-- Found CCOLAMD header in: ${CCOLAMD_INCLUDE}")
ELSE (EXISTS ${CCOLAMD_INCLUDE})
  MESSAGE("-- Did not find CCOLAMD header")
  SET(CCOLAMD_FOUND FALSE)
ENDIF (EXISTS ${CCOLAMD_INCLUDE})

MESSAGE("-- Check for CHOLMOD")
SET(CHOLMOD_FOUND TRUE)

FIND_LIBRARY(CHOLMOD_LIB NAMES cholmod PATHS ${SUITESPARSE_SEARCH_LIBS})
IF (EXISTS ${CHOLMOD_LIB})
  MESSAGE("-- Found CHOLMOD library: ${CHOLMOD_LIB}")
ELSE (EXISTS ${CHOLMOD_LIB})
  MESSAGE("-- Did not find CHOLMOD library")
  SET(CHOLMOD_FOUND FALSE)
ENDIF (EXISTS ${CHOLMOD_LIB})

FIND_PATH(CHOLMOD_INCLUDE NAMES cholmod.h PATHS ${SUITESPARSE_SEARCH_HEADERS})
IF (EXISTS ${CHOLMOD_INCLUDE})
  MESSAGE("-- Found CHOLMOD header in: ${CHOLMOD_INCLUDE}")
ELSE (EXISTS ${CHOLMOD_INCLUDE})
  MESSAGE("-- Did not find CHOLMOD header")
  SET(CHOLMOD_FOUND FALSE)
ENDIF (EXISTS ${CHOLMOD_INCLUDE})

MESSAGE("-- Check for METIS (optional)")
FIND_LIBRARY(METIS_LIB NAMES metis PATHS ${SUITESPARSE_SEARCH_LIBS})

IF (EXISTS ${METIS_LIB})
  MESSAGE("-- Found METIS library: ${METIS_LIB}")
ELSE (EXISTS ${METIS_LIB})
  MESSAGE("-- Did not find METIS library")
ENDIF (EXISTS ${METIS_LIB})

SET(BLAS_AND_LAPACK_FOUND TRUE)
IF (APPLE)
  # Mac OS X has LAPACK/BLAS bundled in a framework called
  # "vecLib". Search for that instead of for the normal "lapack"
  # library.
  FIND_LIBRARY(LAPACK_LIB NAMES vecLib)
ELSE (APPLE)
  FIND_LIBRARY(BLAS_LIB NAMES blas)
  IF (EXISTS ${BLAS_LIB})
    MESSAGE("-- Found BLAS library: ${BLAS_LIB}")
  ELSE (EXISTS ${BLAS_LIB})
    MESSAGE("-- Did not find BLAS library")
    SET(BLAS_AND_LAPACK_FOUND FALSE)
  ENDIF (EXISTS ${BLAS_LIB})
  FIND_LIBRARY(LAPACK_LIB NAMES lapack)
ENDIF (APPLE)

IF (EXISTS ${LAPACK_LIB})
  MESSAGE("-- Found LAPACK library: ${LAPACK_LIB}")
ELSE (EXISTS ${LAPACK_LIB})
  SET(BLAS_AND_LAPACK_FOUND FALSE)
  MESSAGE("-- Did not find LAPACK library")
ENDIF (EXISTS ${LAPACK_LIB})

SET(SUITESPARSE_FOUND
    AMD_FOUND AND
    CAMD_FOUND AND
    COLAMD_FOUND AND
    CCOLAMD_FOUND AND
    CHOLMOD_FOUND AND
    BLAS_AND_LAPACK_FOUND)

# By default, if all of SuiteSparse's dependencies are found, Ceres is
# built with SuiteSparse support. -DSUITESPARSE=ON/OFF can be used to
# enable/disable SuiteSparse explicitly.
IF (DEFINED SUITESPARSE)
  IF (SUITESPARSE)
    IF (NOT SUITESPARSE_FOUND)
      MESSAGE(FATAL_ERROR "One or more of SuiteSparse's dependencies was not found")
    ENDIF (NOT SUITESPARSE_FOUND)
  ELSE (SUITESPARSE)
    ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
  ENDIF (SUITESPARSE)
ELSE (DEFINED SUITESPARSE)
  IF (SUITESPARSE_FOUND)
    MESSAGE("-- Found all SuiteSparse dependencies. Building with SuiteSparse")
    SET(SUITESPARSE ON)
  ELSE (SUITESPARSE_FOUND)
    MESSAGE("-- Did not find all SuiteSparse dependencies. Building without SuiteSparse")
    SET(SUITESPARSE OFF)
    ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
  ENDIF (SUITESPARSE_FOUND)
ENDIF (DEFINED SUITESPARSE)


# By default, if all of CXSparse's dependencies are found, Ceres is
# built with CXSparse support. -DCXSPARSE=ON/OFF can be used to
# enable/disable CXSparse explicitly.
MESSAGE("-- Check for CXSparse")
SET(CXSPARSE_FOUND ON)

FIND_LIBRARY(CXSPARSE_LIB NAMES cxsparse PATHS ${CXSPARSE_SEARCH_LIBS})
IF (EXISTS ${CXSPARSE_LIB})
  MESSAGE("-- Found CXSparse library in: ${CXSPARSE_LIB}")
ELSE (EXISTS ${CXSPARSE_LIB})
  MESSAGE("-- Did not find CXSparse header")
  SET(CXSPARSE_FOUND FALSE)
ENDIF (EXISTS ${CXSPARSE_LIB})

FIND_PATH(CXSPARSE_INCLUDE NAMES cs.h PATHS ${CXSPARSE_SEARCH_HEADERS})
IF (EXISTS ${CXSPARSE_INCLUDE})
  MESSAGE("-- Found CXSparse header in: ${CXSPARSE_INCLUDE}")
ELSE (EXISTS ${CXSPARSE_INCLUDE})
  MESSAGE("-- Did not find CXSparse header")
  SET(CXSPARSE_FOUND FALSE)
ENDIF (EXISTS ${CXSPARSE_INCLUDE})

IF (DEFINED CXSPARSE)
  IF (CXSPARSE)
    IF (NOT CXSPARSE_FOUND)
      MESSAGE(FATAL_ERROR "-- CXSparse not found.")
    ENDIF (NOT CXSPARSE_FOUND)
  ELSE (CXSPARSE)
    ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
  ENDIF (CXSPARSE)
ELSE (DEFINED CXSPARSE)
  IF (CXSPARSE_FOUND)
    MESSAGE("-- Building with CXSparse support.")
    SET(CXSPARSE ON)
  ELSE (CXSPARSE_FOUND)
    MESSAGE("-- Building without CXSparse.")
    SET(CXSPARSE OFF)
    ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
  ENDIF (CXSPARSE_FOUND)
ENDIF (DEFINED CXSPARSE)


# Google Flags
OPTION(GFLAGS
       "Enable Google Flags."
       ON)

IF (GFLAGS)
  MESSAGE("-- Check for Google Flags")
  FIND_LIBRARY(GFLAGS_LIB NAMES gflags PATHS ${SEARCH_LIBS})
  IF (NOT EXISTS ${GFLAGS_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_LIB=...")
  ENDIF (NOT EXISTS ${GFLAGS_LIB})
  MESSAGE("-- Found Google Flags library: ${GFLAGS_LIB}")
  FIND_PATH(GFLAGS_INCLUDE NAMES gflags/gflags.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GFLAGS_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_INCLUDE=...")
  ENDIF (NOT EXISTS ${GFLAGS_INCLUDE})
  MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE}")
ELSE (GFLAGS)
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  ADD_DEFINITIONS(-DCERES_NO_GFLAGS)
ENDIF (GFLAGS)

# Google Logging
MESSAGE("-- Check for Google Log")
FIND_LIBRARY(GLOG_LIB NAMES glog PATHS ${SEARCH_LIBS})
IF (NOT EXISTS ${GLOG_LIB})
  MESSAGE(FATAL_ERROR
          "Can't find Google Log. Please specify: "
          "-DGLOG_LIB=...")
ENDIF (NOT EXISTS ${GLOG_LIB})
MESSAGE("-- Found Google Log library: ${GLOG_LIB}")

FIND_PATH(GLOG_INCLUDE NAMES glog/logging.h PATHS ${SEARCH_HEADERS})
IF (NOT EXISTS ${GLOG_INCLUDE})
  MESSAGE(FATAL_ERROR
          "Can't find Google Log. Please specify: "
          "-DGLOG_INCLUDE=...")
ENDIF (NOT EXISTS ${GLOG_INCLUDE})
MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE}")

# Eigen
MESSAGE("-- Check for Eigen 3.0")
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core PATHS ${EIGEN_SEARCH_HEADERS})
IF (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("-- Found Eigen 3.0: ${EIGEN_INCLUDE}")


# Template specializations for the Schur complement based solvers. If
# compile time, binary size or compiler performance is an issue, you
# may consider disabling this.
OPTION(SCHUR_SPECIALIZATIONS
       "Enable fixed-size schur specializations."
       ON)

IF (NOT SCHUR_SPECIALIZATIONS)
  ADD_DEFINITIONS(-DCERES_RESTRICT_SCHUR_SPECIALIZATION)
  MESSAGE("-- Disabling Schur specializations (faster compiles)")
ENDIF (NOT SCHUR_SPECIALIZATIONS)

# Multithreading using OpenMP
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

# Protocol buffers
OPTION(PROTOBUF
       "Enable protocol buffers support."
       ON)

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
  ${GLOG_INCLUDE}
  ${EIGEN_INCLUDE}
  )

IF (SUITESPARSE)
  INCLUDE_DIRECTORIES(${CHOLMOD_INCLUDE})
ENDIF(SUITESPARSE)

IF (CXSPARSE)
  INCLUDE_DIRECTORIES(${CXSPARSE_INCLUDE})
ENDIF(CXSPARSE)

IF (GFLAGS)
  INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE})
ENDIF (GFLAGS)

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
