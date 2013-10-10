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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
CMAKE_POLICY(VERSION 2.8)

IF (COMMAND cmake_policy)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF (COMMAND cmake_policy)

PROJECT(CERES C CXX)

# Set up the git hook to make Gerrit Change-Id: lines in commit messages.
SET (LOCAL_GIT_DIRECTORY)
IF (EXISTS ${CMAKE_SOURCE_DIR}/.git)
  # .git directory can be found on Unix based system, or on Windows with
  # Git Bash (shipped with msysgit)
  SET (LOCAL_GIT_DIRECTORY ${CMAKE_SOURCE_DIR}/.git)
ELSE (EXISTS ${CMAKE_SOURCE_DIR}/.git)
  # TODO(keir) Add proper Windows support
ENDIF (EXISTS ${CMAKE_SOURCE_DIR}/.git)

IF (EXISTS ${LOCAL_GIT_DIRECTORY})
  IF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
    # Download the hook only if it is not already present
    FILE(DOWNLOAD https://ceres-solver-review.googlesource.com/tools/hooks/commit-msg
         ${CMAKE_BINARY_DIR}/commit-msg)

    # Make the downloaded file executable, since it is not by default.
    FILE(COPY ${CMAKE_BINARY_DIR}/commit-msg
         DESTINATION ${LOCAL_GIT_DIRECTORY}/hooks/
         FILE_PERMISSIONS
           OWNER_READ OWNER_WRITE OWNER_EXECUTE
           GROUP_READ GROUP_WRITE GROUP_EXECUTE
           WORLD_READ WORLD_EXECUTE)
  ENDIF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
ENDIF (EXISTS ${LOCAL_GIT_DIRECTORY})

# Make CMake aware of the cmake folder for local FindXXX scripts.
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Important: Always bump the second number (e.g. 1.3.x to 1.4.0) for any
# release that changes the ABI. The ABI changes for almost any modification to
# include/ceres (e.g. the public API). If you are unsure about whether
# something is an ABI change, please ask on the list.
#
# For versions without ABI changes, bump the smallest number in CERES_VERSION,
# but leave the CERES_ABI_VERSION unchanged.
SET(CERES_VERSION_MAJOR 1)
SET(CERES_VERSION_MINOR 7)
SET(CERES_VERSION_PATCH 0)
SET(CERES_VERSION
    ${CERES_VERSION_MAJOR}.${CERES_VERSION_MINOR}.${CERES_VERSION_PATCH})
SET(CERES_ABI_VERSION 1.7.0)

ENABLE_TESTING()

OPTION(MINIGLOG "Use a stripped down version of glog." OFF)
OPTION(GFLAGS "Enable Google Flags." ON)
OPTION(SUITESPARSE "Enable SuiteSparse." ON)
OPTION(CXSPARSE "Enable CXSparse." ON)
OPTION(LAPACK "Enable use of LAPACK." ON)
# Template specializations for the Schur complement based solvers. If
# compile time, binary size or compiler performance is an issue, you
# may consider disabling this.
OPTION(SCHUR_SPECIALIZATIONS "Enable fixed-size schur specializations." ON)
OPTION(CUSTOM_BLAS
       "Use handcoded BLAS routines (usually faster) instead of Eigen."
       ON)
# Multithreading using OpenMP
OPTION(OPENMP "Enable threaded solving in Ceres (requires OpenMP)" ON)
# TODO(sameeragarwal): Replace this with a positive option instead?
OPTION(DISABLE_TR1
       "Don't use TR1. This replaces some hash tables with sets. Slower."
       OFF)
# Line search minimizer is useful for large scale problems or when
# sparse linear algebra libraries are not available. If compile time,
# binary size or compiler performance is an issue, consider disabling
# this.
OPTION(LINE_SEARCH_MINIMIZER "Enable the line search minimizer." ON)
OPTION(BUILD_TESTING "Enable tests" ON)
OPTION(BUILD_DOCUMENTATION "Build User's Guide (html)" OFF)
OPTION(BUILD_EXAMPLES "Build examples" ON)
OPTION(BUILD_SHARED_LIBS "Build Ceres as a shared library." OFF)

# Default locations to search for on various platforms.

# Libraries
LIST(APPEND CMAKE_LIBRARY_PATH /opt/local/lib)
LIST(APPEND CMAKE_LIBRARY_PATH /usr/lib)
LIST(APPEND CMAKE_LIBRARY_PATH /usr/lib/atlas)
LIST(APPEND CMAKE_LIBRARY_PATH /usr/lib64/atlas)
LIST(APPEND CMAKE_LIBRARY_PATH /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND CMAKE_LIBRARY_PATH /usr/local/lib)

# Headers
LIST(APPEND CMAKE_INCLUDE_PATH /opt/local/include)
LIST(APPEND CMAKE_INCLUDE_PATH /opt/local/var/macports/software/eigen3) # Mac OS X
LIST(APPEND CMAKE_INCLUDE_PATH /opt/local/include/eigen3) # Mac OS X
LIST(APPEND CMAKE_INCLUDE_PATH /usr/include)
LIST(APPEND CMAKE_INCLUDE_PATH /usr/include/eigen3) # Ubuntu 10.04's default location.
LIST(APPEND CMAKE_INCLUDE_PATH /usr/local/homebrew/include) # Mac OS X
LIST(APPEND CMAKE_INCLUDE_PATH /usr/local/homebrew/include/eigen3)  # Mac OS X
LIST(APPEND CMAKE_INCLUDE_PATH /usr/local/include)
LIST(APPEND CMAKE_INCLUDE_PATH /usr/local/include/eigen3)

# Eigen.
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core)
IF (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ELSE (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE("-- Found Eigen 3.x: ${EIGEN_INCLUDE}")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MARK_AS_ADVANCED(EIGEN_INCLUDE)

# LAPACK (& BLAS).
IF (LAPACK)
  FIND_PACKAGE(LAPACK QUIET)
  IF (LAPACK_FOUND)
    MESSAGE("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
  ELSE (LAPACK_FOUND)
    MESSAGE("-- Did not find LAPACK library, disabling LAPACK support.")
  ENDIF (LAPACK_FOUND)

  FIND_PACKAGE(BLAS QUIET)
  IF (BLAS_FOUND)
    MESSAGE("-- Found BLAS library: ${BLAS_LIBRARIES}")
  ELSE (BLAS_FOUND)
    MESSAGE("-- Did not find BLAS library, disabling LAPACK support.")
  ENDIF (BLAS_FOUND)

  IF (NOT (LAPACK_FOUND AND BLAS_FOUND))
    # Retain the help string associated with the LAPACK option
    # when updating it to disable use of LAPACK.
    GET_PROPERTY(HELP_STRING CACHE LAPACK PROPERTY HELPSTRING)
    SET(LAPACK OFF CACHE BOOL "${HELP_STRING}" FORCE)
    ADD_DEFINITIONS(-DCERES_NO_LAPACK)
  ENDIF (NOT (LAPACK_FOUND AND BLAS_FOUND))
ELSE (LAPACK)
  MESSAGE("-- Building without LAPACK.")
  ADD_DEFINITIONS(-DCERES_NO_LAPACK)
ENDIF (LAPACK)

# SuiteSparse.
IF (SUITESPARSE AND NOT LAPACK)
  # If user has disabled LAPACK, but left SUITESPARSE ON, turn it OFF,
  # LAPACK controls whether Ceres will be linked, directly or indirectly
  # via SuiteSparse to LAPACK.
  MESSAGE("-- Disabling SuiteSparse as use of LAPACK has been disabled, "
    "turn ON LAPACK to enable (optional) building with SuiteSparse.")
  # Retain the help string associated with the SUITESPARSE option
  # when updating it to disable use of SuiteSparse.
  GET_PROPERTY(HELP_STRING CACHE SUITESPARSE PROPERTY HELPSTRING)
  SET(SUITESPARSE OFF CACHE BOOL "${HELP_STRING}" FORCE)
ENDIF (SUITESPARSE AND NOT LAPACK)
IF (SUITESPARSE)
  # By default, if SuiteSparse and all dependencies are found, Ceres is
  # built with SuiteSparse support.

  # Check for SuiteSparse and dependencies.
  FIND_PACKAGE(SuiteSparse QUIET)
  IF (SUITESPARSE_FOUND)
    # On Ubuntu the system install of SuiteSparse (v3.4.0) up to at least
    # Ubuntu 13.10 cannot be used to link shared libraries.
    IF (BUILD_SHARED_LIBS AND
        SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)
      MESSAGE(FATAL_ERROR "You are attempting to build Ceres as a shared "
        "library on Ubuntu using a system package install of SuiteSparse "
        "3.4.0. This package is broken and does not support the "
        "construction of shared libraries (you can still build Ceres as "
        "a static library).  If you wish to build a shared version of Ceres "
        "you should uninstall the system install of SuiteSparse "
        "(libsuitesparse-dev) and perform a source install of SuiteSparse "
        "(we recommend that you use the latest version), "
        "see: http://homes.cs.washington.edu/~sagarwal"
        "/ceres-solver/dev/building.html for more information.")
    ENDIF (BUILD_SHARED_LIBS AND
      SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

    # By default, if all of SuiteSparse's dependencies are found, Ceres is
    # built with SuiteSparse support.
    MESSAGE("-- Found all SuiteSparse dependencies. Building with SuiteSparse.")
  ELSE (SUITESPARSE_FOUND)
    # Disable use of SuiteSparse if it cannot be found and continue.
    MESSAGE("-- Did not find all SuiteSparse dependencies, disabling "
      "SuiteSparse support.")
    # Retain the help string associated with the SUITESPARSE option
    # when updating it to disable use of SuiteSparse.
    GET_PROPERTY(HELP_STRING CACHE SUITESPARSE PROPERTY HELPSTRING)
    SET(SUITESPARSE OFF CACHE BOOL "${HELP_STRING}" FORCE)
    ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
  ENDIF (SUITESPARSE_FOUND)
ELSE (SUITESPARSE)
  MESSAGE("-- Building without SuiteSparse.")
  ADD_DEFINITIONS(-DCERES_NO_SUITESPARSE)
ENDIF (SUITESPARSE)

# CXSparse.
IF (CXSPARSE)
  SET(CXSPARSE_FOUND ON)
  FIND_LIBRARY(CXSPARSE_LIB NAMES cxsparse)
  IF (EXISTS ${CXSPARSE_LIB})
    MESSAGE("-- Found CXSparse library in: ${CXSPARSE_LIB}")
  ELSE (EXISTS ${CXSPARSE_LIB})
    MESSAGE("-- Did not find CXSparse header")
    SET(CXSPARSE_FOUND FALSE)
  ENDIF (EXISTS ${CXSPARSE_LIB})
  MARK_AS_ADVANCED(CXSPARSE_LIB)

  FIND_PATH(CXSPARSE_INCLUDE NAMES cs.h)
  IF (EXISTS ${CXSPARSE_INCLUDE})
    MESSAGE("-- Found CXSparse header in: ${CXSPARSE_INCLUDE}")
  ELSE (EXISTS ${CXSPARSE_INCLUDE})
    MESSAGE("-- Did not find CXSparse header")
    SET(CXSPARSE_FOUND FALSE)
  ENDIF (EXISTS ${CXSPARSE_INCLUDE})
  MARK_AS_ADVANCED(CXSPARSE_INCLUDE)

  IF (CXSPARSE_FOUND)
    # By default, if CXSparse and all dependencies are found, Ceres is
    # built with CXSparse support.
    MESSAGE("-- Building with CXSparse.")
  ELSE (CXSPARSE_FOUND)
    # Disable use of CXSparse if it cannot be found and continue.
    MESSAGE("-- Did not find CXSparse, Building without CXSparse.")
    # Retain the help string associated with the CXSPARSE option
    # when updating it to disable use of CXSparse.
    GET_PROPERTY(HELP_STRING CACHE CXSPARSE PROPERTY HELPSTRING)
    SET(CXSPARSE OFF CACHE BOOL "${HELP_STRING}" FORCE)
    ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
  ENDIF (CXSPARSE_FOUND)
ELSE (CXSPARSE)
  MESSAGE("-- Building without CXSparse.")
  ADD_DEFINITIONS(-DCERES_NO_CXSPARSE)
ENDIF (CXSPARSE)

# GFlags.
IF (GFLAGS)
  FIND_LIBRARY(GFLAGS_LIB NAMES gflags)
  IF (NOT EXISTS ${GFLAGS_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_LIB=...")
  ENDIF (NOT EXISTS ${GFLAGS_LIB})
  MARK_AS_ADVANCED(GFLAGS_LIB)
  MESSAGE("-- Found Google Flags library: ${GFLAGS_LIB}")
  FIND_PATH(GFLAGS_INCLUDE NAMES gflags/gflags.h)
  IF (NOT EXISTS ${GFLAGS_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_INCLUDE=...")
  ENDIF (NOT EXISTS ${GFLAGS_INCLUDE})
  MARK_AS_ADVANCED(GFLAGS_INCLUDE)
  MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE}")
ELSE (GFLAGS)
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  ADD_DEFINITIONS(-DCERES_NO_GFLAGS)
ENDIF (GFLAGS)

# MiniGLog.
IF (MINIGLOG)
  SET(GLOG_LIB miniglog)
  MESSAGE("-- Using minimal Glog substitute (library): ${GLOG_LIB}")
  SET(GLOG_INCLUDE internal/ceres/miniglog)
  MESSAGE("-- Using minimal Glog substitute (include): ${GLOG_INCLUDE}")
ELSE (MINIGLOG)
  FIND_LIBRARY(GLOG_LIB NAMES glog)
  IF (EXISTS ${GLOG_LIB})
    MESSAGE("-- Found Google Log library: ${GLOG_LIB}")
  ELSE (EXISTS ${GLOG_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: -DGLOG_LIB=...")
  ENDIF (EXISTS ${GLOG_LIB})
  MARK_AS_ADVANCED(GLOG_LIB)

  FIND_PATH(GLOG_INCLUDE NAMES glog/logging.h)
  IF (EXISTS ${GLOG_INCLUDE})
    MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE}")
  ELSE (EXISTS ${GLOG_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: -DGLOG_INCLUDE=...")
  ENDIF (EXISTS ${GLOG_INCLUDE})
  MARK_AS_ADVANCED(GLOG_INCLUDE)
ENDIF (MINIGLOG)

IF (NOT SCHUR_SPECIALIZATIONS)
  ADD_DEFINITIONS(-DCERES_RESTRICT_SCHUR_SPECIALIZATION)
  MESSAGE("-- Disabling Schur specializations (faster compiles)")
ENDIF (NOT SCHUR_SPECIALIZATIONS)

IF (NOT LINE_SEARCH_MINIMIZER)
  ADD_DEFINITIONS(-DCERES_NO_LINE_SEARCH_MINIMIZER)
  MESSAGE("-- Disabling line search minimizer")
ENDIF (NOT LINE_SEARCH_MINIMIZER)

IF (NOT CUSTOM_BLAS)
  ADD_DEFINITIONS(-DCERES_NO_CUSTOM_BLAS)
  MESSAGE("-- Disabling custom blas")
ENDIF (NOT CUSTOM_BLAS)

IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(OPENMP_FOUND FALSE)
ELSE (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  IF (OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
  ENDIF (OPENMP)
ENDIF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

IF (OPENMP_FOUND)
  MESSAGE("-- Found OpenMP.")
  ADD_DEFINITIONS(-DCERES_USE_OPENMP)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  IF (UNIX)
    # At least on Linux, we need pthreads to be enabled for mutex to
    # compile.  This may not work on Windows or Android.
    FIND_PACKAGE(Threads REQUIRED)
    SET(STATIC_LIBRARY_FLAGS
        "${STATIC_LIBRARY_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
    SET(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
    ADD_DEFINITIONS(-DCERES_HAVE_PTHREAD)
    ADD_DEFINITIONS(-DCERES_HAVE_RWLOCK)
  ENDIF (UNIX)
ELSE (OPENMP_FOUND)
  MESSAGE("-- Can't find OpenMP. Disabling multithreading.")
  ADD_DEFINITIONS(-DCERES_NO_THREADS)
ENDIF (OPENMP_FOUND)

IF (DISABLE_TR1)
  MESSAGE("-- Replacing unordered_map/set with map/set (warning: slower!)")
  ADD_DEFINITIONS(-DCERES_NO_TR1)
ELSE (DISABLE_TR1)
  MESSAGE("-- Using normal TR1 unordered_map/set")
  # Use the std namespace for the hash<> and related templates. This may vary by
  # system.
  IF (MSVC)
    IF (MSVC90)
      # Special case for Visual Studio 2008.
      # Newer versions have got tr1 symbols in another namespace,
      # and this is being handled in Else branch of this condition.
      # Probably Visual studio 2003 and 2005 also shall be handled here,
      # but don't have by hand to verify and most likely they're not
      # used by Ceres users anyway.
      ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std { namespace tr1 {\"")
      ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}}\"")
    ELSE (MSVC90)
      # This is known to work with Visual Studio 2010 Express.
      # Further, for as long Visual Studio 2012 didn't move tr1 to
      # just another namespace, the same define will work for it as well.
      # Hopefully all further versions will also keep working with this define.
      ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std {\"")
      ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}\"")
    ENDIF(MSVC90)
  ELSE (MSVC)
    # This is known to work with recent versions of Linux and Mac OS X.
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_START=namespace std { namespace tr1 {\"")
    ADD_DEFINITIONS("\"-DCERES_HASH_NAMESPACE_END=}}\"")
  ENDIF (MSVC)
ENDIF (DISABLE_TR1)

INCLUDE_DIRECTORIES(
  include
  internal
  internal/ceres
  ${GLOG_INCLUDE}
  ${EIGEN_INCLUDE}
  )

FILE(GLOB CERES_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/*.h)
INSTALL(FILES ${CERES_HDRS} DESTINATION include/ceres)

FILE(GLOB CERES_PUBLIC_INTERNAL_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/internal/*.h)
INSTALL(FILES ${CERES_PUBLIC_INTERNAL_HDRS} DESTINATION include/ceres/internal)

IF (SUITESPARSE)
  INCLUDE_DIRECTORIES(${SUITESPARSE_INCLUDE_DIRS})
ENDIF (SUITESPARSE)

IF (CXSPARSE)
  INCLUDE_DIRECTORIES(${CXSPARSE_INCLUDE})
ENDIF (CXSPARSE)

IF (GFLAGS)
  INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE})
ENDIF (GFLAGS)

IF (BUILD_SHARED_LIBS)
  MESSAGE("-- Building Ceres as a shared library.")
ELSE (BUILD_SHARED_LIBS)
  MESSAGE("-- Building Ceres as a static library.")
ENDIF (BUILD_SHARED_LIBS)

# Change the default build type from Debug to Release, while still
# supporting overriding the build type.
#
# The CACHE STRING logic here and elsewhere is needed to force CMake
# to pay attention to the value of these variables.
IF (NOT CMAKE_BUILD_TYPE)
  MESSAGE("-- No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ELSE (NOT CMAKE_BUILD_TYPE)
  IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("\n=================================================================================")
    MESSAGE("\n-- Build type: Debug. Performance will be terrible!")
    MESSAGE("-- Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
    MESSAGE("\n=================================================================================")
  ENDIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
ENDIF (NOT CMAKE_BUILD_TYPE)

# Set the default Ceres flags to an empty string.
SET (CERES_CXX_FLAGS)

IF (CMAKE_BUILD_TYPE STREQUAL "Release")
  IF (CMAKE_COMPILER_IS_GNUCXX)
    # Linux
    IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
      IF (NOT GCC_VERSION VERSION_LESS 4.2)
        SET (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -march=native -mtune=native")
      ENDIF (NOT GCC_VERSION VERSION_LESS 4.2)
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # Mac OS X
    IF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
      SET (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -msse3")
      # Use of -fast only applicable for Apple's GCC
      # Assume this is being used if GCC version < 4.3 on OSX
      EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER}
        ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      IF (GCC_VERSION VERSION_LESS 4.3)
        SET (CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -fast")
      ENDIF (GCC_VERSION VERSION_LESS 4.3)
    ENDIF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  ENDIF (CMAKE_COMPILER_IS_GNUCXX)
  IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Use of -O4 requires use of gold linker & LLVM-gold plugin, which might
    # well not be present / in use and without which files will compile, but
    # not link ('file not recognized') so explicitly check for support
    INCLUDE(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-O4" HAVE_LTO_SUPPORT)
    IF (HAVE_LTO_SUPPORT)
      MESSAGE(STATUS "Enabling link-time optimization (-O4)")
      SET(CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -O4")
    ELSE ()
      MESSAGE(STATUS "Compiler/linker does not support link-time optimization (-O4), disabling.")
    ENDIF (HAVE_LTO_SUPPORT)
  ENDIF ()
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")

SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CERES_CXX_FLAGS}")

# After the tweaks for the compile settings, disable some warnings on MSVC.
IF (MSVC)
  # Disable signed/unsigned int conversion warnings.
  ADD_DEFINITIONS("/wd4018")
  # Disable warning about using struct/class for the same symobl.
  ADD_DEFINITIONS("/wd4099")
  # Disable warning about the insecurity of using "std::copy".
  ADD_DEFINITIONS("/wd4996")
  # Disable performance warning about int-to-bool conversion.
  ADD_DEFINITIONS("/wd4800")
  # Disable performance warning about fopen insecurity.
  ADD_DEFINITIONS("/wd4996")
  # Disable warning about int64 to int32 conversion. Disabling
  # this warning may not be correct; needs investigation.
  # TODO(keir): Investigate these warnings in more detail.
  ADD_DEFINITIONS("/wd4244")
  # It's not possible to use STL types in DLL interfaces in a portable and
  # reliable way. However, that's what happens with Google Log and Google Flags
  # on Windows. MSVC gets upset about this and throws warnings that we can't do
  # much about. The real solution is to link static versions of Google Log and
  # Google Test, but that seems tricky on Windows. So, disable the warning.
  ADD_DEFINITIONS("/wd4251")

  # Google Flags doesn't have their DLL import/export stuff set up correctly,
  # which results in linker warnings. This is irrelevant for Ceres, so ignore
  # the warnings.
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4049")

  # Tuple sizes of 10 are used by Gtest.
  ADD_DEFINITIONS("-D_VARIADIC_MAX=10")
ENDIF (MSVC)

IF (UNIX)
  # GCC is not strict enough by default, so enable most of the warnings.
  SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
ENDIF (UNIX)

# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600")
  # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
  # option, so check for its presence before adding it to the default flags set.
  CHECK_CXX_COMPILER_FLAG("-Wno-return-type-c-linkage"
                          HAVE_RETURN_TYPE_C_LINKAGE)
  IF (HAVE_RETURN_TYPE_C_LINKAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  ENDIF(HAVE_RETURN_TYPE_C_LINKAGE)
ENDIF ()

# Xcode 4.5.x used Clang 4.1 (Apple version), this has a bug that prevents
# compilation of Ceres (arises in RemoveParameterBlock).
IF (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER}
    ARGS ${CMAKE_CXX_COMPILER_ARG1} -dumpversion
    OUTPUT_VARIABLE CLANG_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  # Use version > 4.0 & < 4.2 to catch all 4.1(.x) versions.
  IF (CLANG_VERSION VERSION_GREATER 4.0 AND
      CLANG_VERSION VERSION_LESS 4.2)
    MESSAGE(FATAL_ERROR "You are attempting to build Ceres on OS X using Xcode "
      "4.5.x (Clang version: ${CLANG_VERSION}). This version of Clang has a "
      "bug that prevents compilation of Ceres, please update to "
      "Xcode >= 4.6.3.")
  ENDIF (CLANG_VERSION VERSION_GREATER 4.0 AND
    CLANG_VERSION VERSION_LESS 4.2)
ENDIF (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

ADD_SUBDIRECTORY(internal/ceres)

IF (BUILD_DOCUMENTATION)
  MESSAGE("-- Documentation building is enabled")

  # Generate the User's Guide (html).
  # The corresponding target is UserGuide, but is included in ALL.
  ADD_SUBDIRECTORY(docs)
ENDIF (BUILD_DOCUMENTATION)

IF (BUILD_EXAMPLES)
  MESSAGE("-- Build the examples.")
  ADD_SUBDIRECTORY(examples)
ELSE (BUILD_EXAMPLES)
  MESSAGE("-- Do not build any example.")
ENDIF (BUILD_EXAMPLES)

# Add an uninstall target to remove all installed files.
CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/cmake/uninstall.cmake.in"
               "${CMAKE_BINARY_DIR}/cmake/uninstall.cmake"
               IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
                  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake/uninstall.cmake)

# Set up install directories. INCLUDE_INSTALL_DIR, LIB_INSTALL_DIR and
# CMAKECONFIG_INSTALL_DIR must not be absolute paths.
IF (WIN32)
  SET(INCLUDE_INSTALL_DIR Include)
  SET(LIB_INSTALL_DIR Lib)
  SET(CMAKECONFIG_INSTALL_DIR CMake)
ELSE ()
  SET(INCLUDE_INSTALL_DIR include)
  SET(LIB_INSTALL_DIR lib)
  SET(CMAKECONFIG_INSTALL_DIR share/Ceres)
ENDIF ()

# This "exports" all targets which have been put into the export set
# "CeresExport". This means that CMake generates a file with the given
# filename, which can later on be loaded by projects using this package.
# This file contains ADD_LIBRARY(bar IMPORTED) statements for each target
# in the export set, so when loaded later on CMake will create "imported"
# library targets from these, which can be used in many ways in the same way
# as a normal library target created via a normal ADD_LIBRARY().
INSTALL(EXPORT CeresExport
        DESTINATION ${CMAKECONFIG_INSTALL_DIR} FILE CeresTargets.cmake)

# Figure out the relative path from the installed Config.cmake file to the
# install prefix (which may be at runtime different from the chosen
# CMAKE_INSTALL_PREFIX if under Windows the package was installed anywhere)
# This relative path will be configured into the CeresConfig.cmake.
FILE(RELATIVE_PATH relInstallDir
     ${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR} ${CMAKE_INSTALL_PREFIX})

# Create a CeresConfig.cmake file. <name>Config.cmake files are searched by
# FIND_PACKAGE() automatically. We configure that file so that we can put any
# information we want in it, e.g. version numbers, include directories, etc.
CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/cmake/CeresConfig.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CeresConfig.cmake" @ONLY)

# Additionally, when CMake has found a CeresConfig.cmake, it can check for a
# CeresConfigVersion.cmake in the same directory when figuring out the version
# of the package when a version has been specified in the FIND_PACKAGE() call,
# e.g. FIND_PACKAGE(Ceres [1.4.2] REQUIRED). The version argument is optional.
CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/cmake/CeresConfigVersion.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/CeresConfigVersion.cmake" @ONLY)

# Install these two files into the same directory as the generated exports-file.
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/CeresConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CeresConfigVersion.cmake"
              "${CMAKE_SOURCE_DIR}/cmake/depend.cmake"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})
