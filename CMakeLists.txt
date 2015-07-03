# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
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
# Authors: keir@google.com (Keir Mierle)
#          alexs.mac@gmail.com (Alex Stewart)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)
CMAKE_POLICY(VERSION 2.8)

CMAKE_POLICY(SET CMP0003 NEW)
IF (POLICY CMP0042)
  CMAKE_POLICY(SET CMP0042 NEW)
ENDIF()

PROJECT(CERES C CXX)

# Make CMake aware of the cmake folder for local FindXXX scripts,
# append rather than set in case the user has passed their own
# additional paths via -D.
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
INCLUDE(UpdateCacheVariable)

# Set up the git hook to make Gerrit Change-Id: lines in commit messages.
INCLUDE(AddGerritCommitHook)
ADD_GERRIT_COMMIT_HOOK()

# On OS X, add the Homebrew prefix to the set of prefixes searched by
# CMake in find_path & find_library.  This should ensure that we can
# still build Ceres even if Homebrew is installed in a non-standard
# location (not /usr/local).
IF (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  FIND_PROGRAM(HOMEBREW_EXECUTABLE brew)
  MARK_AS_ADVANCED(FORCE HOMEBREW_EXECUTABLE)
  IF (HOMEBREW_EXECUTABLE)
    # Detected a Homebrew install, query for its install prefix.
    EXECUTE_PROCESS(COMMAND ${HOMEBREW_EXECUTABLE} --prefix
      OUTPUT_VARIABLE HOMEBREW_INSTALL_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    MESSAGE(STATUS "Detected Homebrew with install prefix: "
      "${HOMEBREW_INSTALL_PREFIX}, adding to CMake search paths.")
    LIST(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_INSTALL_PREFIX}")
  ENDIF()
ENDIF()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# Set postfixes for generated libraries based on buildtype.
SET(CMAKE_RELEASE_POSTFIX "")
SET(CMAKE_DEBUG_POSTFIX "-debug")

# Important: Always bump the second number (e.g. 1.3.x to 1.4.0) for any
# release that changes the ABI. The ABI changes for almost any modification to
# include/ceres (e.g. the public API). If you are unsure about whether
# something is an ABI change, please ask on the list.
#
# For versions without ABI changes, bump the smallest number in CERES_VERSION,
# but leave the CERES_ABI_VERSION unchanged.
SET(CERES_VERSION_MAJOR 1)
SET(CERES_VERSION_MINOR 10)
SET(CERES_VERSION_PATCH 0)
SET(CERES_VERSION
    ${CERES_VERSION_MAJOR}.${CERES_VERSION_MINOR}.${CERES_VERSION_PATCH})
SET(CERES_ABI_VERSION 1.10.0)

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
OPTION(EIGENSPARSE
  "Enable the use of Eigen as a sparse linear algebra library for
   solving the nonlinear least squares problems. Enabling this
   option will result in an LGPL licensed version of Ceres Solver
   as the Simplicial Cholesky factorization in Eigen is licensed under the LGPL.
   This does not affect the covariance estimation algorithm, as it
   depends on the sparse QR factorization algorithm, which is licensed
   under the MPL."
  OFF)
OPTION(CXX11 "Enable use of C++11 if available (requires client code use C++11)." OFF)
OPTION(BUILD_TESTING "Enable tests" ON)
OPTION(BUILD_DOCUMENTATION "Build User's Guide (html)" OFF)
OPTION(BUILD_EXAMPLES "Build examples" ON)
OPTION(BUILD_SHARED_LIBS "Build Ceres as a shared library." OFF)
IF (MSVC)
  OPTION(MSVC_USE_STATIC_CRT
    "MS Visual Studio: Use static C-Run Time Library in place of shared." OFF)

  IF (BUILD_TESTING AND BUILD_SHARED_LIBS)
    MESSAGE(
      "-- Disabling tests. The flags BUILD_TESTING and BUILD_SHARED_LIBS"
      " are incompatible with MSVC."
      )
    UPDATE_CACHE_VARIABLE(BUILD_TESTING OFF)
  ENDIF (BUILD_TESTING AND BUILD_SHARED_LIBS)
ENDIF (MSVC)

# Use ios-cmake to build a static library for iOS
#
# We need to add isysroot to force cmake to find the toolchains from the iOS SDK
# instead of using the standard ones. And add flag mios-simulator-version so clang
# knows we are building for ios simulator but not mac.
#
# You can build for OS (armv7, armv7s, arm64), SIMULATOR (i386) or SIMULATOR64 (x86_64)
# separately and use lipo to merge them into one static library.
#
# There are some features/algorithms are not available in iOS version and the
# minimum supported iOS version is 6.0 now.
#
# Use cmake ../ceres-solver -DCMAKE_TOOLCHAIN_FILE=../ceres-solver/cmake/iOS.cmake \
# -DIOS_PLATFORM=PLATFORM -DEIGEN_INCLUDE_DIR=/path/to/eigen/header
# to config the cmake. The PLATFORM can be one of OS, SIMULATOR and SIMULATOR64.
# Check the documentation in iOS.cmake to find more options.
#
# After building, you will get a single library: libceres.a, which
# you need to add to your Xcode project.
#
# If you use the lapack and blas, then you also need to add Accelerate.framework
# to your Xcode project's linking dependency.
IF (IOS)
  MESSAGE(STATUS "Building Ceres for iOS platform: ${IOS_PLATFORM}")

  UPDATE_CACHE_VARIABLE(MINIGLOG ON)
  MESSAGE(STATUS "Building for iOS, forcing use of miniglog instead of glog.")

  UPDATE_CACHE_VARIABLE(SUITESPARSE OFF)
  UPDATE_CACHE_VARIABLE(CXSPARSE OFF)
  UPDATE_CACHE_VARIABLE(GFLAGS OFF)
  UPDATE_CACHE_VARIABLE(OPENMP OFF)

  # Apple claims that the BLAS call dsyrk_ is a private API, and will not allow you
  # to submit to the Apple Store if the symbol is present.
  UPDATE_CACHE_VARIABLE(LAPACK OFF)

  MESSAGE(STATUS "Building for iOS: SuiteSparse, CXSparse, LAPACK, gflags, and OpenMP are not available.")

  UPDATE_CACHE_VARIABLE(BUILD_EXAMPLES OFF)
  MESSAGE(STATUS "Building for iOS, will not build examples.")

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-abi-version=2 -fobjc-arc -isysroot ${CMAKE_OSX_SYSROOT}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-abi-version=2 -fobjc-arc -isysroot ${CMAKE_OSX_SYSROOT}")

  IF (${IOS_PLATFORM} STREQUAL "SIMULATOR" OR ${IOS_PLATFORM} STREQUAL "SIMULATOR64")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mios-simulator-version-min=6.0")
  ENDIF()
ENDIF (IOS)

# Prior to October 2013, Ceres used some non-CMake standardised variables to
# hold user-specified (as opposed to FindPackage found) include directory and
# library paths for Ceres dependencies.  These were were of the form:
# <DEPENDENCY>_LIB / <DEPENDENCY>_INCLUDE.  Since then, Ceres now has
# FindPackage() scripts for all of its dependencies which obey the standard
# CMake variables: <DEPENDENCY>_LIBRARIES & <DEPENDENCY>_INCLUDE_DIRS.  In order
# to ensure backwards compatibility, convert any legacy variables to
# _directory_ hints for the FindPackage() scripts.
MACRO(HANDLE_LEGACY_INCLUDE_DEPENDENCY_HINT
    LEGACY_VAR DIRECTORY_HINT_VAR)
  IF (DEFINED ${LEGACY_VAR})
    # Get the dependency name (all caps) from the hint directory variable
    # for the warning message.
    STRING(REGEX MATCH "^[^_]*" DEPENDENCY_NAME ${DIRECTORY_HINT_VAR})
    MESSAGE(WARNING "You are defining a legacy variable ${LEGACY_VAR} "
      "to specify the include directory for ${DEPENDENCY_NAME}.  This is "
      "deprecated and support for it will be removed in a future release. "
      "Please use either the search directory hints variable: "
      "${DIRECTORY_HINT_VAR} or ${DEPENDENCY_NAME}_INCLUDE_DIR to specify "
      "exactly the directory used (no search performed), see: "
      "http://ceres-solver.org/building.html for more information.")
    LIST(APPEND ${DIRECTORY_HINT_VAR} ${${LEGACY_VAR}})
  ENDIF (DEFINED ${LEGACY_VAR})
ENDMACRO(HANDLE_LEGACY_INCLUDE_DEPENDENCY_HINT)

MACRO(HANDLE_LEGACY_LIBRARY_DEPENDENCY_HINT
    LEGACY_VAR DIRECTORY_HINT_VAR)
  IF (DEFINED ${LEGACY_VAR})
    # Get the dependency name (all caps) from the hint directory variable
    # for the warning message.
    STRING(REGEX MATCH "^[^_]*" DEPENDENCY_NAME ${DIRECTORY_HINT_VAR})
    MESSAGE(WARNING "You are defining a legacy variable ${LEGACY_VAR} "
      "to specify the library for ${DEPENDENCY_NAME}.  This is "
      "deprecated and support for it will be removed in a future release. "
      "Please use either the search directory hints variable: "
      "${DIRECTORY_HINT_VAR} or ${DEPENDENCY_NAME}_LIBRARY to specify "
      "exactly the library used (no search performed), see: "
      "http://ceres-solver.org/building.html for more information.")
    IF (EXISTS ${${LEGACY_VAR}} AND
        NOT IS_DIRECTORY ${${LEGACY_VAR}})
      # User specified an explicit (library) file using the legacy variable
      # interface, hints to FindPackage() scripts are directories so add the
      # parent directory of the specified file.
      GET_FILENAME_COMPONENT(DIR_HINT ${${LEGACY_VAR}} PATH)
      LIST(APPEND ${DIRECTORY_HINT_VAR} ${DIR_HINT})
    ELSEIF (EXISTS ${${LEGACY_VAR}} AND
            IS_DIRECTORY ${${LEGACY_VAR}})
      # User specified a directory hint using the legacy variable, use it.
      LIST(APPEND ${DIRECTORY_HINT_VAR} ${${LEGACY_VAR}})
    ENDIF()
  ENDIF (DEFINED ${LEGACY_VAR})
ENDMACRO(HANDLE_LEGACY_LIBRARY_DEPENDENCY_HINT)

UNSET(CERES_COMPILE_OPTIONS)

# Eigen.
HANDLE_LEGACY_INCLUDE_DEPENDENCY_HINT(EIGEN_INCLUDE EIGEN_INCLUDE_DIR_HINTS)
FIND_PACKAGE(Eigen REQUIRED)
IF (EIGEN_FOUND)
  IF (EIGEN_VERSION VERSION_LESS 3.1.0)
    MESSAGE(FATAL_ERROR "-- Ceres requires Eigen version >= 3.1.0 in order "
      "that Eigen/SparseCore be available, detected version of Eigen is: "
      "${EIGEN_VERSION}")
  ENDIF (EIGEN_VERSION VERSION_LESS 3.1.0)

  MESSAGE("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
  # Ensure that only MPL2 licensed code is part of the default build.
  MESSAGE("")
  MESSAGE("   ===============================================================")
  IF (EIGENSPARSE)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_USE_EIGEN_SPARSE)
    MESSAGE("   Enabling the use of Eigen as a sparse linear algebra library ")
    MESSAGE("   for solving the nonlinear least squares problems. Enabling ")
    MESSAGE("   this option will result in an LGPL licensed version of ")
    MESSAGE("   Ceres Solver as the Simplicial Cholesky factorization in Eigen")
    MESSAGE("   is licensed under the LGPL. ")

    IF (EIGEN_VERSION VERSION_LESS 3.2.2)
      MESSAGE("   WARNING:")
      MESSAGE("")
      MESSAGE("   Your version of Eigen is older than version 3.2.2.")
      MESSAGE("   The performance of SPARSE_NORMAL_CHOLESKY and SPARSE_SCHUR")
      MESSAGE("   linear solvers will suffer. ")
    ENDIF (EIGEN_VERSION VERSION_LESS 3.2.2)

  ELSE (EIGENSPARSE)
    MESSAGE("   Disabling the use of Eigen as a sparse linear algebra library.")
    MESSAGE("   This does not affect the covariance estimation algorithm ")
    MESSAGE("   which can still use the EIGEN_SPARSE_QR algorithm.")
    ADD_DEFINITIONS(-DEIGEN_MPL2_ONLY)
  ENDIF (EIGENSPARSE)
    MESSAGE("   ===============================================================")
    MESSAGE("")
ENDIF (EIGEN_FOUND)

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
    UPDATE_CACHE_VARIABLE(LAPACK OFF)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_LAPACK)
  ENDIF (NOT (LAPACK_FOUND AND BLAS_FOUND))
ELSE (LAPACK)
  MESSAGE("-- Building without LAPACK.")
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_LAPACK)
ENDIF (LAPACK)

# SuiteSparse.
IF (SUITESPARSE AND NOT LAPACK)
  # If user has disabled LAPACK, but left SUITESPARSE ON, turn it OFF,
  # LAPACK controls whether Ceres will be linked, directly or indirectly
  # via SuiteSparse to LAPACK.
  MESSAGE("-- Disabling SuiteSparse as use of LAPACK has been disabled, "
    "turn ON LAPACK to enable (optional) building with SuiteSparse.")
  UPDATE_CACHE_VARIABLE(SUITESPARSE OFF)
ENDIF (SUITESPARSE AND NOT LAPACK)
IF (SUITESPARSE)
  # By default, if SuiteSparse and all dependencies are found, Ceres is
  # built with SuiteSparse support.

  # Check for SuiteSparse and dependencies.
  FIND_PACKAGE(SuiteSparse)
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
        "see http://ceres-solver.org/building.html for more information.")
    ENDIF (BUILD_SHARED_LIBS AND
      SUITESPARSE_IS_BROKEN_SHARED_LINKING_UBUNTU_SYSTEM_VERSION)

    # By default, if all of SuiteSparse's dependencies are found, Ceres is
    # built with SuiteSparse support.
    MESSAGE("-- Found SuiteSparse ${SUITESPARSE_VERSION}, "
            "building with SuiteSparse.")
  ELSE (SUITESPARSE_FOUND)
    # Disable use of SuiteSparse if it cannot be found and continue.
    MESSAGE("-- Did not find all SuiteSparse dependencies, disabling "
      "SuiteSparse support.")
    UPDATE_CACHE_VARIABLE(SUITESPARSE OFF)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_SUITESPARSE)
  ENDIF (SUITESPARSE_FOUND)
ELSE (SUITESPARSE)
  MESSAGE("-- Building without SuiteSparse.")
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_SUITESPARSE)
ENDIF (SUITESPARSE)

# CXSparse.
IF (CXSPARSE)
  # Don't search with REQUIRED as we can continue without CXSparse.
  FIND_PACKAGE(CXSparse)
  IF (CXSPARSE_FOUND)
    # By default, if CXSparse and all dependencies are found, Ceres is
    # built with CXSparse support.
    MESSAGE("-- Found CXSparse version: ${CXSPARSE_VERSION}, "
      "building with CXSparse.")
  ELSE (CXSPARSE_FOUND)
    # Disable use of CXSparse if it cannot be found and continue.
    MESSAGE("-- Did not find CXSparse, Building without CXSparse.")
    UPDATE_CACHE_VARIABLE(CXSPARSE OFF)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_CXSPARSE)
  ENDIF (CXSPARSE_FOUND)
ELSE (CXSPARSE)
  MESSAGE("-- Building without CXSparse.")
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_CXSPARSE)
  # Mark as advanced (remove from default GUI view) the CXSparse search
  # variables in case user enabled CXSPARSE, FindCXSparse did not find it, so
  # made search variables visible in GUI for user to set, but then user disables
  # CXSPARSE instead of setting them.
  MARK_AS_ADVANCED(FORCE CXSPARSE_INCLUDE_DIR
                         CXSPARSE_LIBRARY)
ENDIF (CXSPARSE)

# Ensure that the user understands they have disabled all sparse libraries.
IF (NOT SUITESPARSE AND NOT CXSPARSE AND NOT EIGENSPARSE)
  MESSAGE("   ===============================================================")
  MESSAGE("   Compiling without any sparse library: SuiteSparse, CXSparse ")
  MESSAGE("   & Eigen (Sparse) are all disabled or unavailable.  No sparse ")
  MESSAGE("   linear solvers (SPARSE_NORMAL_CHOLESKY & SPARSE_SCHUR)")
  MESSAGE("   will be available when Ceres is used.")
  MESSAGE("   ===============================================================")
ENDIF(NOT SUITESPARSE AND NOT CXSPARSE AND NOT EIGENSPARSE)

# GFlags.
IF (GFLAGS)
  HANDLE_LEGACY_INCLUDE_DEPENDENCY_HINT(GFLAGS_INCLUDE GFLAGS_INCLUDE_DIR_HINTS)
  HANDLE_LEGACY_LIBRARY_DEPENDENCY_HINT(GFLAGS_LIB GFLAGS_LIBRARY_DIR_HINTS)

  # Don't search with REQUIRED as we can continue without gflags.
  FIND_PACKAGE(Gflags)
  IF (GFLAGS_FOUND)
    MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE_DIRS}, "
      "in namespace: ${GFLAGS_NAMESPACE}")
  ELSE (GFLAGS_FOUND)
    MESSAGE("-- Did not find Google Flags (gflags), Building without gflags "
      "- no tests or tools will be built!")
    UPDATE_CACHE_VARIABLE(GFLAGS OFF)
  ENDIF (GFLAGS_FOUND)
ELSE (GFLAGS)
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  # Mark as advanced (remove from default GUI view) the gflags search
  # variables in case user enabled GFLAGS, FindGflags did not find it, so
  # made search variables visible in GUI for user to set, but then user disables
  # GFLAGS instead of setting them.
  MARK_AS_ADVANCED(FORCE GFLAGS_INCLUDE_DIR
                         GFLAGS_LIBRARY
                         GFLAGS_NAMESPACE)
ENDIF (GFLAGS)

# MiniGLog.
IF (MINIGLOG)
  MESSAGE("-- Compiling minimal glog substitute into Ceres.")
  SET(GLOG_INCLUDE_DIRS internal/ceres/miniglog)
  MESSAGE("-- Using minimal glog substitute (include): ${GLOG_INCLUDE_DIRS}")

  # Mark as advanced (remove from default GUI view) the glog search
  # variables in case user disables MINIGLOG, FindGlog did not find it, so
  # made search variables visible in GUI for user to set, but then user enables
  # MINIGLOG instead of setting them.
  MARK_AS_ADVANCED(FORCE GLOG_INCLUDE_DIR
                         GLOG_LIBRARY)
ELSE (MINIGLOG)
  HANDLE_LEGACY_INCLUDE_DEPENDENCY_HINT(GLOG_INCLUDE GLOG_INCLUDE_DIR_HINTS)
  HANDLE_LEGACY_LIBRARY_DEPENDENCY_HINT(GLOG_LIB GLOG_LIBRARY_DIR_HINTS)

  # Don't search with REQUIRED so that configuration continues if not found and
  # we can output an error messages explaining MINIGLOG option.
  FIND_PACKAGE(Glog)
  IF (GLOG_FOUND)
    MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE_DIRS}")
  ELSE (GLOG_FOUND)
    MESSAGE(FATAL_ERROR "Can't find Google Log. Please set GLOG_INCLUDE_DIR & "
      "GLOG_LIBRARY or enable MINIGLOG option to use minimal glog "
      "implementation.")
  ENDIF (GLOG_FOUND)
ENDIF (MINIGLOG)

IF (NOT SCHUR_SPECIALIZATIONS)
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_RESTRICT_SCHUR_SPECIALIZATION)
  MESSAGE("-- Disabling Schur specializations (faster compiles)")
ENDIF (NOT SCHUR_SPECIALIZATIONS)

IF (NOT CUSTOM_BLAS)
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_CUSTOM_BLAS)
  MESSAGE("-- Disabling custom blas")
ENDIF (NOT CUSTOM_BLAS)

IF (OPENMP)
  # Clang does not (yet) support OpenMP.
  IF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    UPDATE_CACHE_VARIABLE(OPENMP OFF)
    MESSAGE("-- Compiler is Clang, disabling OpenMP.")
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
  ELSE (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Find quietly s/t as we can continue without OpenMP if it is not found.
    FIND_PACKAGE(OpenMP QUIET)
    IF (OPENMP_FOUND)
      MESSAGE("-- Building with OpenMP.")
      LIST(APPEND CERES_COMPILE_OPTIONS CERES_USE_OPENMP)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      IF (UNIX)
        # At least on Linux, we need pthreads to be enabled for mutex to
        # compile.  This may not work on Windows or Android.
        FIND_PACKAGE(Threads REQUIRED)
        LIST(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_PTHREAD)
        LIST(APPEND CERES_COMPILE_OPTIONS CERES_HAVE_RWLOCK)
      ENDIF (UNIX)
    ELSE (OPENMP_FOUND)
      MESSAGE("-- Failed to find OpenMP, disabling.")
      UPDATE_CACHE_VARIABLE(OPENMP OFF)
      LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
    ENDIF (OPENMP_FOUND)
  ENDIF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
ELSE (OPENMP)
  MESSAGE("-- Building without OpenMP (disabling multithreading).")
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_THREADS)
ENDIF (OPENMP)

# MSVC supports (in some sense) C++11, but does not use the -std=c++11 flag.
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_HAS_CXX11_FLAG)
IF (CXX11)
  # For some compilers (at least GCC 4.8), shared_ptr & unordered_map exist in
  # two places, as the TR1 versions are still present.  In order to avoid any
  # conflicts in user code linking against Ceres which uses C++11, we enable
  # C++11 (which is required when using the non-TR1 versions) if it is available
  # before searching for shared_ptr & unordered_map.
  IF (COMPILER_HAS_CXX11_FLAG)
    # CMAKE_REQUIRED_FLAGS is used by CheckCXXSourceCompiles.
    SET(CMAKE_REQUIRED_FLAGS -std=c++11)
  ENDIF (COMPILER_HAS_CXX11_FLAG)
ENDIF (CXX11)

INCLUDE(FindUnorderedMap)
FIND_UNORDERED_MAP()
IF (UNORDERED_MAP_FOUND)
  IF (HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_STD_UNORDERED_MAP)
  ENDIF(HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
  IF (HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_STD_UNORDERED_MAP_IN_TR1_NAMESPACE)
  ENDIF(HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
  IF (HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_TR1_UNORDERED_MAP)
  ENDIF(HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
ELSE (UNORDERED_MAP_FOUND)
  MESSAGE("-- Replacing unordered_map/set with map/set (warning: slower!), "
    "try enabling CXX11 option if you expect C++11 to be available.")
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_NO_UNORDERED_MAP)
ENDIF()

INCLUDE(FindSharedPtr)
FIND_SHARED_PTR()
IF (SHARED_PTR_FOUND)
  IF (SHARED_PTR_TR1_MEMORY_HEADER)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_TR1_MEMORY_HEADER)
  ENDIF (SHARED_PTR_TR1_MEMORY_HEADER)
  IF (SHARED_PTR_TR1_NAMESPACE)
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_TR1_SHARED_PTR)
  ENDIF (SHARED_PTR_TR1_NAMESPACE)
ELSE (SHARED_PTR_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find shared_ptr, try enabling CXX11 option "
    "if you expect C++11 to be available.")
ENDIF (SHARED_PTR_FOUND)

# To ensure that CXX11 accurately captures whether we are using C++11, even on
# MSVC, check if it is required given where the potentially C++11 features Ceres
# uses were found, and disable it if C++11 is not being used.
IF (CXX11)
  IF (NOT HAVE_SHARED_PTR_IN_STD_NAMESPACE AND
      NOT HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
    MESSAGE("-- Failed to find C++11 components in C++11 locations & "
      "namespaces, disabling CXX11.")
    UPDATE_CACHE_VARIABLE(CXX11 OFF)
  ELSE()
    MESSAGE("   ==============================================================")
    MESSAGE("   Compiling Ceres using C++11.  This will result in a version ")
    MESSAGE("   of Ceres that will require the use of C++11 in client code.")
    MESSAGE("   ==============================================================")
    LIST(APPEND CERES_COMPILE_OPTIONS CERES_USE_CXX11)
    IF (COMPILER_HAS_CXX11_FLAG)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    ENDIF()
  ENDIF()
ENDIF(CXX11)

INCLUDE_DIRECTORIES(
  include
  internal
  internal/ceres
  ${GLOG_INCLUDE_DIRS})
# Eigen SparseQR generates various compiler warnings related to unused and
# uninitialised local variables, which prevents Ceres compilation as we use
# -Werror.  To avoid having to individually suppress these warnings around
# the #include statments for Eigen headers across all GCC/Clang versions, we
# tell CMake to treat Eigen headers as system headers.  This results in all
# compiler warnings from them being suppressed.
#
# Note that this is *not* propagated to clients, ie CERES_INCLUDE_DIRS
# used by clients after find_package(Ceres) does not identify Eigen as
# as system headers.
INCLUDE_DIRECTORIES(SYSTEM ${EIGEN_INCLUDE_DIRS})

IF (SUITESPARSE)
  INCLUDE_DIRECTORIES(${SUITESPARSE_INCLUDE_DIRS})
ENDIF (SUITESPARSE)

IF (CXSPARSE)
  INCLUDE_DIRECTORIES(${CXSPARSE_INCLUDE_DIRS})
ENDIF (CXSPARSE)

IF (GFLAGS)
  INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIRS})
ENDIF (GFLAGS)

IF (BUILD_SHARED_LIBS)
  MESSAGE("-- Building Ceres as a shared library.")
  # The CERES_BUILDING_SHARED_LIBRARY compile definition is NOT stored in
  # CERES_COMPILE_OPTIONS as it must only be defined when Ceres is compiled
  # not when it is used as it controls the CERES_EXPORT macro which provides
  # dllimport/export support in MSVC.
  ADD_DEFINITIONS(-DCERES_BUILDING_SHARED_LIBRARY)
  LIST(APPEND CERES_COMPILE_OPTIONS CERES_USING_SHARED_LIBRARY)
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
    # Use of -flto requires use of gold linker & LLVM-gold plugin, which might
    # well not be present / in use and without which files will compile, but
    # not link ('file not recognized') so explicitly check for support
    INCLUDE(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-flto" HAVE_LTO_SUPPORT)
    IF (HAVE_LTO_SUPPORT)
      MESSAGE(STATUS "Enabling link-time optimization (-flto)")
      SET(CERES_CXX_FLAGS "${CERES_CXX_FLAGS} -flto")
    ELSE ()
      MESSAGE(STATUS "Compiler/linker does not support link-time optimization (-flto), disabling.")
    ENDIF (HAVE_LTO_SUPPORT)
  ENDIF ()
ENDIF (CMAKE_BUILD_TYPE STREQUAL "Release")

SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CERES_CXX_FLAGS}")

IF (MINGW)
  # MinGW produces code that segfaults when performing matrix multiplications
  # in Eigen when compiled with -O3 (see [1]), as such force the use of -O2
  # which works.
  #
  # [1] http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556
  MESSAGE("-- MinGW detected, forcing -O2 instead of -O3 in Release for Eigen due "
          "to a MinGW bug: http://eigen.tuxfamily.org/bz/show_bug.cgi?id=556")
  STRING(REPLACE "-O3" "-O2" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  UPDATE_CACHE_VARIABLE(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
ENDIF (MINGW)

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

  # Update the C/CXX flags for MSVC to use either the static or shared
  # C-Run Time (CRT) library based on the user option: MSVC_USE_STATIC_CRT.
  LIST(APPEND C_CXX_FLAGS
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_MINSIZEREL
    CMAKE_CXX_FLAGS_RELWITHDEBINFO)

  FOREACH(FLAG_VAR ${C_CXX_FLAGS})
    IF (MSVC_USE_STATIC_CRT)
      # Use static CRT.
      IF (${FLAG_VAR} MATCHES "/MD")
        STRING(REGEX REPLACE "/MD" "/MT" ${FLAG_VAR} "${${FLAG_VAR}}")
      ENDIF (${FLAG_VAR} MATCHES "/MD")
    ELSE (MSVC_USE_STATIC_CRT)
      # Use shared, not static, CRT.
      IF (${FLAG_VAR} MATCHES "/MT")
        STRING(REGEX REPLACE "/MT" "/MD" ${FLAG_VAR} "${${FLAG_VAR}}")
      ENDIF (${FLAG_VAR} MATCHES "/MT")
    ENDIF (MSVC_USE_STATIC_CRT)
  ENDFOREACH()

  # Tuple sizes of 10 are used by Gtest.
  ADD_DEFINITIONS("-D_VARIADIC_MAX=10")
ENDIF (MSVC)

IF (UNIX)
  # GCC is not strict enough by default, so enable most of the warnings.
  SET(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Werror=all -Werror=extra -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers")
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
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-Wno-return-type-c-linkage"
                          HAVE_RETURN_TYPE_C_LINKAGE)
  IF (HAVE_RETURN_TYPE_C_LINKAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  ENDIF(HAVE_RETURN_TYPE_C_LINKAGE)
ENDIF ()

# Xcode 4.5.x used Clang 4.1 (Apple version), this has a bug that prevents
# compilation of Ceres.
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

# Configure the Ceres config.h compile options header using the current
# compile options and put the configured header into the Ceres build
# directory.  Note that the ceres/internal subdir in <build>/config where
# the configured config.h is placed is important, because Ceres will be
# built against this configured header, it needs to have the same relative
# include path as it would if it were in the source tree (or installed).
LIST(REMOVE_DUPLICATES CERES_COMPILE_OPTIONS)
INCLUDE(CreateCeresConfig)
CREATE_CERES_CONFIG("${CERES_COMPILE_OPTIONS}"
  ${CMAKE_CURRENT_BINARY_DIR}/config/ceres/internal)
# Force the location containing the configured config.h to the front of the
# include_directories list (by default it is appended to the back) to ensure
# that if the user has an installed version of Ceres in the same location as one
# of the dependencies (e.g. /usr/local) that we find the config.h we just
# configured, not the (older) installed config.h.
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/config)

ADD_SUBDIRECTORY(internal/ceres)

IF (BUILD_DOCUMENTATION)
  FIND_PACKAGE(Sphinx QUIET)
  IF (NOT SPHINX_FOUND)
    MESSAGE("-- Failed to find Sphinx, disabling build of documentation.")
    UPDATE_CACHE_VARIABLE(BUILD_DOCUMENTATION OFF)
  ELSE()
    # Generate the User's Guide (html).
    # The corresponding target is ceres_docs, but is included in ALL.
    MESSAGE("-- Build the HTML documentation.")
    ADD_SUBDIRECTORY(docs)
  ENDIF()
ENDIF (BUILD_DOCUMENTATION)

IF (BUILD_EXAMPLES)
  MESSAGE("-- Build the examples.")
  ADD_SUBDIRECTORY(examples)
ELSE (BUILD_EXAMPLES)
  MESSAGE("-- Do not build any example.")
ENDIF (BUILD_EXAMPLES)

# Setup installation of Ceres public headers.
FILE(GLOB CERES_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/*.h)
INSTALL(FILES ${CERES_HDRS} DESTINATION include/ceres)

FILE(GLOB CERES_PUBLIC_INTERNAL_HDRS ${CMAKE_SOURCE_DIR}/include/ceres/internal/*.h)
INSTALL(FILES ${CERES_PUBLIC_INTERNAL_HDRS} DESTINATION include/ceres/internal)

# Also setup installation of Ceres config.h configured with the current
# build options into the installed headers directory.
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/config/ceres/internal/config.h
        DESTINATION include/ceres/internal)


IF (MINIGLOG)
  # Install miniglog header if being used as logging #includes appear in
  # installed public Ceres headers.
  INSTALL(FILES ${CMAKE_SOURCE_DIR}/internal/ceres/miniglog/glog/logging.h
          DESTINATION include/ceres/internal/miniglog/glog)
ENDIF (MINIGLOG)

# Add an uninstall target to remove all installed files.
CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/cmake/uninstall.cmake.in"
               "${CMAKE_BINARY_DIR}/cmake/uninstall.cmake"
               IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
                  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake/uninstall.cmake)

# Set relative install paths, which are appended to CMAKE_INSTALL_PREFIX to
# generate the absolute install paths.
IF (WIN32)
  SET(RELATIVE_CMAKECONFIG_INSTALL_DIR CMake)
ELSE ()
  SET(RELATIVE_CMAKECONFIG_INSTALL_DIR share/Ceres)
ENDIF ()

# This "exports" all targets which have been put into the export set
# "CeresExport". This means that CMake generates a file with the given
# filename, which can later on be loaded by projects using this package.
# This file contains ADD_LIBRARY(bar IMPORTED) statements for each target
# in the export set, so when loaded later on CMake will create "imported"
# library targets from these, which can be used in many ways in the same way
# as a normal library target created via a normal ADD_LIBRARY().
INSTALL(EXPORT CeresExport
        DESTINATION ${RELATIVE_CMAKECONFIG_INSTALL_DIR} FILE CeresTargets.cmake)

# Figure out the relative path from the installed Config.cmake file to the
# install prefix (which may be at runtime different from the chosen
# CMAKE_INSTALL_PREFIX if under Windows the package was installed anywhere)
# This relative path will be configured into the CeresConfig.cmake.
FILE(RELATIVE_PATH INSTALL_ROOT_REL_CONFIG_INSTALL_DIR
     ${CMAKE_INSTALL_PREFIX}/${RELATIVE_CMAKECONFIG_INSTALL_DIR}
     ${CMAKE_INSTALL_PREFIX})

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

# Install these files into the same directory as the generated exports-file,
# we include the FindPackage scripts for libraries whose headers are included
# in the public API of Ceres and should thus be present in CERES_INCLUDE_DIRS.
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/CeresConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CeresConfigVersion.cmake"
              "${CMAKE_SOURCE_DIR}/cmake/FindEigen.cmake"
              "${CMAKE_SOURCE_DIR}/cmake/FindGlog.cmake"
        DESTINATION ${RELATIVE_CMAKECONFIG_INSTALL_DIR})
