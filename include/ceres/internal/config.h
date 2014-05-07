// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: alexs.mac@gmail.com (Alex Stewart)

// Default configuration options for Ceres.
//
// IMPORTANT: Most users of Ceres will not use this file, when compiling Ceres
//            with CMake, CMake will configure a new config.h with the currently
//            selected Ceres compile options and copy it into the source
//            directory before compilation.  However, for some users of Ceres
//            who compile without CMake, this file provides a default set of
//            options.

#ifndef CERES_PUBLIC_INTERNAL_CONFIG_H_
#define CERES_PUBLIC_INTERNAL_CONFIG_H_

// If defined, Ceres was compiled without LAPACK.
// #define CERES_NO_LAPACK

// If defined, Ceres was compiled without SuiteSparse.
// #define CERES_NO_SUITESPARSE

// If defined, Ceres was compiled without CXSparse.
// #define CERES_NO_CXSPARSE

// If defined, Ceres was compiled without Schur specializations.
// #define CERES_RESTRICT_SCHUR_SPECIALIZATION

// If defined, Ceres was compiled to use Eigen instead of hardcoded BLAS
// routines.
// #define CERES_NO_CUSTOM_BLAS

// If defined, Ceres was compiled without multithreading support.
// #define CERES_NO_THREADS
// If defined Ceres was compiled with OpenMP multithreading support.
#define CERES_USE_OPENMP
// Additionally defined on *nix if Ceres was compiled with OpenMP support,
// as in this case pthreads is also required.
#define CERES_HAVE_PTHREAD
#define CERES_HAVE_RWLOCK

// Which version of unordered map was used when Ceres was compiled. Exactly
// one of these will be defined for any given build.
#define CERES_STD_UNORDERED_MAP
// #define CERES_STD_UNORDERED_MAP_IN_TR1_NAMESPACE
// #define CERES_TR1_UNORDERED_MAP
// #define CERES_NO_UNORDERED_MAP

// If defined Ceres was compiled for Android with noalias() removed from
// matrix-matrix multiplies to work around a bug in the Android NDK.
// #define CERES_WORK_AROUND_ANDROID_NDK_COMPILER_BUG

// If defined, the memory header is in <tr1/memory>, otherwise <memory>.
// #define CERES_TR1_MEMORY_HEADER

// If defined shared_ptr is in std::tr1 namespace, otherwise std.
// #define CERES_TR1_SHARED_PTR

// If defined, Ceres was built as a shared library.
// #define CERES_USING_SHARED_LIBRARY

#endif  // CERES_PUBLIC_INTERNAL_CONFIG_H_
