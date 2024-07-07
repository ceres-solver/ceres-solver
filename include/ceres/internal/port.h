// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2024 Google Inc. All rights reserved.
// http://ceres-solver.org/
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
// Author: keir@google.com (Keir Mierle)

#ifndef CERES_PUBLIC_INTERNAL_PORT_H_
#define CERES_PUBLIC_INTERNAL_PORT_H_

#include <cmath>  // Necessary for __cpp_lib_math_special_functions feature test

// A macro to mark a function/variable/class as deprecated.
// We use compiler specific attributes rather than the c++
// attribute because they do not mix well with each other.
#if defined(_MSC_VER)
#define CERES_DEPRECATED_WITH_MSG(message) __declspec(deprecated(message))
#elif defined(__GNUC__)
#define CERES_DEPRECATED_WITH_MSG(message) __attribute__((deprecated(message)))
#else
// In the worst case fall back to c++ attribute.
#define CERES_DEPRECATED_WITH_MSG(message) [[deprecated(message)]]
#endif

// Indicates whether C++20 is currently active
#ifndef CERES_HAS_CPP20
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
#define CERES_HAS_CPP20
#endif  // __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >=
        // 202002L)
#endif  // !defined(CERES_HAS_CPP20)

// Prevents symbols from being substituted by the corresponding macro definition
// under the same name. For instance, min and max are defined as macros on
// Windows (unless NOMINMAX is defined) which causes compilation errors when
// defining or referencing symbols under the same name.
//
// To be robust in all cases particularly when NOMINMAX cannot be used, use this
// macro to annotate min/max declarations/definitions. Examples:
//
//   int max CERES_PREVENT_MACRO_SUBSTITUTION();
//   min CERES_PREVENT_MACRO_SUBSTITUTION(a, b);
//   max CERES_PREVENT_MACRO_SUBSTITUTION(a, b);
//
// NOTE: In case the symbols for which the substitution must be prevented are
// used within another macro, the substitution must be inhibited using parens as
//
//   (std::numerical_limits<double>::max)()
//
// since the helper macro will not work here. Do not use this technique in
// general case, because it will prevent argument-dependent lookup (ADL).
//
#define CERES_PREVENT_MACRO_SUBSTITUTION  // Yes, it's empty

// CERES_DISABLE_DEPRECATED_WARNING and CERES_RESTORE_DEPRECATED_WARNING allow
// to temporarily disable deprecation warnings
#if defined(_MSC_VER)
#define CERES_DISABLE_DEPRECATED_WARNING \
  _Pragma("warning(push)") _Pragma("warning(disable : 4996)")
#define CERES_RESTORE_DEPRECATED_WARNING _Pragma("warning(pop)")
#else  // defined(_MSC_VER)
#define CERES_DISABLE_DEPRECATED_WARNING
#define CERES_RESTORE_DEPRECATED_WARNING
#endif  // defined(_MSC_VER)

#if defined(__cpp_lib_math_special_functions) &&      \
    ((__cpp_lib_math_special_functions >= 201603L) || \
     defined(__STDCPP_MATH_SPEC_FUNCS__) &&           \
         (__STDCPP_MATH_SPEC_FUNCS__ >= 201003L))
// If defined, indicates whether C++17 Bessel functions (of the first kind) are
// available. Some standard library implementations, such as libc++ (Android
// NDK, Apple, Clang) do not yet provide these functions. Implementations that
// do not support C++17, but support ISO 29124:2010, provide the functions if
// __STDCPP_MATH_SPEC_FUNCS__ is defined by the implementation to a value at
// least 201003L and if the user defines __STDCPP_WANT_MATH_SPEC_FUNCS__ before
// including any standard library headers. Standard library Bessel functions are
// preferred over any other implementation.
#define CERES_HAS_CPP17_BESSEL_FUNCTIONS
#elif defined(_SVID_SOURCE) || defined(_BSD_SOURCE) || defined(_XOPEN_SOURCE)
// If defined, indicates that j0, j1, and jn from <math.h> are available.
#define CERES_HAS_POSIX_BESSEL_FUNCTIONS
#endif

#endif  // CERES_PUBLIC_INTERNAL_PORT_H_
