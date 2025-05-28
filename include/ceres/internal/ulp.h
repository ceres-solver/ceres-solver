// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
// Author: sergiu.deitsch@gmail.com (Sergiu Deitsch)

#ifndef CERES_PUBLIC_INTERNAL_ULP_H_
#define CERES_PUBLIC_INTERNAL_ULP_H_

#include <cmath>
#include <limits>
#include <type_traits>

namespace ceres::internal {

// Determines the unit in the last place (ulp) of a value x. ulp is the spacing
// between consecutive floating-point numbers. We follow Goldberg's definition
// of the function given by
//
//   ulp(x) = 𝛽^(max{e,e_min}−p+1)
//
// for a floating-point type with radix 𝛽, precision p, and integral exponent e.
template <typename T>
constexpr auto Ulp(T x) -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fpclassify;
  using std::ilogb;
  using std::scalbn;

  const int cls = fpclassify(x);

  if (cls == FP_NAN) {
    return x;
  }

  if (cls == FP_INFINITE) {
    return std::numeric_limits<T>::infinity();
  }

  if (cls == FP_NORMAL) {
    return scalbn(std::numeric_limits<T>::epsilon(), ilogb(x));
  }

  return std::numeric_limits<T>::min();
}

template <typename T>
constexpr auto Ulp(T x) -> std::enable_if_t<std::is_integral_v<T>, double> {
  return Ulp(static_cast<double>(x));
}

}  // namespace ceres::internal

#endif  // CERES_PUBLIC_INTERNAL_ULP_H_
