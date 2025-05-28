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

#ifndef CERES_PUBLIC_INTERNAL_COMPENSATED_MATH_H_
#define CERES_PUBLIC_INTERNAL_COMPENSATED_MATH_H_

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace ceres::internal {

// Compute two values s, t that satisfy s + t = x + y exactly where s is the sum
// nearest to x + y and t is the round-off error. The algorithm assumes the
// round-to-nearest mode and |x| >= |y|.
template <typename T>
constexpr auto Fast2Sum(T x, T y)
    -> std::enable_if_t<std::is_floating_point_v<T>, std::pair<T, T>> {
  static_assert(std::numeric_limits<T>::radix <= 3,
                "Fast2Sum supports only radix 2 and 3 floating-point types");
  using std::fabs;
  using std::isgreaterequal;
  assert(isgreaterequal(fabs(x), fabs(y)));
  const T s = x + y;
  const T z = s - x;
  const T t = y - z;
  return std::make_pair(s, t);
}

// Similar to Fast2Sum, but without requiring ordering or a specific radix.
template <typename T>
constexpr auto TwoSum(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>, std::pair<T, T>> {
  const T s = a + b;
  const T a_prime = s - b;
  const T b_prime = s - a_prime;
  const T delta_a = a - a_prime;
  const T delta_b = b - b_prime;
  const T t = delta_a + delta_b;
  return std::make_pair(s, t);
}

template <typename T>
constexpr auto SumWithError(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            std::numeric_limits<T>::radix <= 3,
                        std::pair<T, T>> {
  return Fast2Sum(a, b);
}

template <typename T>
constexpr auto SumWithError(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            (std::numeric_limits<T>::radix > 3),
                        std::pair<T, T>> {
  return TwoSum(a, b);
}

// Computes the round-off error of x * y when xy is already available.
template <typename T>
constexpr auto TwoMultFMA(T x, T y, T xy)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::fpclassify;
  assert(fpclassify((x * y) - xy) == FP_ZERO);
  return fma(x, y, -xy);
}

}  // namespace ceres::internal

#endif  // CERES_PUBLIC_INTERNAL_COMPENSATED_MATH_H_
