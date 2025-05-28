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
//
// This header implements a function for accurately computing the 2-argument
// hypotenuse while avoiding under- and overflows and its reciprocal variant
// along with their corresponding variadic versions. The latter use a
// composition of the former 2-argument hypotenuse function.
//
// The implementation is derived from the following two papers:
//
// [1] Borges, C. F. (2021). Fast Compensated Algorithms for the Reciprocal
//     Square Root, the Reciprocal Hypotenuse, and Givens Rotations.
//     http://arxiv.org/abs/2103.08694
//
// [2] Borges, C. F. (2021). Algorithm 1014: An Improved Algorithm for
//     hypot(x,y). ACM Transactions on Mathematical Software, 47(1), 1–12.
//     https://doi.org/10.1145/3428446

#ifndef CERES_PUBLIC_ACCURATE_NORM_
#define CERES_PUBLIC_ACCURATE_NORM_

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>

#include "ceres/constants.h"
#include "ceres/internal/compensated_math.h"
#include "ceres/internal/ulp.h"

namespace ceres {

namespace internal {

// Helper trait to promote integral types to double and keep floating-point
// types unchanged.
template <typename T, typename Enable = void>
struct Promote {};

template <typename T>
struct Promote<T, std::enable_if_t<std::is_integral_v<T>>> {
  // The canonical floating-point type for integral inputs.
  using type = double;
};

template <typename T>
struct Promote<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  // Type identity.
  using type = T;
};

template <typename... Ts>
using Promote_t = decltype((typename Promote<Ts>::type(0) + ... + 0));

template <typename T, typename Enable = void>
struct AccurateNormTraits {
  // Ratio below which the smaller argument can be ignored.
  // √(ε/2) <=> 1/(√2)·𝛽^((1-p)/2) <=> (√2)/2·𝛽^((1-p)/2)
  static T CutoffRatio() noexcept {
    using std::scalbn;
    return scalbn(constants::sqrt_2_v<T> / T{2},
                  (1 - std::numeric_limits<T>::digits) / 2);
  }

  // √(F_max/2) <=> 1/(√2)·𝛽^(e_max/2) <=> (√2)/2·𝛽^(e_max/2)
  static T Huge() noexcept {
    using std::scalbn;
    return scalbn(constants::sqrt_2_v<T> / T{2},
                  std::numeric_limits<T>::max_exponent / 2);
  }

  // √(F_min) <=> 𝛽^(e_min/2)
  static T Tiny() noexcept {
    using std::scalbn;
    return scalbn(T{1}, (std::numeric_limits<T>::min_exponent - 1) / 2);
  }

  // ulp(√F_min)
  static T Scale() noexcept { return Ulp(Tiny()); }
};

// Computes the sum of squares x^2 + y^2 and its corresponding rounding error.
template <typename T>
constexpr auto UnscaledAccurateSquareNormWithError(T x, T y)
    -> std::enable_if_t<std::is_floating_point_v<T>, std::pair<T, T>> {
  using std::fma;

  const T x_sq = x * x;
  const T y_sq = y * y;

  // Use a radix-independent error-free transform to add the squares.
  const auto [sigma, sigma_e] = TwoSum(x_sq, y_sq);

  // Use 2MultFMA to recover the rounding errors from squaring x and y.
  const T x_error = fma(x, x, -x_sq);
  const T y_error = fma(y, y, -y_sq);
  const auto [square_error, square_error_e] = TwoSum(x_error, y_error);
  const auto [error, error_e] = TwoSum(sigma_e, square_error);
  return std::make_pair(sigma, error + error_e + square_error_e);
}

// Adds the square of x and its rounding error to an accumulated square norm.
template <typename T, typename... Args>
constexpr auto AccumulateSquareNormWithError(T sigma,
                                             T sigma_e,
                                             T x,
                                             Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, std::pair<T, T>> {
  using std::fma;

  const T x_sq = x * x;
  const auto [sum, sum_e] = TwoSum(sigma, x_sq);
  // Use 2MultFMA to recover the rounding error from squaring x.
  const T x_error = fma(x, x, -x_sq);
  const auto [error, error_e] = TwoSum(sigma_e, sum_e);
  const auto [compensated_error, compensated_error_e] = TwoSum(error, x_error);
  const T total_error = compensated_error + error_e + compensated_error_e;

  if constexpr (sizeof...(Args) == 0) {
    return std::make_pair(sum, total_error);
  } else {
    return AccumulateSquareNormWithError(sum, total_error, args...);
  }
}

template <typename T, typename... Args>
constexpr auto UnscaledAccurateSquareNormWithError(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, std::pair<T, T>> {
  const auto [sigma, sigma_e] = UnscaledAccurateSquareNormWithError(x, y);
  return AccumulateSquareNormWithError(sigma, sigma_e, args...);
}

// Computes sqrt(x^2 + y^2) and its round-off error without checking the
// arguments and ensuring invariants. Not intended to be invoked by users.
//
// The functions assumes the arguments to be finite, scaled correctly to avoid
// an under-/overflow and passed in the decreasing order of magnitude, i.e.,
// |x| ≥ |y|.
template <typename T>
constexpr auto UnscaledAccurateNorm(T x, T y)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  const auto [sigma, sigma_e] = UnscaledAccurateSquareNormWithError(x, y);
  const T h = sqrt(sigma);
  const T tau = sigma_e + fma(-h, h, sigma);
  return fma(tau / h, T(0.5), h);
}

template <typename T, typename... Args>
constexpr auto UnscaledAccurateNorm(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  const auto [sigma, sigma_e] =
      UnscaledAccurateSquareNormWithError(x, y, args...);
  const T h = sqrt(sigma);
  const T tau = sigma_e + fma(-h, h, sigma);
  return fma(tau / h, T(0.5), h);
}

// Computes the reciprocal of sqrt(x^2 + y^2) and its round-off error without
// checking the arguments and ensuring invariants. Not intended to be invoked
// by users.
template <typename T>
constexpr auto UnscaledAccurateRNorm(T x, T y)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  auto [sigma, sigma_e] = UnscaledAccurateSquareNormWithError(x, y);
  const T r = T(1) / sigma;
  sigma = fma(-r, sigma_e, fma(-r, sigma, T(1)));
  const T rho = sqrt(r);
  const T tau = fma(-rho, rho, r);
  const T nu = fma(sigma, tau / r, sigma) / 2;
  return fma(rho, nu, rho);
}

template <typename T, typename... Args>
constexpr auto UnscaledAccurateRNorm(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  auto [sigma, sigma_e] = UnscaledAccurateSquareNormWithError(x, y, args...);
  const T r = T(1) / sigma;
  sigma = fma(-r, sigma_e, fma(-r, sigma, T(1)));
  const T rho = sqrt(r);
  const T tau = fma(-rho, rho, r);
  const T nu = fma(sigma, tau / r, sigma) / 2;
  return fma(rho, nu, rho);
}

}  // namespace internal

// Computes the Euclidean norm of two values while avoiding intermediate
// underflow and overflow.
template <typename T>
constexpr auto AccurateNorm(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fabs;
  using std::isinf;
  using std::isnan;

  a = fabs(a);
  b = fabs(b);

  if (isinf(a)) {
    return a;
  }

  if (isinf(b)) {
    return b;
  }

  // Preserve NaN inputs after giving infinities precedence.
  if (isnan(a)) {
    return a;
  }

  if (isnan(b)) {
    return b;
  }

  // Ensure |x| ≥ |y|. No exception can be raised here since the arguments are
  // finite at this point. The same is true for the following comparisons.
  const auto [y, x] = std::minmax(a, b);

  using internal::AccurateNormTraits;

  if (y <= x * AccurateNormTraits<T>::CutoffRatio()) {
    return x;
  }

  using internal::UnscaledAccurateNorm;

  const T scale = AccurateNormTraits<T>::Scale();

  if (x > AccurateNormTraits<T>::Huge()) {
    // Scale x to prevent an overflow
    return UnscaledAccurateNorm(x * scale, y * scale) / scale;
  }

  if (y < AccurateNormTraits<T>::Tiny()) {
    // Scale y to prevent an underflow
    return UnscaledAccurateNorm(x / scale, y / scale) * scale;
  }

  // Avoid rounding errors due to unnecessary scaling
  return UnscaledAccurateNorm(x, y);
}

// Computes the Euclidean norm of three or more values of the same type while
// avoiding intermediate underflow and overflow.
template <typename T, typename... Args>
constexpr auto AccurateNorm(T a, T b, Args&&... args)
    -> std::enable_if_t<(sizeof...(Args) > 0 &&
                         (std::is_same_v<T, std::decay_t<Args>> && ...)),
                        internal::Promote_t<T>> {
  using Type = internal::Promote_t<T>;
  using std::fabs;
  using std::fmax;
  using std::fpclassify;
  using std::ilogb;
  using std::isinf;
  using std::isnan;
  using std::scalbn;

  const Type first = Type(a);
  const Type second = Type(b);
  const std::initializer_list<Type> values = {
      first, second, Type(std::forward<Args>(args))...};

  const auto infinity = std::find_if(
      values.begin(), values.end(), [](Type value) { return isinf(value); });

  if (infinity != values.end()) {
    return fabs(*infinity);
  }

  // Preserve NaN inputs after giving infinities precedence.
  const auto nan = std::find_if(
      values.begin(), values.end(), [](Type value) { return isnan(value); });

  if (nan != values.end()) {
    return *nan;
  }

  const Type first_magnitude = fabs(first);
  const Type second_magnitude = fabs(second);
  const std::initializer_list<Type> magnitudes = {
      first_magnitude,
      second_magnitude,
      fabs(Type(std::forward<Args>(args)))...};

  // Unlike the two-argument overload, normalize by a radix power near the
  // largest magnitude before accumulating the compensated squares. This bounds
  // every square independently of the number of arguments, avoids intermediate
  // overflow and underflow, and requires one square root.
  const Type maximum = std::accumulate(
      magnitudes.begin(), magnitudes.end(), Type(0), [](Type lhs, Type rhs) {
        return fmax(lhs, rhs);
      });

  if (fpclassify(maximum) == FP_ZERO) {
    return Type(0);
  }

  const int exponent = ilogb(maximum);
  // Scale by a radix power to avoid rounding by an arbitrary maximum.

  using internal::UnscaledAccurateNorm;

  const Type normalized_first = scalbn(first_magnitude, -exponent);
  const Type normalized_second = scalbn(second_magnitude, -exponent);
  const auto [y, x] = std::minmax(normalized_first, normalized_second);

  return scalbn(
      UnscaledAccurateNorm(
          x, y, (scalbn(Type(std::forward<Args>(args)), -exponent))...),
      exponent);
}

// Computes the Euclidean norm of two or more arithmetic values after promoting
// all arguments to a common floating-point type.
template <typename T, typename U, typename... Args>
constexpr internal::Promote_t<T, U, Args...> AccurateNorm(T a,
                                                          U b,
                                                          Args&&... args) {
  using Type = internal::Promote_t<T, U, Args...>;
  return AccurateNorm(Type(a), Type(b), Type(std::forward<Args>(args))...);
}

// Computes the reciprocal of the Euclidean norm of two values while avoiding
// intermediate underflow and overflow.
template <typename T>
constexpr auto AccurateRNorm(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fabs;
  using std::fpclassify;
  using std::isinf;
  using std::isnan;
  using std::minmax;

  a = fabs(a);
  b = fabs(b);

  if (isinf(a) || isinf(b)) {
    return T(0);
  }

  // Preserve NaN inputs after giving infinities precedence.
  if (isnan(a)) {
    return a;
  }

  if (isnan(b)) {
    return b;
  }

  // Ensure |x| ≥ |y|. NaNs were handled above, so ordering cannot raise an
  // exception.
  const auto [y, x] = std::minmax(a, b);

  const int cls = fpclassify(x);

  if (cls == FP_INFINITE) {
    return T(0);  // +/-oo, y
  }

  if (cls == FP_ZERO) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  using internal::AccurateNormTraits;

  if (y <= x * AccurateNormTraits<T>::CutoffRatio()) {
    return T(1) / x;
  }

  using internal::UnscaledAccurateRNorm;

  const T scale = AccurateNormTraits<T>::Scale();

  // The rescaling differs from the one used in AccurateNorm because scaling the
  // arguments x and y of a reciprocal hypotenuse yields
  //
  //     1/sqrt(x'^2+y'^2)
  // <=> 1/sqrt((x*s)^2+(y*s)^2)
  // <=> 1/(s*sqrt(x^2+y^2))
  //
  // i.e., to cancel the scale, we need reapply it to the result.

  if (x > AccurateNormTraits<T>::Huge()) {
    // Scale x to prevent an overflow
    return UnscaledAccurateRNorm(x * scale, y * scale) * scale;
  }

  if (y < AccurateNormTraits<T>::Tiny()) {
    // Scale y to prevent an underflow
    return UnscaledAccurateRNorm(x / scale, y / scale) / scale;
  }

  // Avoid rounding errors due to unnecessary scaling
  return UnscaledAccurateRNorm(x, y);
}

// Computes the reciprocal of the Euclidean norm of three or more values of the
// same type while avoiding intermediate underflow and overflow.
template <typename T, typename... Args>
constexpr auto AccurateRNorm(T a, T b, Args&&... args)
    -> std::enable_if_t<(sizeof...(Args) > 0 &&
                         (std::is_same_v<T, std::decay_t<Args>> && ...)),
                        internal::Promote_t<T>> {
  using Type = internal::Promote_t<T>;
  using std::fabs;
  using std::fmax;
  using std::fpclassify;
  using std::ilogb;
  using std::isinf;
  using std::isnan;
  using std::scalbn;

  const Type first = Type(a);
  const Type second = Type(b);
  const std::initializer_list<Type> values = {
      first, second, Type(std::forward<Args>(args))...};

  const auto infinity = std::find_if(
      values.begin(), values.end(), [](Type value) { return isinf(value); });

  if (infinity != values.end()) {
    return Type(0);
  }

  // Preserve NaN inputs after giving infinities precedence.
  const auto nan = std::find_if(
      values.begin(), values.end(), [](Type value) { return isnan(value); });

  if (nan != values.end()) {
    return std::numeric_limits<Type>::quiet_NaN();
  }

  const Type first_magnitude = fabs(first);
  const Type second_magnitude = fabs(second);
  const std::initializer_list<Type> magnitudes = {
      first_magnitude,
      second_magnitude,
      fabs(Type(std::forward<Args>(args)))...};

  const Type maximum = std::accumulate(
      magnitudes.begin(), magnitudes.end(), Type(0), [](Type lhs, Type rhs) {
        return fmax(lhs, rhs);
      });

  if (fpclassify(maximum) == FP_ZERO) {
    return std::numeric_limits<Type>::quiet_NaN();
  }

  const int exponent = ilogb(maximum);
  // Scale by a radix power to avoid rounding by an arbitrary maximum.

  // Normalize by a radix power near the largest magnitude before accumulating
  // the compensated reciprocal squares. Apply the scale after taking the
  // reciprocal to avoid intermediate overflow.
  using internal::UnscaledAccurateRNorm;

  const Type normalized_first = scalbn(first_magnitude, -exponent);
  const Type normalized_second = scalbn(second_magnitude, -exponent);
  const auto [y, x] = std::minmax(normalized_first, normalized_second);
  const Type normalized_rnorm = UnscaledAccurateRNorm(
      x, y, (scalbn(Type(std::forward<Args>(args)), -exponent))...);

  return scalbn(normalized_rnorm, -exponent);
}

// Computes the reciprocal of the Euclidean norm of two or more arithmetic
// values after promoting all arguments to a common floating-point type.
template <typename T, typename U, typename... Args>
constexpr internal::Promote_t<T, U, Args...> AccurateRNorm(T a,
                                                           U b,
                                                           Args&&... args) {
  using Type = internal::Promote_t<T, U, Args...>;
  return AccurateRNorm(Type(a), Type(b), Type(std::forward<Args>(args))...);
}

}  // namespace ceres

#endif  // CERES_PUBLIC_ACCURATE_NORM_
