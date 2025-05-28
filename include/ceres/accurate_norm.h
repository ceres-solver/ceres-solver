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
#include <limits>
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
  using type = double;
};

template <typename T>
struct Promote<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = T;
};

template <typename... Ts>
using Promote_t = decltype((typename Promote<Ts>::type(0) + ... + 0));

template <typename T, typename Enable = void>
struct AccurateNormTraits {
  // √(ε/2) <=> 1/(√2)·𝛽^((1-p)/2) <=> (√2)/2·𝛽^((1-p)/2)
  static T Varying() noexcept {
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
  const T x_sq = x * x;
  const T y_sq = y * y;
  // Recover the rounding error of the floating-point addition of both squares.
  const auto [sigma, sigma_e] = Fast2Sum(x_sq, y_sq);
  // Use the 2MultFMA algorithm to recover the rounding error due to squaring x
  // and y.
  return std::make_pair(
      sigma, sigma_e + TwoMultFMA(y, y, y_sq) + TwoMultFMA(x, x, x_sq));
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

}  // namespace internal

// Computes the Euclidean norm of two values while avoiding intermediate
// underflow and overflow.
template <typename T>
constexpr auto AccurateNorm(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fabs;
  using std::isfinite;
  using std::isinf;
  using std::minmax;

  a = fabs(a);
  b = fabs(b);

  if (isinf(a)) {
    return a;
  }

  if (isinf(b)) {
    return b;
  }

  if (!isfinite(a)) {
    return a;
  }

  if (!isfinite(b)) {
    return b;
  }

  // Ensure |x| ≥ |y|. No exception can be raised here since the arguments are
  // finite at this point. The same is true for the following comparisons.
  const auto [y, x] = minmax(a, b);

  using internal::AccurateNormTraits;

  if (y <= x * AccurateNormTraits<T>::Varying()) {
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
  using std::fma;
  using std::fmax;
  using std::fpclassify;
  using std::isfinite;
  using std::isinf;
  using std::sqrt;

  const Type first = fabs(Type(a));
  const Type second = fabs(Type(b));
  Type nonfinite = Type(0);
  bool has_nonfinite = false;
  const auto update_nonfinite = [&nonfinite, &has_nonfinite](Type value) {
    if (isinf(value) || (!has_nonfinite && !isfinite(value))) {
      nonfinite = value;
      has_nonfinite = true;
    }
  };

  update_nonfinite(first);
  update_nonfinite(second);
  (update_nonfinite(fabs(Type(std::forward<Args>(args)))), ...);
  if (has_nonfinite) {
    return nonfinite;
  }

  // Normalize by the largest magnitude before accumulating the squares. This
  // avoids intermediate overflow and underflow and requires one square root.
  Type scale = fmax(fabs(first), fabs(second));
  ((scale = fmax(scale, fabs(Type(std::forward<Args>(args))))), ...);

  if (fpclassify(scale) == FP_ZERO) {
    return Type(0);
  }

  Type sum = Type(0);
  const auto add_square = [&sum, scale](Type value) {
    const Type normalized = value / scale;
    sum = fma(normalized, normalized, sum);
  };
  add_square(first);
  add_square(second);
  (add_square(Type(std::forward<Args>(args))), ...);

  return scale * sqrt(sum);
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

  if (isnan(a)) {
    return a;
  }

  if (isnan(b)) {
    return b;
  }

  // Ensure |x| ≥ |y|. NaNs were handled above, so ordering cannot raise an
  // exception.
  const auto [y, x] = minmax(a, b);

  const int cls = fpclassify(x);

  if (cls == FP_INFINITE) {
    return T(0);  // +/-oo, y
  }

  if (cls == FP_ZERO) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  using internal::AccurateNormTraits;

  if (y <= x * AccurateNormTraits<T>::Varying()) {
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
  using std::isfinite;
  using std::isinf;
  using std::isnan;

  const Type first = Type(a);
  const Type second = Type(b);

  if (isinf(first) || isinf(second) ||
      (isinf(Type(std::forward<Args>(args))) || ...)) {
    return Type(0);
  }

  if (isnan(first) || isnan(second) ||
      (isnan(Type(std::forward<Args>(args))) || ...)) {
    return std::numeric_limits<Type>::quiet_NaN();
  }

  Type scale = fmax(fabs(first), fabs(second));
  ((scale = fmax(scale, fabs(Type(std::forward<Args>(args))))), ...);

  if (fpclassify(scale) == FP_ZERO) {
    return std::numeric_limits<Type>::quiet_NaN();
  }

  // Normalize by the largest magnitude before accumulating the squares. Apply
  // the scale after taking the reciprocal to avoid intermediate overflow.
  const Type normalized_norm =
      AccurateNorm(first / scale,
                   second / scale,
                   (Type(std::forward<Args>(args)) / scale)...);

  if (scale > Type(1)) {
    return (Type(1) / normalized_norm) / scale;
  }

  const Type inverse_scale = Type(1) / scale;
  if (isfinite(inverse_scale)) {
    return inverse_scale / normalized_norm;
  }

  return Type(1) / (scale * normalized_norm);
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
