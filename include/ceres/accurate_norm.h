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
// This header implements functions for accurately computing the Euclidean norm
// of two or more arguments and its reciprocal while avoiding underflow and
// overflow.
//
// Both functions accumulate the squares of the arguments as an unevaluated sum
// of the rounded sum and its rounding error, and correct the square root of
// the rounded sum, or its reciprocal, by a single Newton step that accounts for
// the rounding errors. They share the same implementation for any number of
// arguments: the result is computed without rescaling if the largest magnitude
// is within a range where rescaling cannot change the result, and otherwise
// after rescaling all arguments by a fixed radix power.
//
// Unlike the 2-argument algorithm in [1], the functions do not return the
// larger argument if the smaller one is negligible. Neglecting arguments does
// not generalize to more arguments since the errors of several neglected
// squares accumulate, and the reciprocal of the larger argument can differ from
// the correctly rounded reciprocal norm.
//
// The implementation is derived from the following two papers:
//
// [1] Borges, C. F. (2021). Algorithm 1014: An Improved Algorithm for
//     hypot(x,y). ACM Transactions on Mathematical Software, 47(1), 1–12.
//     https://doi.org/10.1145/3428446
//
// [2] Borges, C. F. (2021). Fast Compensated Algorithms for the Reciprocal
//     Square Root, the Reciprocal Hypotenuse, and Givens Rotations.
//     http://arxiv.org/abs/2103.08694

#ifndef CERES_PUBLIC_ACCURATE_NORM_H_
#define CERES_PUBLIC_ACCURATE_NORM_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "ceres/internal/compensated_math.h"

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
  // Identity mapping.
  using type = T;
};

// The type of the sum of the promoted arguments, e.g., double if any argument
// is integral and float if all arguments are float. References and
// cv-qualifiers of the argument types are ignored.
template <typename... Ts>
using Promote_t =
    decltype((typename Promote<std::decay_t<Ts>>::type(0) + ... + 0));

// Computes 2^exponent exactly. Unlike std::scalbn, the function can be
// evaluated in constant expressions which avoids runtime library calls for
// compilers that do not fold std::scalbn.
template <typename T>
constexpr T PowerOfTwo(int exponent) noexcept {
  T base = exponent < 0 ? T{0.5} : T{2};
  int n = exponent < 0 ? -exponent : exponent;
  T result{1};

  while (n > 0) {
    if (n % 2 == 1) {
      result *= base;
    }

    n /= 2;

    // Avoid squaring the base beyond the representable range once all bits of
    // the exponent have been consumed.
    if (n > 0) {
      base *= base;
    }
  }

  return result;
}

// Computes ⌈log₂(n)⌉ for a positive n.
constexpr int CeilLog2(int n) noexcept {
  int result = 0;

  while ((1 << result) < n) {
    ++result;
  }

  return result;
}

// The second template parameter allows this trait to be customized using
// SFINAE.
//
// In the following, p denotes the precision of T, and e_min and e_max denote
// its minimum and maximum exponent as defined by IEEE 754, i.e.,
// std::numeric_limits<T>::min_exponent − 1 and max_exponent − 1, respectively.
template <typename T, typename Enable = void>
struct AccurateNormTraits {
  // Smallest magnitude x whose square has an exactly representable rounding
  // error. The error is a multiple of ulp(x)² = 𝛽^(2(e−p+1)) for
  // x ∈ [𝛽^e, 𝛽^(e+1)) which must not fall below the smallest subnormal
  // 𝛽^(e_min−p+1), i.e., e ≥ ⌈(e_min+p−1)/2⌉. Integer division truncates the
  // negative numerator toward zero which yields the ceiling.
  static constexpr T Tiny() noexcept {
    constexpr int e_min = std::numeric_limits<T>::min_exponent - 1;
    return PowerOfTwo<T>((e_min + std::numeric_limits<T>::digits - 1) / 2);
  }

  // Smallest maximum magnitude of the arguments that allows the variadic norm
  // to be computed without rescaling. Every argument whose square has an
  // inexact rounding error is then smaller than 𝛽^(-p) times the maximum and
  // cannot affect the result.
  static constexpr T UnscaledMinimum() noexcept {
    return Tiny() * PowerOfTwo<T>(std::numeric_limits<T>::digits);
  }

  // Largest maximum magnitude of count arguments that allows the variadic norm
  // to be computed without rescaling. The sum of the squares then cannot
  // exceed 𝛽^(e_max).
  static constexpr T UnscaledMaximum(int count) noexcept {
    return PowerOfTwo<T>(
        (std::numeric_limits<T>::max_exponent - 1 - CeilLog2(count)) / 2);
  }

  // Exponent of ulp(√F_min) = 𝛽^(e_min/2−p+1) where F_min = 𝛽^e_min is the
  // smallest normal value.
  static constexpr int ScaleExponent() noexcept {
    return (std::numeric_limits<T>::min_exponent - 1) / 2 -
           std::numeric_limits<T>::digits + 1;
  }

  // Largest maximum magnitude of count arguments that allows the reciprocal
  // norm to be computed without rescaling. The reciprocal norm then is at least
  // Tiny() which keeps the rounding error of its square exact. The norm is at
  // most the maximum magnitude times 𝛽^⌈⌈log₂(count)⌉/2⌉ ≥ √count.
  static constexpr T ReciprocalUnscaledMaximum(int count) noexcept {
    return T{1} / (Tiny() * PowerOfTwo<T>((CeilLog2(count) + 1) / 2));
  }
};

// Determines the largest magnitude of the arguments. An infinite argument
// yields positive infinity, even if another argument is NaN. Otherwise, a NaN
// argument yields a NaN with its payload preserved.
template <typename T, typename... Args>
inline auto MaximumMagnitude(T x, Args... args) noexcept
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            (std::is_same_v<T, Args> && ...),
                        T> {
  using std::fabs;
  using std::fmax;
  using std::isinf;
  using std::isnan;

  // Fold expressions instead of a loop over the arguments allow compilers to
  // generate branch-free code for the common case of finite arguments.
  if ((isinf(x) || ... || isinf(args))) {
    return std::numeric_limits<T>::infinity();
  }

  if ((isnan(x) || ... || isnan(args))) {
    // Keep the first NaN to preserve its payload.
    T nan = fabs(x);
    ((isnan(nan) ? void() : void(nan = fabs(args))), ...);
    return nan;
  }

  T maximum = fabs(x);
  ((maximum = fmax(maximum, T(fabs(args)))), ...);
  return maximum;
}

// Computes the sum of squares x^2 + y^2 + ... as the rounded sum and its
// rounding error.
template <typename T, typename... Args>
inline auto UnscaledAccurateSquareNormWithError(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            (std::is_same_v<T, Args> && ...),
                        std::pair<T, T>> {
  using std::fma;

  // Use 2MultFMA to recover the rounding error from squaring x.
  T sigma = x * x;
  T sigma_e = fma(x, x, -sigma);

  // Add the remaining squares using a radix-independent error-free transform.
  // The rounding errors are small compared to the sum. Accumulating them using
  // plain additions is therefore sufficient.
  const auto accumulate = [&sigma, &sigma_e](T value) {
    const T value_sq = value * value;
    const auto [sum, sum_e] = TwoSum(sigma, value_sq);
    sigma = sum;
    sigma_e += sum_e + fma(value, value, -value_sq);
  };

  accumulate(y);
  (accumulate(args), ...);

  return std::make_pair(sigma, sigma_e);
}

// Computes sqrt(x^2 + y^2 + ...) without checking the arguments. The arguments
// must be finite and scaled such that the sum of their squares does not
// overflow and the rounding errors of all squares that affect the result are
// exact. Not intended to be invoked by users.
template <typename T, typename... Args>
inline auto UnscaledAccurateNorm(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  const auto [sigma, sigma_e] =
      UnscaledAccurateSquareNormWithError(x, y, args...);
  const T h = sqrt(sigma);
  const T tau = sigma_e + fma(-h, h, sigma);
  // To solve h² = σ, Newton's correction term f/df is
  //
  //           h² − σ
  //     δ_h = ────── .
  //            2⋅h
  //
  // The update h − δ_h adds its negation, (σ − h²) / (2⋅h).
  return fma(tau / h, T(0.5), h);
}

// Computes 1 / sqrt(x^2 + y^2 + ...) without checking the arguments. In
// addition to the requirements of UnscaledAccurateNorm, the reciprocal norm
// must be at least Tiny() to keep the rounding error of its square exact. Not
// intended to be invoked by users.
template <typename T, typename... Args>
inline auto UnscaledAccurateRNorm(T x, T y, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::fma;
  using std::sqrt;

  const auto [sigma, sigma_e] =
      UnscaledAccurateSquareNormWithError(x, y, args...);
  const T r = T(1) / sigma;
  // Rounding error of the reciprocal, 1 − r⋅(σ + σ_e)
  const T residual = fma(-r, sigma_e, fma(-r, sigma, T(1)));
  const T rho = sqrt(r);
  // Rounding error of the square root, r − ρ²
  const T tau = fma(-rho, rho, r);
  // To solve 1/ρ² = σ, Newton's correction term g/dg is
  //
  //           ρ⋅(ρ²⋅σ − 1)
  //     δ_ρ = ──────────── .
  //                 2
  //
  // The update ρ − δ_ρ adds its negation, ρ⋅(1 − ρ²⋅σ) / 2, through the final
  // fused multiply-add. Both rounding errors contribute to
  //
  //     1 − ρ²⋅σ = (1 − r⋅σ) + σ⋅(r − ρ²).
  //
  // The update combines both terms as in the compensated reciprocal square
  // root of [2], here applied to the sum of squares.
  const T nu = fma(sigma, tau, residual) / T(2);
  return fma(rho, nu, rho);
}

}  // namespace internal

// Computes the Euclidean norm of two or more floating-point values of the same
// type while avoiding intermediate underflow and overflow. An infinite argument
// produces positive infinity, even if another argument is NaN. Otherwise, a NaN
// argument produces a NaN with its payload preserved. When all arguments are
// zero, the result is zero.
template <typename T, typename... Args>
inline auto AccurateNorm(T a, T b, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            (std::is_same_v<T, Args> && ...),
                        T> {
  using std::fpclassify;
  using std::isfinite;

  const T maximum = internal::MaximumMagnitude(a, b, args...);

  if (!isfinite(maximum)) {
    return maximum;
  }

  if (fpclassify(maximum) == FP_ZERO) {
    return 0;
  }

  using internal::AccurateNormTraits;
  using internal::PowerOfTwo;
  using internal::UnscaledAccurateNorm;

  constexpr int count = 2 + sizeof...(Args);
  // Multiplying by a radix power rounds exactly as std::scalbn does.
  constexpr int exponent = AccurateNormTraits<T>::ScaleExponent();
  constexpr T down = PowerOfTwo<T>(exponent);
  constexpr T up = PowerOfTwo<T>(-exponent);

  // Rescale only if the largest magnitude is outside the range where rescaling
  // cannot change the result. Rescaling moves the largest magnitude into that
  // range. Arguments that underflow or whose squares have inexact rounding
  // errors after scaling down are negligible compared to the largest one.
  // Scaling up makes the rounding errors of all squares exact, including those
  // of subnormal arguments.
  if (maximum > AccurateNormTraits<T>::UnscaledMaximum(count)) {
    return UnscaledAccurateNorm(a * down, b * down, args * down...) * up;
  }

  if (maximum < AccurateNormTraits<T>::UnscaledMinimum()) {
    return UnscaledAccurateNorm(a * up, b * up, args * up...) * down;
  }

  return UnscaledAccurateNorm(a, b, args...);
}

// Computes the Euclidean norm of two or more arithmetic values after promoting
// all arguments to a common floating-point type.
template <typename T, typename U, typename... Args>
inline internal::Promote_t<T, U, Args...> AccurateNorm(T a, U b, Args... args) {
  using PromotedType = internal::Promote_t<T, U, Args...>;
  return AccurateNorm(PromotedType(a), PromotedType(b), PromotedType(args)...);
}

// Computes the reciprocal of the Euclidean norm of two or more floating-point
// values of the same type while avoiding intermediate underflow and overflow.
// An infinite argument produces zero, even if another argument is NaN.
// Otherwise, a NaN argument produces a NaN with its payload preserved. When all
// arguments are zero, the result is NaN.
template <typename T, typename... Args>
inline auto AccurateRNorm(T a, T b, Args... args)
    -> std::enable_if_t<std::is_floating_point_v<T> &&
                            (std::is_same_v<T, Args> && ...),
                        T> {
  using std::fpclassify;
  using std::isinf;
  using std::isnan;

  const T maximum = internal::MaximumMagnitude(a, b, args...);

  if (isinf(maximum)) {
    return 0;
  }

  if (isnan(maximum)) {
    return maximum;
  }

  if (fpclassify(maximum) == FP_ZERO) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  using internal::AccurateNormTraits;
  using internal::PowerOfTwo;
  using internal::UnscaledAccurateRNorm;

  constexpr int count = 2 + sizeof...(Args);
  // Multiplying by a radix power rounds exactly as std::scalbn does.
  constexpr int exponent = AccurateNormTraits<T>::ScaleExponent();
  constexpr T down = PowerOfTwo<T>(exponent);
  constexpr T up = PowerOfTwo<T>(-exponent);

  // Rescale as in AccurateNorm, but apply the scale to the reciprocal instead
  // of its inverse. The reciprocal of the rescaled arguments cannot overflow or
  // underflow because the rescaled largest magnitude is bounded.
  if (maximum > AccurateNormTraits<T>::ReciprocalUnscaledMaximum(count)) {
    return UnscaledAccurateRNorm(a * down, b * down, args * down...) * down;
  }

  if (maximum < AccurateNormTraits<T>::UnscaledMinimum()) {
    // Only a largest magnitude below Tiny() requires the larger power to make
    // the rounding errors of the squares of subnormal arguments exact. Scaling
    // a larger magnitude by that power could exceed
    // ReciprocalUnscaledMaximum(). Scaling by 𝛽^p instead suffices to reach
    // UnscaledMinimum().
    const T scale = maximum < AccurateNormTraits<T>::Tiny()
                        ? up
                        : PowerOfTwo<T>(std::numeric_limits<T>::digits);
    return UnscaledAccurateRNorm(a * scale, b * scale, args * scale...) * scale;
  }

  return UnscaledAccurateRNorm(a, b, args...);
}

// Computes the reciprocal of the Euclidean norm of two or more arithmetic
// values after promoting all arguments to a common floating-point type.
template <typename T, typename U, typename... Args>
inline internal::Promote_t<T, U, Args...> AccurateRNorm(T a,
                                                        U b,
                                                        Args... args) {
  using PromotedType = internal::Promote_t<T, U, Args...>;
  return AccurateRNorm(PromotedType(a), PromotedType(b), PromotedType(args)...);
}

}  // namespace ceres

#endif  // CERES_PUBLIC_ACCURATE_NORM_H_
