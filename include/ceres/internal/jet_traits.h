// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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

#ifndef CERES_PUBLIC_INTERNAL_JET_TRAITS_H_
#define CERES_PUBLIC_INTERNAL_JET_TRAITS_H_

#include <tuple>
#include <type_traits>
#include <utility>

#include "ceres/jet_fwd.h"

namespace ceres {
namespace internal {

// Predicate that determines whether T is a Jet.
template <typename T, typename E = void>
struct IsJet : std::false_type {};

template <typename T, int N>
struct IsJet<ceres::Jet<T, N>> : std::true_type {};

// Convenience variable template for IsJet.
template <typename T>
constexpr bool IsJet_v = IsJet<T>::value;

// Predicate that determines whether any of the Types is a Jet.
template <typename... Types>
struct IsAnyJet : std::false_type {};

template <typename T, typename... Types>
struct IsAnyJet<T, Types...> : IsAnyJet<Types...> {};

template <typename T, int N, typename... Types>
struct IsAnyJet<ceres::Jet<T, N>, Types...> : std::true_type {};

// Convenience variable template for IsAnyJet.
template <typename... Types>
constexpr bool IsAnyJet_v = IsAnyJet<Types...>::value;

// Extracts the underlying floating-point from a type T.
template <typename T, typename E = void>
struct UnderlyingScalar {
  using type = T;
};

template <typename T, int N>
struct UnderlyingScalar<ceres::Jet<T, N>, std::enable_if_t<IsJet_v<T>>>
    : UnderlyingScalar<typename T::Scalar> {};

template <typename T, int N>
struct UnderlyingScalar<ceres::Jet<T, N>, std::enable_if_t<!IsJet_v<T>>>
    : UnderlyingScalar<T> {};

// Convenience template alias for UnderlyingScalar type trait.
template <typename T>
using UnderlyingScalar_t = typename UnderlyingScalar<T>::type;

// Determines the rank of a type. This allows to ensure that types passed as
// arguments are compatible to each other. The rank of Jet is determined by the
// dimensions of the dual part. The rank of scalar is always 0.
template <typename T, typename E = void>
struct Rank : std::integral_constant<int, -1> {};

template <typename T>
struct Rank<T, std::enable_if_t<std::is_scalar<T>::value>>
    : std::integral_constant<int, 0> {};

template <typename T, int N>
struct Rank<ceres::Jet<T, N>> : std::integral_constant<int, N> {};

// Convenience variable template for Rank.
template <typename T>
constexpr int Rank_v = Rank<T>::value;

template <typename... Types>
struct AllSame;

template <>
struct AllSame<> : std::true_type {};

template <typename T>
struct AllSame<T> : std::true_type {};

template <typename T1, typename T2>
struct AllSame<T1, T2> : std::is_same<T1, T2> {};

template <typename T1, typename T2, typename... Types>
struct AllSame<T1, T2, Types...> {
  static constexpr bool value =
      AllSame<T1, T2>::value && AllSame<T2, Types...>::value;
};

template <typename... Types>
constexpr bool AllSame_v = AllSame<Types...>::value;

template <typename... Types>
using Ranks_t = std::integer_sequence<int, Rank_v<Types>...>;

// Removes all elements from a integer sequence corresponding to specified Value
template <typename T, T Value, typename... Sequence>
struct Remove;

// Final filtered sequence
template <typename T, T Value, T... Values>
struct Remove<T,
              Value,
              std::integer_sequence<T, Values...>,
              std::integer_sequence<T>> {
  using type = std::integer_sequence<T, Values...>;
};

// Found a matching value
template <typename T, T Value, T... Head, T... Tail>
struct Remove<T,
              Value,
              std::integer_sequence<T, Head...>,
              std::integer_sequence<T, Value, Tail...>>
    : Remove<T,
             Value,
             std::integer_sequence<T, Head...>,
             std::integer_sequence<T, Tail...>> {};

// Move one element from the tail to the head
template <typename T, T Value, T... Head, T T1, T... Tail>
struct Remove<T,
              Value,
              std::integer_sequence<T, Head...>,
              std::integer_sequence<T, T1, Tail...>>
    : Remove<T,
             Value,
             std::integer_sequence<T, Head..., T1>,
             std::integer_sequence<T, Tail...>> {};

// Start recursion by splitting the integer sequence into two separate ones
template <typename T, T Value, T... Tail>
struct Remove<T, Value, std::integer_sequence<T, Tail...>>
    : Remove<T,
             Value,
             std::integer_sequence<T>,
             std::integer_sequence<T, Tail...>> {};

template <typename T, T Value, typename Sequence>
using Remove_t = typename Remove<T, Value, Sequence>::type;

// Determines whether the values of an integer sequence are all the same.
template <typename T>
struct AllEqual;

template <>
struct AllEqual<std::integer_sequence<int>> : std::true_type {};

template <int Value>
struct AllEqual<std::integer_sequence<int, Value>> : std::true_type {};

template <int V1, int V2, int... Values>
struct AllEqual<std::integer_sequence<int, V1, V2, Values...>> {
  static constexpr bool value =
      V1 == V2 && AllEqual<std::integer_sequence<int, V2, Values...>>::value;
};

// Convenience variable template for AllEqual.
template <class Sequence>
constexpr bool AllEqual_v = AllEqual<Sequence>::value;

// Returns the scalar part of a type. This overload acts as identity.
template <typename T>
constexpr decltype(auto) scalar(T&& value) noexcept {
  return std::forward<T>(value);
}

// Returns the scalar part of a Jet whose underlying type is a scalar.
template <typename T, int N, std::enable_if_t<!IsJet_v<T>>* = nullptr>
constexpr decltype(auto) scalar(const ceres::Jet<T, N>& value) noexcept {
  return value.a;
}

// Recursively expands the scalar part of a Jet whose underlying type is a Jet
// itself.
template <typename T, int N, std::enable_if_t<IsJet_v<T>>* = nullptr>
constexpr decltype(auto) scalar(const ceres::Jet<T, N>& value) noexcept {
  return scalar(value.a);
}

}  // namespace internal

// Type traits ensuring at least one of the types is a Jet,
// the underlying scalar types are the same and Jet dimensions match.
//
// The type trait can be further specialized if necessary.
//
// This trait is a candidate for a concept definition once C++20 features can
// be used.
template <typename... Types>
// clang-format off
struct CompatibleJetOperands : std::integral_constant
<
    bool
    // At least one of the types is a Jet
    , internal::IsAnyJet_v<Types...>
    // The underlying floating-point types are exactly the same
    && internal::AllSame_v<internal::UnderlyingScalar_t<Types>...>
    // Non-zero ranks of types are equal
    && internal::AllEqual_v<internal::Remove_t<int, 0, internal::Ranks_t<Types...>>>
>
// clang-format on
{};

template <typename... Types>
// clang-format off
struct PromotableJetOperands : std::integral_constant
<
    bool
    // Types can be compatible among each other
    , internal::IsAnyJet_v<Types...>
    // Non-zero ranks of types are equal
    && internal::AllEqual_v<internal::Remove_t<int, 0, internal::Ranks_t<Types...>>>
>
// clang-format on
{};

// Convenience variable templates ensuring at least one of the types is a Jet,
// the underlying scalar types are the same and Jet dimensions match.
//
// This trait is a candidate for a concept definition once C++20 features can
// be used.
template <typename... Types>
constexpr bool CompatibleJetOperands_v = CompatibleJetOperands<Types...>::value;

// Convenience variable templates ensuring at least one of the types is a Jet,
// the underlying scalar types are compatible among each other and Jet
// dimensions match.
//
// This trait is a candidate for a concept definition once C++20 features can
// be used.
template <typename... Types>
constexpr bool PromotableJetOperands_v = PromotableJetOperands<Types...>::value;

}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_JET_TRAITS_H_
