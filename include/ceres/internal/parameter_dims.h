// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: jodebo_beck@gmx.de (Johannes Beck)
//
// The ParameterDims class holds either the fixed sizes of the parameter blocks
// (like in SizedCostFunction, etc.) or specifies that the dimensions are not
// known at compile time.

#ifndef CERES_PUBLIC_INTERNAL_PARAMETER_DIMS_H_
#define CERES_PUBLIC_INTERNAL_PARAMETER_DIMS_H_

#include <array>

#include "integer_sequence.h"
#include "integer_sequence_algorithm.h"

namespace ceres {
namespace internal {

template <bool Dynamic, int... Ns>
class ParameterDims {
 private:
  template <int N, int... Ts>
  static constexpr bool isValid(integer_sequence<int, N, Ts...> /* NOT USED*/) {
    return (N <= 0) ? false : isValid(integer_sequence<int, Ts...>());
  }

  static constexpr bool isValid(integer_sequence<int> /* NOT USED*/) {
    return true;
  }

 public:
  static constexpr bool kIsValid = isValid(integer_sequence<int, Ns...>());
  static_assert(kIsValid,
                "Invalid parameter block dimension detected. Each parameter "
                "block dimension must be bigger than zero.");

  static constexpr bool kIsDynamic = Dynamic;
  static constexpr int kNumParameterBlocks = sizeof...(Ns);
  static constexpr int kNumParameters =
      Sum<integer_sequence<int, Ns...>>::value;
  static_assert(kIsDynamic || kNumParameterBlocks > 0,
                "At least one parameter block must be specified.");

  using Parameters = integer_sequence<int, Ns...>;

  static constexpr int getDim(int dim) { return params_[dim]; }

 private:
  static constexpr std::array<int, kNumParameterBlocks> params_{Ns...};
};

template <typename ParameterDims, typename T, int... Indices>
inline std::array<T*, ParameterDims::kNumParameterBlocks> GetUnpackedParameters(
    T* ptr, integer_sequence<int, Indices...> /*NOT USED*/) {
  return std::array<T*, ParameterDims::kNumParameterBlocks>{{ptr + Indices...}};
}

template <bool Dynamic, int... Ns>
constexpr std::array<int, ParameterDims<Dynamic, Ns...>::kNumParameterBlocks>
    ParameterDims<Dynamic, Ns...>::params_;

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_PARAMETER_DIMS_H_
