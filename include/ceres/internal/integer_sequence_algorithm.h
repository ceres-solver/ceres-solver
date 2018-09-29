// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Algorithms to be used together with integer_sequence, like computing the sum
// or the prefix sum at compile time.

#ifndef CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_ALGORITHM_H_
#define CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_ALGORITHM_H_

#include "integer_sequence.h"

namespace ceres {
namespace internal {

template <typename Seq>
struct SumImpl;

template <typename T, T N, T... Ns>
struct SumImpl<integer_sequence<T, N, Ns...>> {
  static constexpr T value = N + SumImpl<integer_sequence<T, Ns...>>::value;
};

template <typename T>
struct SumImpl<integer_sequence<T>> {
  static constexpr T value = T(0);
};

template <typename Seq>
struct Sum {
 private:
  using T = typename Seq::value_type;

 public:
  static constexpr T value = SumImpl<Seq>::value;
};

template <typename T, T Sum, typename SeqIn, typename SeqOut>
struct PrefixSumImpl;

template <typename T, T Sum, T N, T... Ns, T... Rs>
struct PrefixSumImpl<T, Sum, integer_sequence<T, N, Ns...>,
                     integer_sequence<T, Rs...>> {
  using Type = typename PrefixSumImpl<T, Sum + N, integer_sequence<T, Ns...>,
                                      integer_sequence<T, Rs..., Sum>>::Type;
};

template <typename T, T Sum, typename SeqOut>
struct PrefixSumImpl<T, Sum, integer_sequence<T>, SeqOut> {
  using Type = SeqOut;
};

template <typename Seq>
struct PrefixSumT {
 private:
  using T = typename Seq::value_type;

 public:
  using Type = typename PrefixSumImpl<T, T(0), Seq, integer_sequence<T>>::Type;
};

template <typename Seq>
using PrefixSum = typename PrefixSumT<Seq>::Type;

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_ALGORITHM_H_
