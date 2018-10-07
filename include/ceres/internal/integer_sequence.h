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
// This class mimics std::integer_sequence. That is the reason to follow the
// naming convention of the stl and not the google one. Once Ceres switches
// to c++ 14 this class can be removed.

#ifndef CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_H_
#define CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_H_

#if __cplusplus >= 201402L
// We have at least c++ 14 support. Use integer_sequence from the standard.
// Sometimes the STL implementation uses a compiler intrinsic to generate
// the sequences which will speed up compilation.
#include <utility>

namespace ceres {
namespace internal {
template <typename T, T... Ns>
using integer_sequence = std::integer_sequence<T, Ns...>;

template <typename T, T N>
using make_integer_sequence = std::make_integer_sequence<T, N>;

}  // namespace internal
}  // namespace ceres
#else

namespace ceres {
namespace internal {

template <typename T, T... Ns>
struct integer_sequence {
  using value_type = T;
};

// Implementation of make_integer_sequence.
//
// Recursively instantiate make_integer_sequence_impl until Ns
// contains the sequence 0, 1, ..., Total-1.
//
// Example for Total = 4:
//                            T    CurIdx, Total, Ns...
// make_integer_sequence_impl<int, 0,      4                >
// make_integer_sequence_impl<int, 1,      4,     0         >
// make_integer_sequence_impl<int, 2,      4,     0, 1      >
// make_integer_sequence_impl<int, 3,      4,     0, 1, 2   >
// make_integer_sequence_impl<int, 4,      4,     0, 1, 2, 3>
//                                                ^^^^^^^^^^
//                                                resulting sequence.
//
// The implemented algorithm has linear complexity for simplicity. A O(log(N))
// implementation can be found e.g. here:
// https://stackoverflow.com/questions/17424477/implementation-c14-make-integer-sequence
template <typename T, T CurIdx, T Total, T... Ns>
struct make_integer_sequence_impl {
  using type = typename make_integer_sequence_impl<T, CurIdx + 1, Total, Ns...,
                                                   CurIdx>::type;
};

// End of 'recursion' when CurIdx reaches Total. All indices 0, 1, ..., N-1 are
// contained in Ns. The final integer_sequence is created here.
template <typename T, T Total, T... Ns>
struct make_integer_sequence_impl<T, Total, Total, Ns...> {
  using type = integer_sequence<T, Ns...>;
};

// A helper alias template to simplify creation of integer_sequence with 0, 1,
// ..., N-1 as Ns:
template <typename T, T N>
using make_integer_sequence =
    typename make_integer_sequence_impl<T, 0, N>::type;

}  // namespace internal
}  // namespace ceres

#endif

#endif  // CERES_PUBLIC_INTERNAL_INTEGER_SEQUENCE_H_
