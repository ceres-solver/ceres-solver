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

#include "ceres/internal/integer_sequence.h"

#include <type_traits>

namespace ceres {
namespace internal {

// Unit test for integer_sequence<...>::value_type
static_assert(std::is_same<integer_sequence<unsigned int, 0>::value_type,
                           unsigned int>::value,
              "Unit test of integer sequence value type failed.");

// Unit tests for make_integer_sequence
static_assert(
    std::is_same<make_integer_sequence<int, 0>, integer_sequence<int>>::value,
    "Unit test of make integer sequence failed.");
static_assert(std::is_same<make_integer_sequence<int, 1>,
                           integer_sequence<int, 0>>::value,
              "Unit test of make integer sequence failed.");
static_assert(std::is_same<make_integer_sequence<int, 2>,
                           integer_sequence<int, 0, 1>>::value,
              "Unit test of make integer sequence failed.");
static_assert(std::is_same<make_integer_sequence<int, 3>,
                           integer_sequence<int, 0, 1, 2>>::value,
              "Unit test of make integer sequence failed.");

}  // namespace internal
}  // namespace ceres
