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

#include "ceres/internal/integer_sequence_algorithm.h"

#include <type_traits>
#include <utility>

namespace ceres {
namespace internal {

// Unit tests for summation of integer sequence.
static_assert(Sum<std::integer_sequence<int>>::Value == 0,
              "Unit test of summing up an integer sequence failed.");
static_assert(Sum<std::integer_sequence<int, 2>>::Value == 2,
              "Unit test of summing up an integer sequence failed.");
static_assert(Sum<std::integer_sequence<int, 2, 3>>::Value == 5,
              "Unit test of summing up an integer sequence failed.");
static_assert(Sum<std::integer_sequence<int, 2, 3, 10>>::Value == 15,
              "Unit test of summing up an integer sequence failed.");
static_assert(Sum<std::integer_sequence<int, 2, 3, 10, 4>>::Value == 19,
              "Unit test of summing up an integer sequence failed.");
static_assert(Sum<std::integer_sequence<int, 2, 3, 10, 4, 1>>::Value == 20,
              "Unit test of summing up an integer sequence failed.");

// Unit tests for exclusive scan of integer sequence.
static_assert(std::is_same<ExclusiveScan<std::integer_sequence<int>>,
                           std::integer_sequence<int>>::value,
              "Unit test of calculating the exclusive scan of an integer "
              "sequence failed.");
static_assert(std::is_same<ExclusiveScan<std::integer_sequence<int, 2>>,
                           std::integer_sequence<int, 0>>::value,
              "Unit test of calculating the exclusive scan of an integer "
              "sequence failed.");
static_assert(std::is_same<ExclusiveScan<std::integer_sequence<int, 2, 1>>,
                           std::integer_sequence<int, 0, 2>>::value,
              "Unit test of calculating the exclusive scan of an integer "
              "sequence failed.");
static_assert(std::is_same<ExclusiveScan<std::integer_sequence<int, 2, 1, 10>>,
                           std::integer_sequence<int, 0, 2, 3>>::value,
              "Unit test of calculating the exclusive scan of an integer "
              "sequence failed.");

}  // namespace internal
}  // namespace ceres
