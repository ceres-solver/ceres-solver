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

#include "ceres/internal/tuple_algorithm.h"

#include <type_traits>

namespace ceres {
namespace internal {

// Test generation of homogeneous tuple types
static_assert(std::is_same<TupleOf_t<int, 2>, std::tuple<int, int>>::value,
              "homogeneous tuple type does not match");
static_assert(
    std::is_same<TupleOf_t<short, 3>, std::tuple<short, short, short>>::value,
    "homogeneous tuple type does not match");
static_assert(std::is_same<TupleOf_t<double, 1>, std::tuple<double>>::value,
              "homogeneous tuple type does not match");
static_assert(std::is_same<TupleOf_t<int, 0>, std::tuple<>>::value,
              "homogeneous tuple type does not match");
static_assert(std::is_same<TupleOf_t<short, 0>, std::tuple<>>::value,
              "homogeneous tuple type does not match");
static_assert(std::is_same<TupleOf_t<double, 0>, std::tuple<>>::value,
              "homogeneous tuple type does not match");

// Test summation over tuple values
static_assert(ComputeSum(std::make_tuple()) == 0,
              "Sum of an empty tuple must be 0");
static_assert(ComputeSum(std::make_tuple(1, 1, 1)) == 3,
              "Sum of a tuple must be 3");
static_assert(ComputeSum(std::make_tuple(-1, -2, -3)) == -6,
              "Sum of a tuple must be -6");
static_assert(ComputeSum(std::make_tuple(1, 1, 0, 0)) == 2,
              "Sum of a tuple must be 2");

// Test exclusive scan over tuple values
static_assert(ComputeExclusiveScan(std::make_tuple()) == std::tuple<>{},
              "ExclusiveScan of tuple values does not match");
static_assert(ComputeExclusiveScan(std::make_tuple(0, 0, 0, 0)) ==
                  std::make_tuple(0, 0, 0, 0),
              "ExclusiveScan of tuple values does not match");
static_assert(ComputeExclusiveScan(std::make_tuple(1, 1, 1)) ==
                  std::make_tuple(0, 1, 2),
              "ExclusiveScan of tuple values does not match");
static_assert(ComputeExclusiveScan(std::make_tuple(-1, -2, -3, -4)) ==
                  std::make_tuple(0, -1, -3, -6),
              "ExclusiveScan of tuple values does not match");

// Test type promotion
static_assert(std::is_same<decltype(ComputeSum(std::make_tuple(1, 1l, 1ll))),
                           long long>::value,
              "Sum result type does not match");

static_assert(
    std::is_same<decltype(ComputeExclusiveScan(std::make_tuple(1, 1l, 1ll))),
                 std::tuple<long long, long long, long long>>::value,
    "ExclusiveScan result tuple type does not match");

}  // namespace internal
}  // namespace ceres
