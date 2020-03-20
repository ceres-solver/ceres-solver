// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//

#ifndef CERES_PUBLIC_INTERNAL_STATIC_FIXED_ARRAY_H_
#define CERES_PUBLIC_INTERNAL_STATIC_FIXED_ARRAY_H_

#include <array>
#include <vector>

#include "ceres/internal/fixed_array.h"
#include "ceres/types.h"
namespace ceres {
namespace internal {

// StaticFixedArray selects the best array implementation based on template
// arguments. If the size is not known at compile-time, pass
// ceres::DYNAMIC as a size-template argument.
//
// Three different containers are selected in different scenarios:
//
//   size == DYNAMIC:
//      -> ceres::internal::FixedArray<T, max_stack_size>(size)

//   size != DYNAMIC  &&  size <= max_stack_size
//      -> std::array<T,size>

//   size != DYNAMIC  &&  size >  max_stack_size
//      -> std::vector<T>(size)
//
template <typename T,
          int size,
          int max_stack_size,
          bool dynamic = (size == DYNAMIC),
          bool fits_on_stack = (size < max_stack_size)>
struct StaticFixedArray {};

template <typename T, int size, int max_stack_size, bool fits_on_stack>
struct StaticFixedArray<T, size, max_stack_size, true, fits_on_stack>
    : ceres::internal::FixedArray<T, max_stack_size> {
  StaticFixedArray(int s) : ceres::internal::FixedArray<T, max_stack_size>(s) {}
};

template <typename T, int size, int max_stack_size>
struct StaticFixedArray<T, size, max_stack_size, false, true>
    : std::array<T, size> {
  StaticFixedArray(int s) {}
};

template <typename T, int size, int max_stack_size>
struct StaticFixedArray<T, size, max_stack_size, false, false>
    : std::vector<T> {
  StaticFixedArray(int s) : std::vector<T>(s) {}
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_FIXED_ARRAY_H_
