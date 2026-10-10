// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#ifndef CERES_PUBLIC_INTERNAL_ARRAY_SELECTOR_H_
#define CERES_PUBLIC_INTERNAL_ARRAY_SELECTOR_H_

// IWYU pragma: private

#include <array>

#include "Eigen/Core"
#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "ceres/types.h"

namespace ceres::internal {

// ArraySelector selects the best array implementation based on template
// arguments. If the size is not known at compile-time, pass
// ceres::DYNAMIC as the size template argument.
//
// Three different containers are selected in different scenarios:
//
//   num_elements == DYNAMIC:
//     -> absl::FixedArray<T, max_num_elements_on_stack>(size)
//   num_elements != DYNAMIC && num_elements <= max_num_elements_on_stack:
//     -> std::array<T, num_elements>
//   num_elements != DYNAMIC && num_elements > max_num_elements_on_stack:
//     -> absl::FixedArray<T, 0>(num_elements)
//
template <typename T,
          int num_elements,
          int max_num_elements_on_stack,
          bool fits_on_stack =
              (num_elements != DYNAMIC &&
               num_elements <= max_num_elements_on_stack)>
struct ArraySelector : std::array<T, num_elements> {
  EIGEN_STRONG_INLINE explicit ArraySelector(int s) {
    DCHECK_EQ(s, num_elements);
  }
  EIGEN_STRONG_INLINE ArraySelector(int s, const T& init) {
    DCHECK_EQ(s, num_elements);
    this->fill(init);
  }
};

template <typename T, int num_elements, int max_num_elements_on_stack>
struct ArraySelector<T, num_elements, max_num_elements_on_stack, false>
    : absl::FixedArray<
          T,
          num_elements == DYNAMIC ? max_num_elements_on_stack : 0> {
  using Base = absl::FixedArray<
      T,
      num_elements == DYNAMIC ? max_num_elements_on_stack : 0>;

  EIGEN_STRONG_INLINE explicit ArraySelector(int s) : Base(s) {
    DCHECK(num_elements == DYNAMIC || s == num_elements);
  }
  EIGEN_STRONG_INLINE ArraySelector(int s, const T& init) : Base(s, init) {
    DCHECK(num_elements == DYNAMIC || s == num_elements);
  }
};

}  // namespace ceres::internal

#endif  // CERES_PUBLIC_INTERNAL_ARRAY_SELECTOR_H_
