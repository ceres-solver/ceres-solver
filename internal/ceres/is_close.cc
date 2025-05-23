// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2025 Google Inc. All rights reserved.
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
// Authors: keir@google.com (Keir Mierle), dgossow@google.com (David Gossow)

#include "ceres/is_close.h"

#include <algorithm>
#include <cmath>

namespace ceres::internal {
bool IsClose(double x,
             double y,
             double relative_precision,
             double* relative_error,
             double* absolute_error) {
  double local_absolute_error;
  double local_relative_error;
  if (absolute_error == nullptr) {
    absolute_error = &local_absolute_error;
  }
  if (relative_error == nullptr) {
    relative_error = &local_relative_error;
  }
  *absolute_error = std::fabs(x - y);
  if (std::fpclassify(x) == FP_ZERO || std::fpclassify(y) == FP_ZERO) {
    // If x or y is exactly zero, then relative difference doesn't have any
    // meaning. Take the absolute difference instead.
    *relative_error = *absolute_error;
  } else {
    *relative_error = *absolute_error / std::max(std::fabs(x), std::fabs(y));
  }
  return *relative_error < std::fabs(relative_precision);
}
}  // namespace ceres::internal
