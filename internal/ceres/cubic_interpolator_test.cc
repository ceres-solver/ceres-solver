// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/cubic_interpolator.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(CubicInterpolator1, NeedsAtleastTwoValues) {
  double x[] = {1};
  EXPECT_DEATH_IF_SUPPORTED(CubicInterpolator1 c(0, x), "num_values > 1");
  EXPECT_DEATH_IF_SUPPORTED(CubicInterpolator1 c(1, x), "num_values > 1");
}

TEST(CubicInterpolator1, TwoValues) {
  double x[] = {0, 1};
  CubicInterpolator1 interpolator(2, x);
  double f, fx;
  EXPECT_TRUE(interpolator.Evaluate(0.0, &f, &fx));
  EXPECT_NEAR(f, x[0], 1e-16);
  EXPECT_TRUE(interpolator.Evaluate(1.0, &f, &fx));
  EXPECT_NEAR(f, x[1], 1e-16);
}

}  // namespace internal
}  // namespace ceres
