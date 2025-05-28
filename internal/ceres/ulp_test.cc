// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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

#include "ceres/internal/ulp.h"

#include <cmath>
#include <limits>

#include "gtest/gtest.h"

namespace {

template <typename T>
class UlpTest : public testing::Test {};

using Types = testing::Types<float, double, long double>;
TYPED_TEST_SUITE(UlpTest, Types);

TYPED_TEST(UlpTest, SpecialValues) {
  using Scalar = TypeParam;

  EXPECT_TRUE(std::isnan(
      ceres::internal::Ulp(std::numeric_limits<Scalar>::quiet_NaN())));
  EXPECT_TRUE(std::isinf(
      ceres::internal::Ulp(+std::numeric_limits<Scalar>::infinity())));
  EXPECT_TRUE(std::isinf(
      ceres::internal::Ulp(-std::numeric_limits<Scalar>::infinity())));
  EXPECT_EQ(ceres::internal::Ulp(Scalar{0}),
            std::numeric_limits<Scalar>::denorm_min());
}

TYPED_TEST(UlpTest, NormalAndSubnormalSpacing) {
  using Scalar = TypeParam;
  using Limits = std::numeric_limits<Scalar>;

  EXPECT_EQ(ceres::internal::Ulp(Scalar{1}), Limits::epsilon());
  EXPECT_EQ(ceres::internal::Ulp(Scalar{2}), Scalar{2} * Limits::epsilon());

  if constexpr (Limits::has_denorm != std::denorm_absent) {
    const Scalar denorm_min = Limits::denorm_min();
    EXPECT_EQ(ceres::internal::Ulp(denorm_min), denorm_min);
    EXPECT_EQ(ceres::internal::Ulp(Scalar{2} * denorm_min), denorm_min);
  }
}

}  // namespace
