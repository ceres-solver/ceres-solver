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
// Type-parameterized tests of manifolds over unit quaternions stored in the
// order (w, x, y, z), which use the norm of the tangent vector as the rotation
// angle. A test instantiates them as
//
//   INSTANTIATE_TYPED_TEST_SUITE_P(Name,
//                                  QuaternionManifoldPlusTest,
//                                  ::testing::Types<Manifold>);

#ifndef CERES_INTERNAL_QUATERNION_MANIFOLD_TEST_UTILS_H_
#define CERES_INTERNAL_QUATERNION_MANIFOLD_TEST_UTILS_H_

#include <limits>

#include "absl/strings/str_format.h"
#include "ceres/constants.h"
#include "ceres/internal/eigen.h"
#include "ceres/manifold_test_utils.h"
#include "ceres/rotation.h"
#include "ceres/test_util.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {

// Returns a random unit quaternion.
inline Vector RandomQuaternion() {
  Vector x = Vector::Random(4);
  x.normalize();
  return x;
}

// Matches a manifold whose Plus(x, delta) is the product of the quaternion of
// the rotation by |delta| about delta and x, up to the relative distance
// tolerance.
MATCHER_P3(QuaternionPlusIsCorrectAt,
           x,
           delta,
           tolerance,
           absl::StrFormat("%s Plus(x, delta) within a relative distance of "
                           "%s of the rotated quaternion for x = %s and "
                           "delta = %s",
                           negation ? "doesn't compute" : "computes",
                           ::testing::PrintToString(tolerance),
                           ::testing::PrintToString(x.transpose()),
                           ::testing::PrintToString(delta.transpose()))) {
  // AngleAxisToQuaternion uses |delta|/2 as the rotation angle whereas the
  // quaternion manifolds use |delta| for historical reasons.
  const Vector two_delta = delta * 2;
  Vector delta_q(4);
  AngleAxisToQuaternion(two_delta.data(), delta_q.data());

  Vector expected(4);
  QuaternionProduct(delta_q.data(), x.data(), expected.data());
  Vector actual(4);
  if (!arg.Plus(x.data(), delta.data(), actual.data())) {
    *result_listener << "whose Plus fails";
    return false;
  }

  return ::testing::ExplainMatchResult(
      MatrixRelativelyNear(expected, tolerance), actual, result_listener);
}

template <typename Manifold>
class QuaternionManifoldPlusTest : public ::testing::Test {
 protected:
  static constexpr int kNumTrials = 1000;
  static constexpr double kTolerance = 1e-9;

  // Expects Plus and the manifold invariants to hold at random unit
  // quaternions for random tangent vectors scaled to the given norm.
  void ExpectPlusIsCorrectForDeltaOfNorm(double norm) const {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      const Vector x = RandomQuaternion();
      const Vector y = RandomQuaternion();
      const Vector delta = norm * Vector::Random(3).normalized();
      EXPECT_THAT(manifold_, QuaternionPlusIsCorrectAt(x, delta, kTolerance));
      EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(manifold_, x, delta, y, kTolerance);
    }
  }

  Manifold manifold_;
};

TYPED_TEST_SUITE_P(QuaternionManifoldPlusTest);

TYPED_TEST_P(QuaternionManifoldPlusTest, PlusPiBy2) {
  constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
  const Vector x = Vector::Unit(4, 0);

  for (int axis = 0; axis < 3; ++axis) {
    SCOPED_TRACE(absl::StrFormat("axis %d", axis));
    const Vector delta = constants::pi / 2 * Vector::Unit(3, axis);
    Vector x_plus_delta = Vector::Zero(4);
    ASSERT_TRUE(
        this->manifold_.Plus(x.data(), delta.data(), x_plus_delta.data()));

    // Rotating the identity by π about the axis yields the imaginary unit of
    // that axis up to the sign.
    const Vector expected = Vector::Unit(4, axis + 1);
    EXPECT_THAT(x_plus_delta,
                ::testing::AnyOf(MatrixNear(expected, kEpsilon),
                                 MatrixNear(-expected, kEpsilon)));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(
        this->manifold_, x, delta, x_plus_delta, this->kTolerance);
  }
}

TYPED_TEST_P(QuaternionManifoldPlusTest, GenericDelta) {
  for (int trial = 0; trial < this->kNumTrials; ++trial) {
    const Vector x = RandomQuaternion();
    const Vector y = RandomQuaternion();
    const Vector delta = Vector::Random(3);
    EXPECT_THAT(this->manifold_,
                QuaternionPlusIsCorrectAt(x, delta, this->kTolerance));
    EXPECT_THAT_MANIFOLD_INVARIANTS_HOLD(
        this->manifold_, x, delta, y, this->kTolerance);
  }
}

TYPED_TEST_P(QuaternionManifoldPlusTest, SmallDelta) {
  this->ExpectPlusIsCorrectForDeltaOfNorm(1e-6);
}

TYPED_TEST_P(QuaternionManifoldPlusTest, DeltaJustBelowPi) {
  this->ExpectPlusIsCorrectForDeltaOfNorm(constants::pi - 1e-6);
}

REGISTER_TYPED_TEST_SUITE_P(QuaternionManifoldPlusTest,
                            PlusPiBy2,
                            GenericDelta,
                            SmallDelta,
                            DeltaJustBelowPi);

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_QUATERNION_MANIFOLD_TEST_UTILS_H_
