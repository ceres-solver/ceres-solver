// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//
// Tests the use of Cere's Jet type with the quaternions found in util/math/. In
// theory, the unittests for the quaternion class should be type parameterized
// to make for easier testing of instantiations of the quaternion class, but it
// is not so, and not obviously worth the work to make the switch at this time.

#include "base/stringprintf.h"
#include "gtest/gtest.h"
#include "util/math/mathlimits.h"
#include "util/math/matrix3x3-inl.h"
#include "util/math/quaternion-inl.h"
#include "util/math/vector3-inl.h"
#include "ceres/test_util.h"
#include "ceres/jet.h"
#include "ceres/jet_traits.h"

namespace ceres {
namespace internal {

// Use a 4-element derivative to simulate the case where each of the
// quaternion elements are derivative parameters.
typedef Jet<double, 4> J;

struct JetTraitsTest : public ::testing::Test {
 protected:
  JetTraitsTest()
      : a(J(1.1, 0), J(2.1, 1), J(3.1, 2), J(4.1, 3)),
        b(J(0.1, 0), J(1.1, 1), J(2.1, 2), J(5.0, 3)),
        double_a(a[0].a, a[1].a, a[2].a, a[3].a),
        double_b(b[0].a, b[1].a, b[2].a, b[3].a) {
    // The quaternions should be valid rotations, so normalize them.
    a.Normalize();
    b.Normalize();
    double_a.Normalize();
    double_b.Normalize();
  }

  virtual ~JetTraitsTest() {}

  // A couple of arbitrary normalized quaternions.
  Quaternion<J> a, b;

  // The equivalent of a, b but in scalar form.
  Quaternion<double> double_a, double_b;
};

// Compare scalar multiplication to jet multiplication. Ignores derivatives.
TEST_F(JetTraitsTest, QuaternionScalarMultiplicationWorks) {
  Quaternion<J> c = a * b;
  Quaternion<double> double_c = double_a * double_b;

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(double_c[i], c[i].a);
  }
}

// Compare scalar slerp to jet slerp. Ignores derivatives.
TEST_F(JetTraitsTest, QuaternionScalarSlerpWorks) {
  const J fraction(0.1);
  Quaternion<J> c = Quaternion<J>::Slerp(a, b, fraction);
  Quaternion<double> double_c =
      Quaternion<double>::Slerp(double_a, double_b, fraction.a);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(double_c[i], c[i].a);
  }
}

// On a 32-bit optimized build, the mismatch is about 1.4e-14.
double const kTolerance = 1e-13;

void ExpectJetsClose(const J &x, const J &y) {
  ExpectClose(x.a, y.a, kTolerance);
  ExpectClose(x.v[0], y.v[0], kTolerance);
  ExpectClose(x.v[1], y.v[1], kTolerance);
  ExpectClose(x.v[2], y.v[2], kTolerance);
  ExpectClose(x.v[3], y.v[3], kTolerance);
}

void ExpectQuaternionsClose(const Quaternion<J>& x, const Quaternion<J>& y) {
  for (int i = 0; i < 4; ++i) {
    ExpectJetsClose(x[i], y[i]);
  }
}

// Compare jet slurp to jet slerp using identies, checking derivatives.
TEST_F(JetTraitsTest, CheckSlerpIdentitiesWithNontrivialDerivatives) {
  // Do a slerp to 0.75 directly.
  Quaternion<J> direct = Quaternion<J>::Slerp(a, b, J(0.75));

  // Now go part way twice, in theory ending at the same place.
  Quaternion<J> intermediate = Quaternion<J>::Slerp(a, b, J(0.5));
  Quaternion<J> indirect = Quaternion<J>::Slerp(intermediate, b, J(0.5));

  // Check that the destination is the same, including derivatives.
  ExpectQuaternionsClose(direct, indirect);
}

TEST_F(JetTraitsTest, CheckAxisAngleIsInvertibleWithNontrivialDerivatives) {
  Vector3<J> axis;
  J angle;
  a.GetAxisAngle(&axis, &angle);
  b.SetFromAxisAngle(axis, angle);

  ExpectQuaternionsClose(a, b);
}

TEST_F(JetTraitsTest,
       CheckRotationMatrixIsInvertibleWithNontrivialDerivatives) {
  Vector3<J> axis;
  J angle;
  Matrix3x3<J> R;
  a.ToRotationMatrix(&R);
  b.SetFromRotationMatrix(R);

  ExpectQuaternionsClose(a, b);
}

// This doesn't check correctnenss, only that the instantiation compiles.
TEST_F(JetTraitsTest, CheckRotationBetweenIsCompilable) {
  // Get two arbitrary vectors x and y.
  Vector3<J> x, y;
  J ignored_angle;
  a.GetAxisAngle(&x, &ignored_angle);
  b.GetAxisAngle(&y, &ignored_angle);

  Quaternion<J> between_x_and_y = Quaternion<J>::RotationBetween(x, y);

  // Prevent optimizing this away.
  EXPECT_NE(between_x_and_y[0].a, 0.0);
}

TEST_F(JetTraitsTest, CheckRotatedWorksAsExpected) {
  // Get two arbitrary vectors x and y.
  Vector3<J> x;
  J ignored_angle;
  a.GetAxisAngle(&x, &ignored_angle);

  // Rotate via a quaternion.
  Vector3<J> y = b.Rotated(x);

  // Rotate via a rotation matrix.
  Matrix3x3<J> R;
  b.ToRotationMatrix(&R);
  Vector3<J> yp = R * x;

  ExpectJetsClose(yp[0], y[0]);
  ExpectJetsClose(yp[1], y[1]);
  ExpectJetsClose(yp[2], y[2]);
}

TEST_F(JetTraitsTest, CheckRotatedWorksAsExpectedWithDoubles) {
  // Get two arbitrary vectors x and y.
  Vector3<double> x;
  double ignored_angle;
  double_a.GetAxisAngle(&x, &ignored_angle);

  // Rotate via a quaternion.
  Vector3<double> y = double_b.Rotated(x);

  // Rotate via a rotation matrix.
  Matrix3x3<double> R;
  double_b.ToRotationMatrix(&R);
  Vector3<double> yp = R * x;

  ExpectClose(yp[0], y[0], kTolerance);
  ExpectClose(yp[1], y[1], kTolerance);
  ExpectClose(yp[2], y[2], kTolerance);
}

}  // namespace internal
}  // namespace ceres
