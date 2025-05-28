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

#include "ceres/accurate_norm.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "ceres/constants.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

// Matches if the argument is at most n floating-point values away from the
// expected value.
MATCHER_P2(MaxNumUlp, expected, n, "") {
  using Scalar = std::decay_t<decltype(arg)>;
  const Scalar target = static_cast<Scalar>(expected);
  Scalar value = arg;

  for (int distance = 0; value != target && distance < n; ++distance) {
    value = std::nextafter(value, target);
  }

  *result_listener << "actual " << arg << " is not within " << n
                   << " ULP of expected " << target;
  return value == target;
}

TEST(AccurateNorm, Promote) {
  static_assert(std::is_same_v<ceres::internal::Promote_t<int>, double>,
                "Promotion of an int must be a double");
  static_assert(std::is_same_v<ceres::internal::Promote_t<int, int>, double>,
                "Promotion of multuple ints must be a double");
  static_assert(std::is_same_v<ceres::internal::Promote_t<unsigned>, double>,
                "Promotion of an unsigned int must be double");
  static_assert(std::is_same_v<ceres::internal::Promote_t<long>, double>,
                "Promotion of a long must be double");
  static_assert(
      std::is_same_v<ceres::internal::Promote_t<int, long, float>, double>,
      "Promotion of arithmetic types must be double");
}

TEST(AccurateNorm, PromotesLvalueArguments) {
  int integer = 3;
  double floating_point = 4.0;
  double zero = 0.0;

  EXPECT_EQ(ceres::AccurateNorm(integer, floating_point, zero), 5.0);
}

TEST(AccurateNorm, PromotesIntegralArgumentsOfTheSameType) {
  static_assert(std::is_same_v<decltype(ceres::AccurateNorm(3, 4)), double>);
  static_assert(std::is_same_v<decltype(ceres::AccurateNorm(3, 4, 0)), double>);
  static_assert(
      std::is_same_v<decltype(ceres::AccurateNorm(3.0F, 4.0F, 0.0F)), float>);

  EXPECT_EQ(ceres::AccurateNorm(3, 4), 5.0);
  EXPECT_EQ(ceres::AccurateNorm(3, 4, 0), 5.0);
}

#if GTEST_HAS_TYPED_TEST

template <typename T>
class AccurateNormTest : public testing::Test {
 public:
  static constexpr auto kTiny = std::numeric_limits<T>::min();
  static constexpr auto kHuge = std::numeric_limits<T>::max();
};

using Types = testing::Types<float, double, long double>;

TYPED_TEST_SUITE(AccurateNormTest, Types);

TEST(AccurateNorm, ScaleIsRadixExponent) {
  using Traits = ceres::internal::AccurateNormTraits<double>;

  // ulp(√F_min) = ulp(2^-511) = 2^-563 for double
  static_assert(std::is_same_v<decltype(Traits::ScaleExponent()), int>);
  static_assert(Traits::ScaleExponent() == -563);
}

TYPED_TEST(AccurateNormTest, Norm) {
  using Scalar = TypeParam;

  EXPECT_THAT(ceres::AccurateNorm(this->kTiny, Scalar{0}),
              MaxNumUlp(this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateNorm(this->kTiny, Scalar{0}, Scalar{0}),
              MaxNumUlp(this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateNorm(Scalar{0}, this->kTiny),
              MaxNumUlp(this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateNorm(Scalar{0}, Scalar{0}, this->kTiny),
              MaxNumUlp(this->kTiny, 0));

  EXPECT_THAT(ceres::AccurateNorm(this->kHuge, Scalar{0}),
              MaxNumUlp(this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateNorm(this->kHuge, Scalar{0}, Scalar{0}),
              MaxNumUlp(this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateNorm(Scalar{0}, this->kHuge),
              MaxNumUlp(this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateNorm(Scalar{0}, Scalar{0}, this->kHuge),
              MaxNumUlp(this->kHuge, 0));

  EXPECT_THAT(ceres::AccurateNorm(this->kTiny, this->kTiny),
              MaxNumUlp(this->kTiny * std::sqrt(Scalar{2}), 1));

  EXPECT_THAT(ceres::AccurateNorm(Scalar{0}, Scalar{0}),
              MaxNumUlp(Scalar{0}, 0));

  EXPECT_TRUE(std::isinf(
      ceres::AccurateNorm(+std::numeric_limits<Scalar>::infinity(), 0)));
  EXPECT_TRUE(std::isinf(
      ceres::AccurateNorm(-std::numeric_limits<Scalar>::infinity(), 0)));

  EXPECT_TRUE(std::isinf(
      ceres::AccurateNorm(0, +std::numeric_limits<Scalar>::infinity())));
  EXPECT_TRUE(std::isinf(
      ceres::AccurateNorm(0, -std::numeric_limits<Scalar>::infinity())));

  EXPECT_TRUE(std::isnan(
      ceres::AccurateNorm(std::numeric_limits<Scalar>::quiet_NaN(), 0)));
  EXPECT_TRUE(std::isnan(
      ceres::AccurateNorm(0, std::numeric_limits<Scalar>::quiet_NaN())));
}

TEST(AccurateNorm, VariadicNormAccuracy) {
  EXPECT_THAT(ceres::AccurateNorm(1.0, 1.0, 1.0),
              MaxNumUlp(ceres::constants::sqrt_3, 0));

  // Combination exposing a difference of at least two ULPs in inaccurate
  // implementations found by random search.
  constexpr double kFirst = 0.0;
  constexpr double kSecond = -0x1.c4a46e8d5e9f3p-940;
  constexpr double kThird = 0x1.2870a0a1f3fa3p-943;
  constexpr double kExpected = 0x1.c628110110bf1p-940;
  EXPECT_THAT(ceres::AccurateNorm(kFirst, kSecond, kThird),
              MaxNumUlp(kExpected, 0));
}

TEST(AccurateNorm, HandlesTableMakerDilemma) {
  // Values from Borges, Algorithm 1014, Section 6.
  constexpr double kFirst = 0x1.a308e1455f447p+0;
  constexpr double kSecond = 0x1.9d931a83ef879p+0;
  constexpr double kExpected = 0x1.2660d009d54f9p+1;

  EXPECT_THAT(ceres::AccurateNorm(kFirst, kSecond), MaxNumUlp(kExpected, 1));
}

TEST(AccurateNorm, ReturnsLargerArgumentWhenSmallerIsNegligible) {
  constexpr double kLarger = 1.0;
  // √(ε/2) is the largest ratio of the arguments for which the norm equals the
  // larger argument.
  constexpr double kSmaller = 0x1.6a09e667f3bcdp-27;

  EXPECT_EQ(ceres::AccurateNorm(kLarger, kSmaller), kLarger);
}

TYPED_TEST(AccurateNormTest, AccountsForSmallerArgumentAboveCutoff) {
  using Scalar = TypeParam;
  // The squared ratio of the arguments equals the machine epsilon which raises
  // the norm by three quarters of an ULP above the larger argument.
  constexpr Scalar kLarger{1.5};
  const Scalar kSmaller =
      kLarger * std::sqrt(std::numeric_limits<Scalar>::epsilon());
  const Scalar kExpected =
      std::nextafter(kLarger, std::numeric_limits<Scalar>::infinity());

  EXPECT_EQ(ceres::AccurateNorm(kLarger, kSmaller), kExpected);
}

TEST(AccurateNorm, RecoversSquaringErrorsOfSmallArguments) {
  // The arguments are not rescaled but their squares are small enough for the
  // rounding errors of the squares to underflow unless the scaling threshold
  // accounts for them.
  EXPECT_EQ(ceres::AccurateNorm(0x1.23342ep-63f, 0x1.271008p-63f),
            0x1.9e8fdcp-63f);
  EXPECT_EQ(ceres::AccurateNorm(0x1.4a15545f86ef4p-510, 0x1.3a578a98bcf9dp-510),
            0x1.c7d03b992e6a8p-510);
#if LDBL_MANT_DIG == 64 && LDBL_MIN_EXP == -16381
  EXPECT_EQ(ceres::AccurateNorm(0x8.b912469c50fab7ap-8194L,
                                0x8.c54f2ded5ba8268p-8194L),
            0xc.5eb4864ce876f86p-8194L);
#endif
}

TEST(AccurateNorm, IsAtLeastAsAccurateAsHypot) {
  // The expected values are correctly rounded. Implementations of std::hypot,
  // e.g., the one shipped with libstdc++ 16, are off by one ULP for these
  // arguments.
  constexpr double kFirst = 0x1.424bf2ed916bfp+0;
  constexpr double kSecond = 0x1.1435107c5d458p+0;
  constexpr double kExpected = 0x1.a8758df39043fp+0;

  EXPECT_EQ(ceres::AccurateNorm(kFirst, kSecond), kExpected);
  EXPECT_LE(std::fabs(ceres::AccurateNorm(kFirst, kSecond) - kExpected),
            std::fabs(std::hypot(kFirst, kSecond) - kExpected));

  constexpr double kThreeFirst = 0x1.c11f6531eb66ep+0;
  constexpr double kThreeSecond = 0x1.f30567547a34cp+0;
  constexpr double kThreeThird = 0x1.1e0edcc120696p+0;
  constexpr double kThreeExpected = 0x1.6ce2663da03a7p+1;

  EXPECT_EQ(ceres::AccurateNorm(kThreeFirst, kThreeSecond, kThreeThird),
            kThreeExpected);
  EXPECT_LE(
      std::fabs(ceres::AccurateNorm(kThreeFirst, kThreeSecond, kThreeThird) -
                kThreeExpected),
      std::fabs(std::hypot(kThreeFirst, kThreeSecond, kThreeThird) -
                kThreeExpected));
}

TYPED_TEST(AccurateNormTest, PowerOfTwoIsExact) {
  using Scalar = TypeParam;
  using std::scalbn;
  constexpr int kMinExponent = std::numeric_limits<Scalar>::min_exponent -
                               std::numeric_limits<Scalar>::digits;
  constexpr int kMaxExponent = std::numeric_limits<Scalar>::max_exponent - 1;

  for (int exponent = kMinExponent; exponent <= kMaxExponent; ++exponent) {
    EXPECT_EQ(ceres::internal::PowerOfTwo<Scalar>(exponent),
              scalbn(Scalar{1}, exponent))
        << "exponent " << exponent;
  }
}

TEST(AccurateNorm, TraitsAreConstantExpressions) {
  // Compilers that do not fold std::scalbn would otherwise compute the
  // thresholds at runtime on every invocation.
  using DoubleTraits = ceres::internal::AccurateNormTraits<double>;
  static_assert(DoubleTraits::Tiny() == 0x1p-485);
  static_assert(DoubleTraits::UnscaledMinimum() == 0x1p-432);
  static_assert(DoubleTraits::UnscaledMaximum(2) == 0x1p+511);
  static_assert(DoubleTraits::UnscaledMaximum(3) == 0x1p+510);

  using FloatTraits = ceres::internal::AccurateNormTraits<float>;
  static_assert(FloatTraits::Tiny() == 0x1p-51f);
  static_assert(FloatTraits::UnscaledMinimum() == 0x1p-27f);
  static_assert(FloatTraits::UnscaledMaximum(3) == 0x1p+62f);
}

TEST(AccurateNorm, VariadicNormIsAccurateForExtremeMagnitudes) {
  // Subnormal arguments
  EXPECT_EQ(ceres::AccurateNorm(0x0.123456789abcdp-1022,
                                0x0.fedcba9876543p-1022,
                                0x0.0000000000001p-1022),
            0x0.ff82f53036b9cp-1022);
  // Small arguments including a negligible one
  EXPECT_EQ(ceres::AccurateNorm(
                0x1.3a578a98bcf9dp-600, -0x1.4a15545f86ef4p-601, 0x1.5p-650),
            0x1.6308e012498e5p-600);
  // Arguments close to the largest finite value
  EXPECT_EQ(ceres::AccurateNorm(0x1.ffffffffffffp+1022,
                                0x1.4a15545f86ef4p+1021,
                                -0x1.3a578a98bcf9dp+1000),
            0x1.0cf8b69a0aff1p+1023);
  // Large arguments including a negligible one
  EXPECT_EQ(ceres::AccurateNorm(0x1.8p+1000, 0x1p-1000, 0x1.4p+999),
            0x1.ap+1000);
  // The norm exceeds the largest finite value
  EXPECT_EQ(ceres::AccurateNorm(0x1.ep+1023, 0x1.ep+1023, 0x1.ep+1023),
            std::numeric_limits<double>::infinity());
}

TEST(AccurateNorm, VariadicNormIsAccurateAroundUnscaledRange) {
  constexpr double kSqrt3Largest = 0x1.bb67ae8584caap+510;
  constexpr double kSqrt3Smallest = 0x1.bb67ae8584caap-432;

  // Inside the range that requires no rescaling
  EXPECT_EQ(ceres::AccurateNorm(0x1p+510, 0x1p+510, 0x1p+510), kSqrt3Largest);
  EXPECT_EQ(ceres::AccurateNorm(0x1p-432, 0x1p-432, 0x1p-432), kSqrt3Smallest);
  EXPECT_EQ(ceres::AccurateNorm(
                0x1.3a578a98bcf9dp+509, -0x1.4a15545f86ef4p+510, 0x1.5p-200),
            0x1.6d979f8e15b5ap+510);

  // Just outside the range that requires no rescaling
  EXPECT_EQ(ceres::AccurateNorm(0x1.fffffffffffffp+510,
                                0x1.fffffffffffffp+510,
                                0x1.fffffffffffffp+510),
            0x1.bb67ae8584caap+511);
  EXPECT_EQ(ceres::AccurateNorm(0x1.fffffffffffffp-433,
                                0x1.fffffffffffffp-433,
                                0x1.fffffffffffffp-433),
            kSqrt3Smallest);
  EXPECT_EQ(ceres::AccurateNorm(
                0x1.4a15545f86ef4p-432, -0x1.3a578a98bcf9dp-433, 0x1.5p-500),
            0x1.6d979f8e15b5ap-432);
}

TEST(AccurateNorm, RescalesByRadixPowersWithoutRounding) {
  constexpr int kLargeExponent = 512;
  constexpr int kSmallExponent = -514;

  EXPECT_EQ(ceres::AccurateNorm(std::scalbn(3.0, kLargeExponent),
                                std::scalbn(4.0, kLargeExponent)),
            std::scalbn(5.0, kLargeExponent));
  EXPECT_EQ(ceres::AccurateNorm(std::scalbn(3.0, kSmallExponent),
                                std::scalbn(4.0, kSmallExponent)),
            std::scalbn(5.0, kSmallExponent));
}

TEST(AccurateNorm, VariadicNormHandlesWideDynamicRange) {
  // Combination found by random search that exposes a difference of more than
  // two ULPs when normalized values are not stored.
  constexpr double kFirst = -0x1.5fdef349a2773p+922;
  constexpr double kSecond = -0x1.72ec46b66e1d9p-114;
  constexpr double kThird = -0x1.eeb9ef28337eep-462;
  constexpr double kFourth = -0x1.137462de2cf44p+205;
  constexpr double kFifth = -0x1.3a6cb8edf6622p+264;
  constexpr double kSixth = 0x1.0f5b36fe2970ap-326;
  constexpr double kSeventh = -0x1.5483d7f40eca8p-537;
  constexpr double kEighth = 0x1.4694b77d1bb38p-643;
  constexpr double kExpectedNorm = 0x1.5fdef349a2773p+922;

  EXPECT_THAT(
      ceres::AccurateNorm(
          kFirst, kSecond, kThird, kFourth, kFifth, kSixth, kSeventh, kEighth),
      MaxNumUlp(kExpectedNorm, 0));
}

TEST(AccurateNorm, VariadicNormRescalesLargeArguments) {
  constexpr double kFirst = -0x1.5cf602c1b383ep-360;
  constexpr double kSecond = 0x1.e7855aa96a0c7p+850;
  constexpr double kThird = -0x1.e3bf678284e4p+876;
  constexpr double kFourth = 0x1.7d18d55d5723fp+173;
  constexpr double kFifth = -0x1.1a6c3b4a7f1f8p+437;
  constexpr double kSixth = -0x1.14b8d80b68457p-561;
  constexpr double kExpectedNorm = 0x1.e3bf678284e41p+876;

  EXPECT_THAT(
      ceres::AccurateNorm(kFirst, kSecond, kThird, kFourth, kFifth, kSixth),
      MaxNumUlp(kExpectedNorm, 0));
}

TEST(AccurateNorm, VariadicNormReturnsPositiveInfinity) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();

  EXPECT_EQ(ceres::AccurateNorm(-kInfinity, 1.0, 2.0), kInfinity);
}

TEST(AccurateNorm, NonfiniteArgumentHandling) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

  EXPECT_EQ(ceres::AccurateNorm(kNaN, kInfinity), kInfinity);
  EXPECT_EQ(ceres::AccurateNorm(kInfinity, kNaN), kInfinity);
  EXPECT_EQ(ceres::AccurateNorm(1.0, kInfinity, kNaN), kInfinity);
  EXPECT_EQ(ceres::AccurateNorm(kNaN, 1.0, kInfinity), kInfinity);
  EXPECT_EQ(ceres::AccurateNorm(kNaN, kInfinity, 1.0), kInfinity);
  EXPECT_EQ(ceres::AccurateNorm(kInfinity, kNaN, 1.0), kInfinity);

  EXPECT_TRUE(std::isnan(ceres::AccurateNorm(kNaN, kNaN, 1.0)));
  EXPECT_TRUE(std::isnan(ceres::AccurateNorm(1.0, kNaN, kNaN)));
  EXPECT_TRUE(std::isnan(ceres::AccurateNorm(kNaN, 1.0, kNaN)));
}

TEST(AccurateNorm, PreservesNaNPayloadAcrossArity) {
  const double nan = std::copysign(std::nan("12345"), -1.0);

  const double norm = ceres::AccurateNorm(nan, 1.0);
  const double variadic_norm = ceres::AccurateNorm(nan, 1.0, 2.0);
  EXPECT_EQ(std::memcmp(&variadic_norm, &norm, sizeof(double)), 0);
}

TEST(AccurateNorm, PreservesNaNPayloadOfAnyArgument) {
  const double nan = std::copysign(std::nan("12345"), -1.0);
  const double expected = std::fabs(nan);

  const double first = ceres::AccurateNorm(nan, 1.0, 2.0);
  const double last = ceres::AccurateNorm(1.0, 2.0, nan);
  EXPECT_EQ(std::memcmp(&first, &expected, sizeof(double)), 0);
  EXPECT_EQ(std::memcmp(&last, &expected, sizeof(double)), 0);
}

#endif
