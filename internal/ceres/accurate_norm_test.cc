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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>

#include "ceres/internal/compensated_math.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#if defined(CERES_HAS_CPP20)
#include <version>
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
#include <bit>
#endif
#endif

namespace {

template <std::size_t N>
struct MakeInteger;

template <>
struct MakeInteger<1> {
  using type = std::int8_t;
};

template <>
struct MakeInteger<2> {
  using type = std::int16_t;
};

template <>
struct MakeInteger<4> {
  using type = std::int32_t;
};

template <>
struct MakeInteger<8> {
  using type = std::int64_t;
};

// Provide base 2 C++20 std::bit_cast fast version.
#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
// Computes the signed ULP distance between two floating-point values using
// their binary representations.
template <typename T>
constexpr auto UlpDistance2(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>,
                        typename MakeInteger<sizeof(T)>::type> {
  static_assert(std::numeric_limits<T>::is_iec559,
                "std::bit_cast based ulp distance computation requires IEC "
                "60559 floating-point bit layout");
  using std::isgreater;

  if (isgreater(a, b)) {
    return -UlpDistance2(b, a);
  }

  using Integer = typename MakeInteger<sizeof(T)>::type;

  const auto x = std::bit_cast<Integer>(a);
  const auto y = std::bit_cast<Integer>(b);

  // a <= b
  if ((x < 0) != (y < 0)) {
    if (x < 0) {
      return -(std::numeric_limits<Integer>::min() - x) + y;
    }

    // y < 0
    return (std::numeric_limits<Integer>::min() - y) + x;
  }

  return y - x;
}
#endif

// Computes the signed ULP distance between two finite floating-point values.
template <typename T>
constexpr auto UlpDistance(T a, T b)
    -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  using std::copysign;
  using std::fabs;
  using std::fmin;
  using std::fpclassify;
  using std::ilogb;
  using std::isgreater;
  using std::scalbn;
  using std::signbit;

  const int cls1 = fpclassify(a);

  if (!std::isfinite(a)) {
    throw std::domain_error{"'a' must be finite but " + std::to_string(a) +
                            " was given"};
  }

  const int cls2 = fpclassify(b);

  if (!std::isfinite(b)) {
    throw std::domain_error{"'b' must be finite but " + std::to_string(b) +
                            " was given"};
  }

  if (isgreater(a, b)) {
    return -UlpDistance(b, a);
  }

  const int cls3 = fpclassify(a - b);

  if (cls3 == FP_ZERO) {
    return T{0};
  }

  const bool s1 = signbit(a);
  const bool s2 = signbit(b);

  const bool different_signs = s1 != s2;
  const bool nonzero1 = cls2 == FP_ZERO || (cls1 != FP_ZERO && different_signs);
  const bool nonzero2 = cls1 == FP_ZERO || (cls2 != FP_ZERO && different_signs);

  if (nonzero1 || nonzero2) {
    // Either of the operands is zero prohibiting logarithm computation. Split
    // the computation and compute the distance from the denormalized minimum
    // with the sign of the operand in the direction of the operand.
    T result{0};

    // Only one of the operands can be zero at this point. However, if their
    // signs are different, we need to compute the distance in both directions
    // starting from zero.
    for (const auto [use, value] :
         {std::make_pair(nonzero1, a), std::make_pair(nonzero2, b)}) {
      if (use) {
        result +=
            T{1} +
            fabs(UlpDistance(
                copysign(std::numeric_limits<T>::denorm_min(), value), value));
      }
    }

    return result;
  }

  T result{0};

  // a, b are either both positive or both negative. Above we already ensure a <
  // b.
  if (s1) {
    return UlpDistance(-b, -a);
  }

  int e1 = cls1 == FP_SUBNORMAL ? std::numeric_limits<T>::min_exponent
                                : ilogb(a) + 1;
  const T upper1 = scalbn(T{1}, e1);

  if (isgreater(b, upper1)) {
    const int e2 = ilogb(b);
    const T upper2 = scalbn(T{1}, e2);

    result = UlpDistance(upper2, b) +
             scalbn(e2 - e1, std::numeric_limits<T>::digits - 1);
  }

  e1 = std::numeric_limits<T>::digits - e1;

  using ceres::internal::TwoSum;

  T x;
  T y;

  if (cls1 == FP_SUBNORMAL || cls3 == FP_SUBNORMAL) {
    // Avoid an underflow by scaling the denormalized values to the normal range
    const T a2 = scalbn(a, std::numeric_limits<T>::digits);
    const T b2 = scalbn(b, std::numeric_limits<T>::digits);
    const T mb = fmin(scalbn(upper1, std::numeric_limits<T>::digits), b2);
    // Account for the above scaling
    e1 -= std::numeric_limits<T>::digits;
    std::tie(x, y) = TwoSum(-mb, a2);
  } else {
    const T mb = fmin(upper1, b);
    // compute a - mb and its error
    std::tie(x, y) = TwoSum(-mb, a);
  }

  return result + scalbn(fabs(x), e1) + scalbn(fabs(y), e1);
}

}  // namespace

MATCHER_P2(MaxNumUlp, b, n, "") {
  const auto distance = UlpDistance(arg, b);
  *result_listener << "actual " << arg << " is " << distance
                   << " ULP from expected " << b << " with maximum " << n;
  return distance >= -n && distance <= n;
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
  EXPECT_EQ(ceres::AccurateRNorm(integer, floating_point, zero), 0.2);
}

TEST(AccurateNorm, CompensatedSumReportsRoundoffCorrection) {
  const auto [sum, error] =
      ceres::internal::KahanBabuskaNeumaierSum({1.0e16, 1.0});

  EXPECT_EQ(sum, 1.0e16);
  EXPECT_EQ(error, 1.0);
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
  constexpr int expected = (std::numeric_limits<double>::min_exponent - 1) / 2 +
                           1 - std::numeric_limits<double>::digits;

  EXPECT_TRUE((std::is_same_v<decltype(Traits::ScaleExponent()), int>));
  EXPECT_EQ(Traits::ScaleExponent(), expected);
}

TYPED_TEST(AccurateNormTest, FloatDistance) {
  using Scalar = TypeParam;

  EXPECT_EQ(UlpDistance(Scalar{0}, Scalar{0}), 0);

  EXPECT_EQ(
      UlpDistance(
          Scalar{0},
          std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity())),
      +1);
  EXPECT_EQ(
      UlpDistance(
          Scalar{0},
          std::nextafter(Scalar{0}, -std::numeric_limits<Scalar>::infinity())),
      -1);

  EXPECT_EQ(
      UlpDistance(std::nextafter(-std::numeric_limits<Scalar>::denorm_min(),
                                 -std::numeric_limits<Scalar>::infinity()),
                  +std::numeric_limits<Scalar>::denorm_min()),
      +3);
  EXPECT_EQ(UlpDistance(-std::numeric_limits<Scalar>::denorm_min(),
                        +std::numeric_limits<Scalar>::denorm_min()),
            +2);
  EXPECT_EQ(UlpDistance(+std::numeric_limits<Scalar>::denorm_min(),
                        -std::numeric_limits<Scalar>::denorm_min()),
            -2);

#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
  if constexpr (std::numeric_limits<Scalar>::radix == 2 &&
                requires { typename MakeInteger<sizeof(Scalar)>::type; }) {
    EXPECT_EQ(
        UlpDistance2(
            Scalar{0},
            std::nextafter(Scalar{0}, std::numeric_limits<Scalar>::infinity())),
        +1);
    EXPECT_EQ(
        UlpDistance2(Scalar{0},
                     std::nextafter(Scalar{0},
                                    -std::numeric_limits<Scalar>::infinity())),
        -1);

    EXPECT_EQ(
        UlpDistance2(std::nextafter(-std::numeric_limits<Scalar>::denorm_min(),
                                    -std::numeric_limits<Scalar>::infinity()),
                     +std::numeric_limits<Scalar>::denorm_min()),
        +3);
    EXPECT_EQ(UlpDistance2(-std::numeric_limits<Scalar>::denorm_min(),
                           +std::numeric_limits<Scalar>::denorm_min()),
              +2);
    EXPECT_EQ(UlpDistance2(+std::numeric_limits<Scalar>::denorm_min(),
                           -std::numeric_limits<Scalar>::denorm_min()),
              -2);
  }
#endif
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

TYPED_TEST(AccurateNormTest, RNorm) {
  using Scalar = TypeParam;

  EXPECT_THAT(ceres::AccurateRNorm(this->kTiny, Scalar{0}),
              MaxNumUlp(1 / this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateRNorm(this->kTiny, Scalar{0}, Scalar{0}),
              MaxNumUlp(1 / this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateRNorm(Scalar{0}, this->kTiny),
              MaxNumUlp(1 / this->kTiny, 0));
  EXPECT_THAT(ceres::AccurateRNorm(Scalar{0}, Scalar{0}, this->kTiny),
              MaxNumUlp(1 / this->kTiny, 0));

  EXPECT_THAT(ceres::AccurateRNorm(this->kTiny, this->kTiny),
              MaxNumUlp(1 / (std::sqrt(Scalar{2}) * this->kTiny), 1));

  const auto tiny3 = std::sqrt(this->kTiny) / Scalar{3};
  EXPECT_THAT(
      ceres::AccurateRNorm(tiny3, tiny3, tiny3),
      MaxNumUlp(ceres::constants::sqrt_3_v<Scalar> / std::sqrt(this->kTiny),
                1));

  const auto tiny4 = std::sqrt(this->kTiny) / Scalar{4};
  EXPECT_THAT(ceres::AccurateRNorm(tiny4, tiny4, tiny4, tiny4),
              MaxNumUlp(Scalar{2} / std::sqrt(this->kTiny), 1));

  EXPECT_THAT(ceres::AccurateRNorm(this->kHuge, Scalar{0}),
              MaxNumUlp(1 / this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateRNorm(this->kHuge, Scalar{0}, Scalar{0}),
              MaxNumUlp(1 / this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateRNorm(Scalar{0}, this->kHuge),
              MaxNumUlp(1 / this->kHuge, 0));
  EXPECT_THAT(ceres::AccurateRNorm(Scalar{0}, Scalar{0}, this->kHuge),
              MaxNumUlp(1 / this->kHuge, 0));

  EXPECT_TRUE(std::isnan(ceres::AccurateRNorm(0, 0)));

  const auto large = std::sqrt(this->kHuge / 2);
  EXPECT_THAT(ceres::AccurateRNorm(large, large),
              MaxNumUlp(1 / std::sqrt(this->kHuge), 1));

  EXPECT_EQ(ceres::AccurateRNorm(+std::numeric_limits<Scalar>::infinity(), 0),
            0);
  EXPECT_EQ(ceres::AccurateRNorm(-std::numeric_limits<Scalar>::infinity(), 0),
            0);

  EXPECT_EQ(ceres::AccurateRNorm(0, +std::numeric_limits<Scalar>::infinity()),
            0);
  EXPECT_EQ(ceres::AccurateRNorm(0, -std::numeric_limits<Scalar>::infinity()),
            0);

  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
  EXPECT_EQ(ceres::AccurateRNorm(nan, infinity), Scalar{0});
  EXPECT_EQ(ceres::AccurateRNorm(infinity, nan), Scalar{0});
  EXPECT_EQ(ceres::AccurateRNorm(nan, infinity, Scalar{1}), Scalar{0});
  EXPECT_EQ(ceres::AccurateRNorm(infinity, nan, Scalar{1}), Scalar{0});

  EXPECT_TRUE(std::isnan(
      ceres::AccurateRNorm(std::numeric_limits<Scalar>::quiet_NaN(), 0)));
  EXPECT_TRUE(std::isnan(
      ceres::AccurateRNorm(0, std::numeric_limits<Scalar>::quiet_NaN())));
}

TEST(AccurateNorm, ReciprocalNormSmallArguments) {
  constexpr double kArgument = 1e-100;
  const double expected = 1 / (std::sqrt(2.0) * kArgument);
  const double expected_three_argument = 1 / (std::sqrt(3.0) * kArgument);

  EXPECT_THAT(ceres::AccurateRNorm(kArgument, kArgument),
              MaxNumUlp(expected, 1));
  EXPECT_THAT(ceres::AccurateRNorm(kArgument, kArgument, kArgument),
              MaxNumUlp(expected_three_argument, 1));
}

TEST(AccurateNorm, ReciprocalNormIsInvariantToSignsAndOrder) {
  // The reciprocal hypotenuse is invariant under signs and argument order.
  constexpr double kExpected = 0.2;

  EXPECT_EQ(ceres::AccurateRNorm(3.0, 4.0), kExpected);
  EXPECT_EQ(ceres::AccurateRNorm(4.0, 3.0), kExpected);
  EXPECT_EQ(ceres::AccurateRNorm(-3.0, 4.0), kExpected);
  EXPECT_EQ(ceres::AccurateRNorm(3.0, -4.0), kExpected);
}

TEST(AccurateNorm, ReciprocalNormRescalesByRadixPowersWithoutRounding) {
  // These operands exercise the radix-power rescaling branches.
  constexpr int kLargeExponent = 512;
  constexpr int kSmallExponent = -514;

  EXPECT_EQ(ceres::AccurateRNorm(std::scalbn(3.0, kLargeExponent),
                                 std::scalbn(4.0, kLargeExponent)),
            std::scalbn(0.2, -kLargeExponent));
  EXPECT_EQ(ceres::AccurateRNorm(std::scalbn(3.0, kSmallExponent),
                                 std::scalbn(4.0, kSmallExponent)),
            std::scalbn(0.2, -kSmallExponent));
}

TEST(AccurateNorm, VariadicReciprocalNormOrdersByMagnitude) {
  constexpr double kFirst = -1.0;
  constexpr double kSecond = 0.5;
  constexpr double kThird = 0.25;
  const double expected = 1 / std::hypot(kFirst, kSecond, kThird);

  EXPECT_THAT(ceres::AccurateRNorm(kFirst, kSecond, kThird),
              MaxNumUlp(expected, 1));
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
  using Traits = ceres::internal::AccurateNormTraits<double>;
  constexpr double kLarger = 1.0;
  const double kSmaller = Traits::CutoffRatio();

  EXPECT_EQ(ceres::AccurateNorm(kLarger, kSmaller), kLarger);
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
  constexpr double kExpectedRNorm = 0x1.748010bb48c7cp-923;

  EXPECT_THAT(
      ceres::AccurateNorm(
          kFirst, kSecond, kThird, kFourth, kFifth, kSixth, kSeventh, kEighth),
      MaxNumUlp(kExpectedNorm, 0));
  EXPECT_THAT(
      ceres::AccurateRNorm(
          kFirst, kSecond, kThird, kFourth, kFifth, kSixth, kSeventh, kEighth),
      MaxNumUlp(kExpectedRNorm, 0));
}

TEST(AccurateNorm, VariadicNormUsesRadixPowerNormalization) {
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

TEST(AccurateNorm, VariadicRNormUsesRadixPowerNormalization) {
  constexpr double kFirst = 0x1.b5bae0f220324p-790;
  constexpr double kSecond = -0x1.dbe4984f6662p+922;
  constexpr double kThird = -0x1.1d795b706aa6ep-13;
  constexpr double kFourth = 0x1.d94bdccd99128p+611;
  constexpr double kFifth = -0x1.738b5e5278dcap-365;
  constexpr double kSixth = 0x1.49b718b5da944p-191;
  constexpr double kSeventh = 0x1.6091a40b7733p+922;
  constexpr double kExpected = 0x1.ba9cab0b4f49bp-924;

  EXPECT_THAT(ceres::AccurateRNorm(
                  kFirst, kSecond, kThird, kFourth, kFifth, kSixth, kSeventh),
              MaxNumUlp(kExpected, 0));
}

TEST(AccurateNorm, VariadicRNormAccuracyWithinOneUlp) {
  // Combination found by random search with a one ULP rounding difference.
  constexpr double kFirst = 0x1.435e0c899da16p-163;
  constexpr double kSecond = 0x1.61091418e8617p+825;
  constexpr double kThird = -0x1.a0645fc0b9c0ap+395;
  constexpr double kFourth = 0x1.013f4a26833d2p-597;
  constexpr double kFifth = 0x1.6ae50a45142b2p+238;
  constexpr double kSixth = -0x1.af001dacbda5dp-87;
  constexpr double kSeventh = -0x1.1a68ec954a99ap+163;
  constexpr double kExpected = 0x1.73457fef4890ap-826;

  EXPECT_THAT(ceres::AccurateRNorm(
                  kFirst, kSecond, kThird, kFourth, kFifth, kSixth, kSeventh),
              MaxNumUlp(kExpected, 1));
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

  const double rnorm = ceres::AccurateRNorm(nan, 1.0);
  const double variadic_rnorm = ceres::AccurateRNorm(nan, 1.0, 2.0);
  EXPECT_EQ(std::memcmp(&variadic_rnorm, &rnorm, sizeof(double)), 0);
}

#endif
