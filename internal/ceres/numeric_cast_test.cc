#include "ceres/numeric_cast.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace ceres {

TEST(NumericCastTest, SignedToSigned) {
  // Casting signed to signed of the smaller size

  {
    // Casting int64_t to int32_t
    // int32_t range is [-2147483648, 2147483647]

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::int32_t>(-2147483648L - 1L), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int32_t>(2147483647L + 1L), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int32_t>(-(1L << 40)), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int32_t>(1L << 40), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::int32_t>(-2147483648L), -2147483648);
    EXPECT_EQ(numeric_cast<std::int32_t>(-2147483647L), -2147483647);
    EXPECT_EQ(numeric_cast<std::int32_t>(2147483646L), 2147483646);
    EXPECT_EQ(numeric_cast<std::int32_t>(2147483647L), 2147483647);
  }

  {
    // Casting int32_t to int8_t
    // int8_t range is [-128, 127]

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::int8_t>(-130), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int8_t>(-129), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int8_t>(128), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int8_t>(129), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::int8_t>(-128), -128);
    EXPECT_EQ(numeric_cast<std::int8_t>(-127), -127);
    EXPECT_EQ(numeric_cast<std::int8_t>(126), 126);
    EXPECT_EQ(numeric_cast<std::int8_t>(127), 127);
  }
}

TEST(NumericCastTest, SignedToUnsigned) {
  {
    // Casting signed to unsigned of the higher size
    // int32_t ([-2147483648, 2147483647]) to uint64_t ([0, 18446744073709551615])

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::uint64_t>(-2147483648), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::uint64_t>(-1), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::uint64_t>(0), 0u);
    EXPECT_EQ(numeric_cast<std::uint64_t>(1), 1u);
    EXPECT_EQ(numeric_cast<std::uint64_t>(2147483646), 2147483646u);
    EXPECT_EQ(numeric_cast<std::uint64_t>(2147483647), 2147483647u);
  }

  {
    // Casting signed to unsigned of the same size
    // int32_t ([-2147483648, 2147483647]) to uint32_t ([0, 4294967295])

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::uint32_t>(-2147483648), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::uint32_t>(-1), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::uint32_t>(0), 0u);
    EXPECT_EQ(numeric_cast<std::uint32_t>(1), 1u);
    EXPECT_EQ(numeric_cast<std::uint32_t>(2147483646), 2147483646u);
    EXPECT_EQ(numeric_cast<std::uint32_t>(2147483647), 2147483647u);
  }

  {
    // Casting signed to unsigned of the lower size
    // int32_t ([-2147483648, 2147483647]) to uint16_t ([0, 65535])

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::uint16_t>(-2147483648), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::uint16_t>(-32768), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::uint16_t>(-1), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::uint16_t>(65536), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::uint16_t>(0), 0u);
    EXPECT_EQ(numeric_cast<std::uint16_t>(127), 127u);
    EXPECT_EQ(numeric_cast<std::uint16_t>(32767), 32767u);
    EXPECT_EQ(numeric_cast<std::uint16_t>(65535), 65535u);
  }
}

TEST(NumericCastTest, UnsignedToSigned) {
  {
    // Casting unsigned to signed of the same size
    // uint32_t ([0, 4294967295]) to int32_t ([-2147483648, 2147483647])

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::int32_t>(2147483648u), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int32_t>(4294967295u), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::int32_t>(0u), 0);
    EXPECT_EQ(numeric_cast<std::int32_t>(2147483647u), 2147483647);
  }

  {
    // Casting unsigned to signed of the smaller size
    // uint32_t ([0, 4294967295]) to int16_t ([-32768, 32767])

    // Incorrect casts
    EXPECT_THROW(numeric_cast<std::int16_t>(32768u), std::runtime_error);
    EXPECT_THROW(numeric_cast<std::int16_t>(4294967295u), std::runtime_error);

    // Correct casts
    EXPECT_EQ(numeric_cast<std::int16_t>(0u), 0);
    EXPECT_EQ(numeric_cast<std::int16_t>(32767u), 32767);
  }

  {
    // Casting unsigned to signed of the higher size (it is acutally a safe cast)
    // uint32_t ([0, 4294967295]) to int64_t ([-9223372036854775808, 9223372036854775807])
    EXPECT_EQ(numeric_cast<std::int64_t>(0u), 0L);
    EXPECT_EQ(numeric_cast<std::int64_t>(4294967295u), 4294967295L);
  }
}

TEST(NumericCastTest, UnsignedToUnsigned) {
  // Casting unsigned to unsigned of the smaller size
  // uint32_t ([0, 4294967295]) to uint16_t ([0, 65535])

  // Incorrect casts
  EXPECT_THROW(numeric_cast<std::uint16_t>(65536u), std::runtime_error);
  EXPECT_THROW(numeric_cast<std::uint16_t>(4294967295u), std::runtime_error);

  // Correct casts
  EXPECT_EQ(numeric_cast<std::uint16_t>(0u), 0u);
  EXPECT_EQ(numeric_cast<std::uint16_t>(65535u), 65535u);
}

}  // namespace ceres
