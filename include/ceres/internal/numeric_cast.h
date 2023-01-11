#ifndef CERES_PUBLIC_NUMERIC_CAST_
#define CERES_PUBLIC_NUMERIC_CAST_

#include <limits>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace ceres {

template <typename TargetT, typename SourceT>
static inline std::runtime_error MakeOutOfRangeError(const SourceT src) noexcept {
  return std::runtime_error(
      "Value " + std::to_string(src) + " is out of target type range [" +
       std::to_string(std::numeric_limits<TargetT>::min()) + ", " +
       std::to_string(std::numeric_limits<TargetT>::max()) + "]");
}

/** In numeric_casts"s below we intentionally disable to/from char/bool conversions.
 * 1. char. C++ standard does not specify whether char is signed or unsigned. It is even not
 *  guaranteed that expression (std::is_same_v<char, signed char> ||
 *  std::is_same_v<char, unsinged char>) is true. Moreover it is not so on the most systems.
 *  Because of this unclear status of char it is better do not use "numeric_cast" for casts
 *  from/to char.
 * 2. bool. It is a logical value not a numeric one but "numeric_cast" is designed for safe casts
 *  between numbers. If you need to convert numeric value to a logical one please make it
 *  explicitely by comparison with 0.
 */

// Signed integer to signed integer of the smaller size
// int64_t -> int32_t
// int32_t -> int16_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_signed_v<SourceT> &&
    std::is_signed_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) < sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  if (src < static_cast<SourceT>(std::numeric_limits<TargetT>::min()) ||
      src > static_cast<SourceT>(std::numeric_limits<TargetT>::max())) {
    throw MakeOutOfRangeError<TargetT>(src);
  }

  return static_cast<TargetT>(src);
}

// Signed integer to signed integer of the same or higher size
// int32_t -> int32_t
// int32_t -> int64_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_signed_v<SourceT> &&
    std::is_signed_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) >= sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  return static_cast<TargetT>(src);
}

// Unsigned integer to unsigned integer of the smaller size
// uint64_t -> uint32_t
// uint32_t -> uint16_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_unsigned_v<SourceT> &&
    std::is_unsigned_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) < sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  if (src > static_cast<SourceT>(std::numeric_limits<TargetT>::max())) {
    throw MakeOutOfRangeError<TargetT>(src);
  }

  return static_cast<TargetT>(src);
}

// Signed integer to unsigned integer of smaller size
// int64_t -> uint32_t
// int32_t -> uint16_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_signed_v<SourceT> &&
    std::is_unsigned_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) < sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  if (src < static_cast<SourceT>(0) ||
      src > static_cast<SourceT>(std::numeric_limits<TargetT>::max())) {
    throw MakeOutOfRangeError<TargetT>(src);
  }

  return static_cast<TargetT>(src);
}

// Signed integer to unsigned integer of the same or higher size
// int64_t -> uint64_t
// int32_t -> uint64_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_signed_v<SourceT> &&
    std::is_unsigned_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) >= sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  if (src < static_cast<SourceT>(0)) {
    throw MakeOutOfRangeError<TargetT>(src);
  }

  return static_cast<TargetT>(src);
}

// Unsigned to signed of the same or smaller size
// uint64_t -> int64_t
// uint64_t -> int32_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_unsigned_v<SourceT> &&
    std::is_signed_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    sizeof(TargetT) <= sizeof(SourceT),
    TargetT>
numeric_cast(SourceT src) {
  if (src > static_cast<SourceT>(std::numeric_limits<TargetT>::max())) {
    throw MakeOutOfRangeError<TargetT>(src);
  }

  return static_cast<TargetT>(src);
}

// Unsigned to signed of the higher size
// uint32_t -> int64_t
// uint16_t -> int32_t
template <typename TargetT, typename SourceT>
constexpr std::enable_if_t<
    std::is_integral_v<SourceT> &&
    std::is_integral_v<TargetT> &&
    std::is_unsigned_v<SourceT> &&
    std::is_signed_v<TargetT> &&
    !std::is_same_v<SourceT, bool> &&
    !std::is_same_v<SourceT, char> &&
    !std::is_same_v<TargetT, bool> &&
    !std::is_same_v<TargetT, char> &&
    (sizeof(TargetT) > sizeof(SourceT)),
    TargetT>
numeric_cast(SourceT src) {
  return static_cast<TargetT>(src);
}

}  // namespace ceres

#endif  // CERES_PUBLIC_NUMERIC_CAST_
