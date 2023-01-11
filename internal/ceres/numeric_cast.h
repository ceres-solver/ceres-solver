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

/** Intentionally disabling to/from char/bool conversions.
 * C++ standard do not specify whether char or bool are signed or unsigned. Moreover
 * it is even not guaranteed that any of std::is_same_v<char, signed char> and
 * std::is_same_v<char, unsinged char> is true. Because of this unclear status
 * of char and bool it is better do not use "numeric_cast" for casts from/to these types
 * at all
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
