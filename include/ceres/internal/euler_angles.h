// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2022 Google Inc. All rights reserved.
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

#ifndef CERES_PUBLIC_INTERNAL_EULER_ANGLES_H_
#define CERES_PUBLIC_INTERNAL_EULER_ANGLES_H_

#include <type_traits>

namespace ceres {
namespace internal {

// The EulerSystem struct represents an Euler Angle Convention in compile time.
// It acts like a trait structure and is also used as a tag for dispatching
// Euler angle conversion function templates
//
// Internally, it implements the convention laid out in "Euler angle
// conversion", Ken Shoemake, Graphics Gems IV, where a choice of axis for the
// first rotation (out of 3) and 3 binary choices compactly specify all 24
// rotation conventions
//
//  - InnerAxis: Axis for the first rotation
//
//  - Parity: Defines if the second rotation is on a successive axis, e.g. XYZ
//  even, ZYZ odd
//
//  - AngleConvention: Toggles using Euler's original definition of his
//  angles, which the last axis repeated, i.e. ZYZ, ZXZ, etc, or the definition
//  introduced by the nautical and aerospace fields, i.e. using ZYX for
//  roll-pitch-yaw
//
//  - IsFrameRotating: Toggles whether the three rotations are be in a global
//  frame of reference (extrinsic) or in a body centred frame of reference

namespace axis {
struct X : std::integral_constant<int, 0> {};
struct Y : std::integral_constant<int, 1> {};
struct Z : std::integral_constant<int, 2> {};
}  // namespace axis

struct Even;
struct Odd;

struct ProperEuler;
struct TaitBryan;

struct Extrinsic;
struct Intrinsic;

template <typename _InnerAxis,
          typename _Parity,
          typename _AngleConvention,
          typename _FrameConvention>
struct EulerSystem {
  using InnerAxis = _InnerAxis;
  using Parity = _Parity;
  using AngleConvention = _AngleConvention;
  using FrameConvention = _FrameConvention;
};

// Define human readable aliases to the type of the tags
using ExtrinsicXYZ_t = EulerSystem<axis::X, Even, TaitBryan, Extrinsic>;
using ExtrinsicXYX_t = EulerSystem<axis::X, Even, ProperEuler, Extrinsic>;
using ExtrinsicXZY_t = EulerSystem<axis::X, Odd, TaitBryan, Extrinsic>;
using ExtrinsicXZX_t = EulerSystem<axis::X, Odd, ProperEuler, Extrinsic>;
using ExtrinsicYZX_t = EulerSystem<axis::Y, Even, TaitBryan, Extrinsic>;
using ExtrinsicYZY_t = EulerSystem<axis::Y, Even, ProperEuler, Extrinsic>;
using ExtrinsicYXZ_t = EulerSystem<axis::Y, Odd, TaitBryan, Extrinsic>;
using ExtrinsicYXY_t = EulerSystem<axis::Y, Odd, ProperEuler, Extrinsic>;
using ExtrinsicZXY_t = EulerSystem<axis::Z, Even, TaitBryan, Extrinsic>;
using ExtrinsicZXZ_t = EulerSystem<axis::Z, Even, ProperEuler, Extrinsic>;
using ExtrinsicZYX_t = EulerSystem<axis::Z, Odd, TaitBryan, Extrinsic>;
using ExtrinsicZYZ_t = EulerSystem<axis::Z, Odd, ProperEuler, Extrinsic>;
/* Rotating axes */
using IntrinsicZYX_t = EulerSystem<axis::X, Even, TaitBryan, Intrinsic>;
using IntrinsicXYX_t = EulerSystem<axis::X, Even, ProperEuler, Intrinsic>;
using IntrinsicYZX_t = EulerSystem<axis::X, Odd, TaitBryan, Intrinsic>;
using IntrinsicXZX_t = EulerSystem<axis::X, Odd, ProperEuler, Intrinsic>;
using IntrinsicXZY_t = EulerSystem<axis::Y, Even, TaitBryan, Intrinsic>;
using IntrinsicYZY_t = EulerSystem<axis::Y, Even, ProperEuler, Intrinsic>;
using IntrinsicZXY_t = EulerSystem<axis::Y, Odd, TaitBryan, Intrinsic>;
using IntrinsicYXY_t = EulerSystem<axis::Y, Odd, ProperEuler, Intrinsic>;
using IntrinsicYXZ_t = EulerSystem<axis::Z, Even, TaitBryan, Intrinsic>;
using IntrinsicZXZ_t = EulerSystem<axis::Z, Even, ProperEuler, Intrinsic>;
using IntrinsicXYZ_t = EulerSystem<axis::Z, Odd, TaitBryan, Intrinsic>;
using IntrinsicZYZ_t = EulerSystem<axis::Z, Odd, ProperEuler, Intrinsic>;

}  // namespace internal

// Defining the tags themselves
inline constexpr internal::ExtrinsicXYZ_t ExtrinsicXYZ{};
inline constexpr internal::ExtrinsicXYX_t ExtrinsicXYX{};
inline constexpr internal::ExtrinsicXZY_t ExtrinsicXZY{};
inline constexpr internal::ExtrinsicXZX_t ExtrinsicXZX{};
inline constexpr internal::ExtrinsicYZX_t ExtrinsicYZX{};
inline constexpr internal::ExtrinsicYZY_t ExtrinsicYZY{};
inline constexpr internal::ExtrinsicYXZ_t ExtrinsicYXZ{};
inline constexpr internal::ExtrinsicYXY_t ExtrinsicYXY{};
inline constexpr internal::ExtrinsicZXY_t ExtrinsicZXY{};
inline constexpr internal::ExtrinsicZXZ_t ExtrinsicZXZ{};
inline constexpr internal::ExtrinsicZYX_t ExtrinsicZYX{};
inline constexpr internal::ExtrinsicZYZ_t ExtrinsicZYZ{};
inline constexpr internal::IntrinsicZYX_t IntrinsicZYX{};
inline constexpr internal::IntrinsicXYX_t IntrinsicXYX{};
inline constexpr internal::IntrinsicYZX_t IntrinsicYZX{};
inline constexpr internal::IntrinsicXZX_t IntrinsicXZX{};
inline constexpr internal::IntrinsicXZY_t IntrinsicXZY{};
inline constexpr internal::IntrinsicYZY_t IntrinsicYZY{};
inline constexpr internal::IntrinsicZXY_t IntrinsicZXY{};
inline constexpr internal::IntrinsicYXY_t IntrinsicYXY{};
inline constexpr internal::IntrinsicYXZ_t IntrinsicYXZ{};
inline constexpr internal::IntrinsicZXZ_t IntrinsicZXZ{};
inline constexpr internal::IntrinsicXYZ_t IntrinsicXYZ{};
inline constexpr internal::IntrinsicZYZ_t IntrinsicZYZ{};

template <typename EulerSystem>
struct InnerAxis : std::integral_constant<int, EulerSystem::InnerAxis::value> {
};

template <typename EulerSystem>
inline constexpr int InnerAxis_v = InnerAxis<EulerSystem>::value;

template <typename EulerSystem>
struct IsParityOdd
    : std::integral_constant<
          bool,
          std::is_same_v<typename EulerSystem::Parity, internal::Odd>> {};

template <typename EulerSystem>
inline constexpr int IsParityOdd_v = IsParityOdd<EulerSystem>::value;

template <typename EulerSystem>
struct IsProperEuler : std::integral_constant<
                           bool,
                           std::is_same_v<typename EulerSystem::AngleConvention,
                                          internal::ProperEuler>> {};

template <typename EulerSystem>
inline constexpr int IsProperEuler_v = IsProperEuler<EulerSystem>::value;

template <typename EulerSystem>
struct IsIntrinsic : std::integral_constant<
                         bool,
                         std::is_same_v<typename EulerSystem::FrameConvention,
                                        internal::Intrinsic>> {};

template <typename EulerSystem>
inline constexpr int IsIntrinsic_v = IsIntrinsic<EulerSystem>::value;

}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_EULER_ANGLES_H_
