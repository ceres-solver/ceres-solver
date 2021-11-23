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
//  - IsParityOdd: Toggles 'parity' of the Euler System, which  defines if the
//  second rotation is on a successive axis, e.g. XYZ even, ZYZ odd
//
//  - IsLastAxisRepeated: Toggles using Euler's original definition of his
//  angles, which the last axis repeated, i.e. ZYZ, ZXZ, etc, or the definition
//  introduced by the nautical and aerospace fields, i.e. using ZYX for
//  roll-pitch-yaw
//
//  - IsFrameRotating: Toggles whether the three rotations are be in a global
//  frame of reference (extrinsic) or in a body centred frame of reference

template <int InnerAxis,
          bool IsParityOdd,
          bool IsLastAxisRepeated,
          bool IsFrameRotating>
struct EulerSystem {
  static_assert(
      InnerAxis >= 0 && InnerAxis < 3,
      "Inner axis must be the 1st, 2nd, or 3rd standard basis vector");
  static constexpr int kInnerAxis = InnerAxis;
  static constexpr bool kIsParityOdd = IsParityOdd;
  static constexpr bool kIsLastAxisRepeated = IsLastAxisRepeated;
  static constexpr bool kIsFrameRotating = IsFrameRotating;
};

// Define human readable aliases to the type of the tags
using ExtrinsicXYZ_t = EulerSystem<0, false, false, false>;
using ExtrinsicXYX_t = EulerSystem<0, false, true, false>;
using ExtrinsicXZY_t = EulerSystem<0, true, false, false>;
using ExtrinsicXZX_t = EulerSystem<0, true, true, false>;
using ExtrinsicYZX_t = EulerSystem<1, false, false, false>;
using ExtrinsicYZY_t = EulerSystem<1, false, true, false>;
using ExtrinsicYXZ_t = EulerSystem<1, true, false, false>;
using ExtrinsicYXY_t = EulerSystem<1, true, true, false>;
using ExtrinsicZXY_t = EulerSystem<2, false, false, false>;
using ExtrinsicZXZ_t = EulerSystem<2, false, true, false>;
using ExtrinsicZYX_t = EulerSystem<2, true, false, false>;
using ExtrinsicZYZ_t = EulerSystem<2, true, true, false>;
/* Rotating axes */
using IntrinsicZYX_t = EulerSystem<0, false, false, true>;
using IntrinsicXYX_t = EulerSystem<0, false, true, true>;
using IntrinsicYZX_t = EulerSystem<0, true, false, true>;
using IntrinsicXZX_t = EulerSystem<0, true, true, true>;
using IntrinsicXZY_t = EulerSystem<1, false, false, true>;
using IntrinsicYZY_t = EulerSystem<1, false, true, true>;
using IntrinsicZXY_t = EulerSystem<1, true, false, true>;
using IntrinsicYXY_t = EulerSystem<1, true, true, true>;
using IntrinsicYXZ_t = EulerSystem<2, false, false, true>;
using IntrinsicZXZ_t = EulerSystem<2, false, true, true>;
using IntrinsicXYZ_t = EulerSystem<2, true, false, true>;
using IntrinsicZYZ_t = EulerSystem<2, true, true, true>;

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

}  // namespace ceres

#endif  // CERES_PUBLIC_INTERNAL_EULER_ANGLES_H_
