// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
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
// Author: sergey.vfx@gmail.com (Sergey Sharybin)
//         mierle@gmail.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_SO3_H_
#define CERES_PUBLIC_SO3_H_

#include "ceres/autodiff_local_parameterization.h"
#include "ceres/rotation.h"

namespace ceres {

// Plus functor which gets initial rotation R,
// rotation delta and computes
//
//   R_plus_delta = R * AngleAxisToRotationMatrix(delta)
struct RotationMatrixPlus {
  template<typename T>
  bool operator()(const T* x,  // Rotation 3x3 col-major.
                  const T* delta,  // Angle-axis delta
                  T* x_plus_delta) const {
    T angle_axis[3];

    RotationMatrixToAngleAxis(x, angle_axis);

    angle_axis[0] += delta[0];
    angle_axis[1] += delta[1];
    angle_axis[2] += delta[2];

    AngleAxisToRotationMatrix(angle_axis, x_plus_delta);
    return true;
  }
};

// Local parameterization from 3x3 col-major rotation matrix
// space to angle-axis rotation space using auto differentiation
// to compute the Jacobian.
//
// This is useful when you need to optimize a rotation matrix
// in-place, and want to avoid re-parameterizing the problem with
// angle axis or another representation. To use a rotation matrix
// as a parameter block, set the parameterization:
//
//   problem.SetParameterization(rotation_matrix_block
//                               new AutoDiffRotationMatrixParameterization);
//
// Note: the same parameterization object can be shared across
// different parameter blocks.
typedef AutoDiffLocalParameterization<RotationMatrixPlus, 9, 3>
      RotationMatrixLocalParameterization;

}  // namespace ceres

#endif  // CERES_PUBLIC_SO3_LOCAL_PARAMETERIZATION_H_
