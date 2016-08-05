// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

#include "eigen_quaternion_parameterization.h"

#include "Eigen/Geometry"

namespace ceres {
namespace examples {

bool EigenQuaternionParameterization::Plus(const double* q_ptr,
                                           const double* delta,
                                           double* q_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> delta_theta_map(delta);
  Eigen::Map<Eigen::Quaterniond> q_plus_delta_map(q_plus_delta);
  Eigen::Map<const Eigen::Quaterniond> q_map(q_ptr);

  const double norm_delta = delta_theta_map.norm();
  if (norm_delta > 0.0) {
    const double sin_term = std::sin(norm_delta) / norm_delta;

    // Note, in this constructor w is first.
    Eigen::Quaterniond delta_q(std::cos(norm_delta), sin_term * delta[0],
                               sin_term * delta[1], sin_term * delta[2]);
    q_plus_delta_map = delta_q * q_map;
  } else {
    q_plus_delta_map = q_map;
  }

  return true;
}

bool EigenQuaternionParameterization::ComputeJacobian(const double* x,
                                                      double* jacobian) const {
  jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
  jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
  jacobian[6] =  x[1]; jacobian[7]  = -x[0]; jacobian[8]  =  x[3];  // NOLINT
  jacobian[9] = -x[0]; jacobian[10] = -x[1]; jacobian[11] = -x[2];  // NOLINT

  return true;
}

}  // namespace examples
}  // namespace ceres
