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

#ifndef EXAMPLES_CERES_EIGEN_QUATERNION_PARAMETERIZATION_H_
#define EXAMPLES_CERES_EIGEN_QUATERNION_PARAMETERIZATION_H_

#include "ceres/local_parameterization.h"

namespace ceres {
namespace examples {

// Implements the quaternion local parameterization for Eigen's implementation
// of the quaternion. The quaternion is Hamiltonian based with the scalar as the
// last element. The difference with Ceres's quaternion represenation is where
// the scalar part is stored.
//
// Plus(x, delta) = [sin(|delta|) delta / |delta|, cos(|delta|)] * x
//
// with * being the quaternion multiplication operator. Here the last element of
// the quaternion vector is the real (cos theta) part.
class EigenQuaternionParameterization : public ceres::LocalParameterization {
 public:

  virtual ~EigenQuaternionParameterization() {}

  virtual bool Plus(const double* q, const double* delta,
                    double* q_plus_delta) const;

  virtual bool ComputeJacobian(const double* x, double* jacobian) const;

  virtual int GlobalSize() const { return 4; }
  virtual int LocalSize() const { return 3; }
};

}  // namespace examples
}  // namespace ceres


#endif  // EXAMPLES_CERES_EIGEN_QUATERNION_PARAMETERIZATION_H_
