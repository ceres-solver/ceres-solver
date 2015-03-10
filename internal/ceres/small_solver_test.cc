// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)

#include "ceres/small_solver.h"

#include <algorithm>
#include <cmath>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "ceres/fpclassify.h"
#include "ceres/stringprintf.h"
#include "ceres/test_util.h"

namespace ceres {
namespace internal {

typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;

class F {
 public:
  typedef double Scalar;
  enum {
    // Can also be Eigen::Dynamic
    NUM_PARAMETERS = 3,
    NUM_RESIDUALS = 4,
  };
  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    double x = parameters[0];
    double y = parameters[1];
    double z = parameters[2];

    residuals[0] = x + 2*y + 4*z;
    residuals[1] = y * z;

    if (jacobian) {
      jacobian[0 * 3 + 0] = 1;
      jacobian[0 * 3 + 1] = 2;
      jacobian[0 * 3 + 2] = 4;
      jacobian[1 * 3 + 0] = 0;
      jacobian[1 * 3 + 1] = 1;
      jacobian[1 * 3 + 2] = 1;
    }
    return true;
  }
};

TEST(SmallSolver, Works) {
  Vec3 x(0.76026643, -30.01799744, 0.55192142);
  F f;
  SmallSolver<F> solver;
  solver.solve(f, &x);
  Vec3 expected_min_x(2, 5, 0);
  ExpectClose(expected_min_x(0), x(0), 1e-5);
  ExpectClose(expected_min_x(1), x(1), 1e-5);
  ExpectClose(expected_min_x(2), x(2), 1e-5);
}

}  // namespace internal
}  // namespace ceres
