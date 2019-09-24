// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)
//
// This file is included both, by the generator and the final program. During
// generation the macro CODE_GENERATION is set.

#include "ceres/jet.h"
#include "ceres/rotation.h"

namespace ceres {

// same as AngleAxisRotatePoint, but with PHI funtions instead of branches
template <typename T>
inline void AngleAxisRotatePointPHI(const T angle_axis[3],
                                    const T pt[3],
                                    T result[3]) {
  DCHECK_NE(pt, result) << "Inplace rotation is not supported.";

  const T theta2 = DotProduct(angle_axis, angle_axis);
  auto condition = theta2 > T(std::numeric_limits<double>::epsilon());

  T phiTrue[3];
  T phiFalse[3];
  {
    // Away from zero, use the rodriguez formula
    //
    //   result = pt costheta +
    //            (w x pt) * sintheta +
    //            w (w . pt) (1 - costheta)
    //
    // We want to be careful to only evaluate the square root if the
    // norm of the angle_axis vector is greater than zero. Otherwise
    // we get a division by zero.
    //
    const T theta = sqrt(theta2);
    const T costheta = cos(theta);
    const T sintheta = sin(theta);
    const T theta_inverse = T(1.0) / theta;

    const T w[3] = {angle_axis[0] * theta_inverse,
                    angle_axis[1] * theta_inverse,
                    angle_axis[2] * theta_inverse};

    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    const T w_cross_pt[3] = {w[1] * pt[2] - w[2] * pt[1],
                             w[2] * pt[0] - w[0] * pt[2],
                             w[0] * pt[1] - w[1] * pt[0]};
    const T tmp =
        (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

    phiTrue[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
    phiTrue[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
    phiTrue[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
  }
  {
    // Near zero, the first order Taylor approximation of the rotation
    // matrix R corresponding to a vector w and angle w is
    //
    //   R = I + hat(w) * sin(theta)
    //
    // But sintheta ~ theta and theta * w = angle_axis, which gives us
    //
    //  R = I + hat(w)
    //
    // and actually performing multiplication with the point pt, gives us
    // R * pt = pt + w x pt.
    //
    // Switching to the Taylor expansion near zero provides meaningful
    // derivatives when evaluated using Jets.
    //
    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    const T w_cross_pt[3] = {angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                             angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                             angle_axis[0] * pt[1] - angle_axis[1] * pt[0]};

    phiFalse[0] = pt[0] + w_cross_pt[0];
    phiFalse[1] = pt[1] + w_cross_pt[1];
    phiFalse[2] = pt[2] + w_cross_pt[2];
  }
  result[0] = PHI(condition, phiTrue[0], phiFalse[0]);
  result[1] = PHI(condition, phiTrue[1], phiFalse[1]);
  result[2] = PHI(condition, phiTrue[2], phiFalse[2]);
}

struct SnavelyReprojectionErrorGen {
  SnavelyReprojectionErrorGen(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
#ifdef CODE_GENERATION
    T ox = CERES_EXTERNAL_CONSTANT(observed_x);
    T oy = CERES_EXTERNAL_CONSTANT(observed_y);
#else
    T ox(observed_x);
    T oy(observed_y);
#endif

    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
#ifdef CODE_GENERATION
    AngleAxisRotatePointPHI(camera, point, p);
#else
    AngleAxisRotatePoint(camera, point, p);
#endif

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    const T xp = -p[0] / p[2];
    const T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (l1 + l2 * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    const T predicted_x = focal * distortion * xp;
    const T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - ox;
    residuals[1] = predicted_y - oy;

    return true;
  }

// During generation this file does not exist yet.
#ifndef CODE_GENERATION
#include "autodiff_codegen_benchmark_gen.h"
#endif

  double observed_x;
  double observed_y;
};

}  // namespace ceres
