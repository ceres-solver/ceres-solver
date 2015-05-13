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
// Author: tbennun@gmail.com (Tal Ben-Nun)
//
// This header file contains a data structure with options for numeric
// differentiation.

#ifndef CERES_PUBLIC_NUMERIC_DIFF_OPTIONS_H_
#define CERES_PUBLIC_NUMERIC_DIFF_OPTIONS_H_

namespace ceres {

// This data structure contains various options pertaining to numeric
// differentiation, such as convergence criteria and step sizes.
struct CERES_EXPORT NumericDiffOptions {
  // A constructor that issues default numeric differentiation parameters.
  NumericDiffOptions() {
    relative_step_size = 1e-6;
    adaptive_initial_step_size = 1e-2;
    max_adaptive_extrapolations = 10;
    adaptive_epsilon = 1e-12;
    adaptive_step_shrink_factor = 2.0;
    adaptive_relative_error = false;
  }

  // Numeric differentiation step size (multiplied by parameter block's
  // order of magnitude). If parameters are close to zero, the step size
  // is set to sqrt(machine_epsilon).
  double relative_step_size;

  // Initial step size for adaptive numeric differentiation (multiplied
  // by parameter block's order of magnitude).
  // If parameters are close to zero, Ridders' method sets the step size
  // directly to this value.
  //
  // Note for Ridders' method: In order for the algorithm to converge,
  // the step size should be initialized to a value that is large enough
  // to produce a significant change in the function. As the derivative
  // is estimated, the step size decreases.
  double adaptive_initial_step_size;

  // Maximal number of adaptive extrapolations (sampling) in Ridders'
  // method.
  int max_adaptive_extrapolations;

  // Convergence criterion on extrapolation error for Ridders adaptive
  // differentiation.
  double adaptive_epsilon;

  // The factor in which to shrink the step size with each extrapolation
  // in Ridders' method.
  double adaptive_step_shrink_factor;

  // During extrapolation error estimation in Ridders' method, use
  // relative error (norm(1 - a/b)) instead of norm(a - b).
  bool adaptive_relative_error;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_NUMERIC_DIFF_OPTIONS_H_
