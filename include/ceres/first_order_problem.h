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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_PUBLIC_FIRST_ORDER_PROBLEM_H_
#define CERES_PUBLIC_FIRST_ORDER_PROBLEM_H_

#include "glog/logging.h"

// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.
// WARNING: THIS IS EXPERIMENTAL CODE. THE API WILL CHANGE.

namespace ceres {

// A FirstOrder problem is a minimization problem that must be solved
// using just the value and the gradient of the objective function.
class FirstOrderProblem {
 public:
  virtual ~FirstOrderProblem() {}

  // cost is guaranteed never to be null.
  // gradient may or maynot be null.
  //
  // The return value indicates whether the evaluation was successful
  // or not.
  //
  virtual bool Evaluate(const double* parameters,
                        double* cost,
                        double* gradient) const = 0;
  virtual int NumParameters() const = 0;

  virtual int NumTangentSpaceParameters() const {
    return NumParameters();
  }

  virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    CHECK_NOTNULL(x);
    CHECK_NOTNULL(delta);
    CHECK_NOTNULL(x_plus_delta);
    CHECK_EQ(NumParameters(), NumTangentSpaceParameters());

    for (int i = 0; i < NumParameters(); ++i) {
      x_plus_delta[i] = x[i] + delta[i];
    }

    return true;
  }
};

}  // namespace ceres

#endif  // CERES_PUBLIC_FIRST_ORDER_PROBLEM_H_
