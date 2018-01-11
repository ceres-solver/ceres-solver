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
// Author: mierle@gmail.com (Keir Mierle)
//
// This callback interface allows Ceres to notify you when it is about to
// evaluate the residuals or jacobians with a new set of parameters. The use
// case for this is to allow sharing some computation between residual blocks,
// by doing that computation before Ceres calls CostFunction::Evaluate() on all
// the residuals. It also enables caching results between a pure residual
// evaluation and a residual & jacobian evaluation, via the
// new_evaluation_point argument.
//
// One example use is where the cost functions provided to Ceres depend on
// positions which are computed via IMU integration, where points at the end of
// a time window depend on all previous points in time. Naive cost function
// evaluation would have O(n^2) complexity in the number of points along the
// integration path if there was a Ceres cost function for each point. By
// pre-computing part of the evalutation inside a PrepareForResidual() or
// similar callback, that data can get re-used by the cost functions when Ceres
// calls them as part of normal evaluation. In this example, compute would
// become O(n) instead of O(n^2).
//
// Note that Ceres provides no mechanism to share data other than the callback.
// Users must either use global data or pass in shared data to their cost
// functions behind the scenes; this all happens without Ceres knowing. As far
// as Ceres is concerned, it is just evaluating cost functions like any other;
// it just so happens that behind the scenes the cost functions reuse
// pre-computed data to execute faster.

#ifndef CERES_PUBLIC_EVALUATION_CALLBACK_H_
#define CERES_PUBLIC_EVALUATION_CALLBACK_H_

namespace ceres {

// Mechanism to get notified by Ceres when it is about to evaluate the cost
// functions with new parameters.
class CERES_EXPORT EvaluationCallback {
 public:
  virtual ~EvaluationCallback() {}

  // Called before Ceres requests residuals or jacobians for a given setting of
  // the parameters. User parameters (the double* values provided to the cost
  // functions) are fixed until the next call to PrepareForEvaluation() If
  // new_evaluation_point == true, then this is a new point that is different
  // from the last evaluated point. Otherwise, it is the same point that was
  // evaluated previously (either jacobian or residual) and the user can use
  // cached results from previous evaluations.
  virtual void PrepareForEvaluation(bool jacobians,
                                    bool new_evaluation_point) = 0;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_EVALUATION_CALLBACK_H_
