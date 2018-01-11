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
// This callback allows Ceres to notify you when it is about to evaluate the
// residuals with a new set of parameters. The use case for this is to allow
// sharing some computation between residual blocks, by doing that computation
// before Ceres calls CostFunction::Evaluate() on all the residuals.

#ifndef CERES_PUBLIC_EVALUATION_CALLBACK_H_
#define CERES_PUBLIC_EVALUATION_CALLBACK_H_

namespace ceres {

// Mechanism to get notified by Ceres when it is about to evaluate the cost
// functions with new parameters. The calls will come in pairs, with
// EvaluationComplete() always run last. For example:
//
//   evaluation_callback->PrepareForResidualAndJacobianEvaluation()
//   evaluation_callback->EvaluationComplete()
//   ...
//   evaluation_callback->PrepareForResidualOnlyEvaluation()
//   evaluation_callback->EvaluationComplete()
//   ...
//   evaluation_callback->PrepareForResidualAndJacobianEvaluation()
//   evaluation_callback->EvaluationComplete()
//
// and so on. The following calls should never happen (guaranteed by Ceres):
//
//   evaluation_callback->PrepareForResidualAndJacobianEvaluation()
//   evaluation_callback->PrepareForResidualAndJacobianEvaluation()
//   evaluation_callback->EvaluationComplete()
//
//   evaluation_callback->EvaluationComplete()
//   evaluation_callback->EvaluationComplete()
//
//   evaluation_callback->EvaluationComplete()
//   evaluation_callback->PrepareForResidualAndJacobianEvaluation()
//
class CERES_EXPORT EvaluationCallback {
 public:
  virtual ~EvaluationCallback() {}

  // Called by Ceres before evaluating a Problem's residuals with new parameter
  // values. Before this call, Ceres pushes the values to evaluate back into
  // the user parameter pointers. Ceres guarantees that the values pushed into
  // the parameter pointers will not change until EvaluationComplete() is
  // called. See Solver::Options::update_state_every_iteration for more
  // information about the mechanism to access the parameter values.
  //
  // Note: This does NOT guarantee the order of cost function evaluation or
  // which thread the evaluation will happen on. Ceres may call the cost
  // functions in any order, and with any subset of parameters marked as
  // requesting the jacobians.
  virtual void PrepareForResidualAndJacobianEvaluation() = 0;

  // Same as PrepareForResidualAndJacobianEvaluation(), but Ceres guarantees
  // that it will not request any jacobians in the subsequent evaluation.
  virtual void PrepareForResidualOnlyEvaluation() = 0;

  // Called after Ceres is finished evaluating the residuals.
  virtual void EvaluationComplete() = 0;
};

// XXX This should be private to Ceres but is in this header for now:
class ScopedEvaluationCallbackContext {
  void ~ScopedEvaluationCallbackContext() {
    if (callback_ != NULL) {
      callback_->EvaluationComplete();
    }
  }

  void set(EvaluationCallback* callback) {
    callback_ = callback;
  }
};

}  // namespace ceres

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_PUBLIC_EVALUATION_CALLBACK_H_
