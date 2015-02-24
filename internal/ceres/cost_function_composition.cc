// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: richie.stebbing@gmail.com (Richard Stebbing)

#include "ceres/cost_function_composition.h"
#include "glog/logging.h"

namespace ceres {

CostFunctionComposition::CostFunctionComposition(
  CostFunction* f,
  Ownership ownership)
    : f_(CHECK_NOTNULL(f)),
      owns_f_(ownership),
      is_finalised_(false) {
  set_num_residuals(f->num_residuals());

  // Postpone setting `parameter_block_sizes` until `Finalize`.
}

CostFunctionComposition::~CostFunctionComposition() {
  if (owns_f_ == DO_NOT_TAKE_OWNERSHIP) {
    f_.release();
  }
}

void CostFunctionComposition::AddInputCostFunction(
    CostFunction* g,
    const std::vector<double*>& parameter_blocks,
    Ownership ownership) {
  // TODO `CostFunctionComposition::AddInputCostFunction`.
}

void CostFunctionComposition::AddInputParameterBlock(
    double* values) {
  // TODO `CostFunctionComposition::AddInputParameterBlock`.
  // TODO Infer size of parameter block from `f_`.
}

void CostFunctionComposition::Finalize() {
  CHECK(!is_finalised_);

  // TODO Ensure that all inputs to `f_` have been added.

  // TODO Ensure that the number of residuals for each input function is the
  // same as the dimension of the input parameter blocks for `f_`.

  // TODO Set `parameter_block_sizes`.

  is_finalised_ = true;
}

bool CostFunctionComposition::Evaluate(const double* const* parameters,
                                       double* residuals,
                                       double** jacobians) const {
  CHECK(is_finalised_);

  // TODO `CostFunctionComposition::Evaluate`.

  return true;
}

}  // namespace ceres
