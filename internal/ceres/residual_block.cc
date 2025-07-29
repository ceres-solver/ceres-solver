// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/residual_block.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "ceres/corrector.h"
#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/parameter_block.h"
#include "ceres/residual_block_utils.h"
#include "ceres/small_blas.h"

using Eigen::Dynamic;

namespace ceres::internal {

ResidualBlock::ResidualBlock(
    const CostFunction* cost_function,
    const LossFunction* loss_function,
    const std::vector<ParameterBlock*>& parameter_blocks,
    int index)
    : cost_function_(cost_function),
      loss_function_(loss_function),
      parameter_blocks_(
          new ParameterBlock*[cost_function->parameter_block_sizes().size()]),
      index_(index) {
  CHECK(cost_function_ != nullptr);
  std::copy(parameter_blocks.begin(),
            parameter_blocks.end(),
            parameter_blocks_.get());
}

bool ResidualBlock::Evaluate(const bool apply_loss_function,
                             double* cost,
                             double* residuals,
                             double** jacobians,
                             double* scratch) const {
  const int num_parameter_blocks = NumParameterBlocks();
  const int num_residuals = cost_function_->num_residuals();

  // Collect the parameters from their blocks. This will rarely allocate, since
  // residuals taking more than 8 parameter block arguments are rare.
  absl::FixedArray<const double*> parameters(num_parameter_blocks);
  for (int i = 0; i < num_parameter_blocks; ++i) {
    parameters[i] = parameter_blocks_[i]->state();
  }

  // Put pointers into the scratch space into global_jacobians as appropriate.
  absl::FixedArray<double*> global_jacobians(num_parameter_blocks);
  if (jacobians != nullptr) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      const ParameterBlock* parameter_block = parameter_blocks_[i];
      if (jacobians[i] != nullptr &&
          parameter_block->PlusJacobian() != nullptr) {
        global_jacobians[i] = scratch;
        scratch += num_residuals * parameter_block->Size();
      } else {
        global_jacobians[i] = jacobians[i];
      }
    }
  }

  // If the caller didn't request residuals, use the scratch space for them.
  bool outputting_residuals = (residuals != nullptr);
  if (!outputting_residuals) {
    residuals = scratch;
  }

  // Invalidate the evaluation buffers so that we can check them after
  // the CostFunction::Evaluate call, to see if all the return values
  // that were required were written to and that they are finite.
  double** eval_jacobians =
      (jacobians != nullptr) ? global_jacobians.data() : nullptr;

  InvalidateEvaluation(*this, cost, residuals, eval_jacobians);

  if (!cost_function_->Evaluate(parameters.data(), residuals, eval_jacobians)) {
    return false;
  }

  if (!IsEvaluationValid(*this, parameters.data(), residuals, eval_jacobians)) {
    LOG(WARNING)
        << "\n\n"
           "Error in evaluating the ResidualBlock.\n\n"
           "There are two possible reasons. Either the CostFunction did not "
           "evaluate and fill all    \n"  // NOLINT
           "residual and jacobians that were requested or there was a "
           "non-finite "
           "value (nan/infinite)\n"  // NOLINT
           "generated during the jacobian computation. \n\n"
        << EvaluationToString(
               *this, parameters.data(), cost, residuals, eval_jacobians);
    return false;
  }

  double squared_norm = VectorRef(residuals, num_residuals).squaredNorm();

  // Update the plus_jacobian for the manifolds.
  if (jacobians != nullptr) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      if (jacobians[i] != nullptr) {
        const ParameterBlock* parameter_block = parameter_blocks_[i];

        // Apply the Manifold::PlusJacobian to the ambient jacobians.
        if (parameter_block->PlusJacobian() != nullptr) {
          // jacobians[i] = global_jacobians[i] * global_to_local_jacobian.
          MatrixMatrixMultiply<Dynamic, Dynamic, Dynamic, Dynamic, 0>(
              global_jacobians[i],
              num_residuals,
              parameter_block->Size(),
              parameter_block->PlusJacobian(),
              parameter_block->Size(),
              parameter_block->TangentSize(),
              jacobians[i],
              0,
              0,
              num_residuals,
              parameter_block->TangentSize());
        }
      }
    }
  }

  if (loss_function_ == nullptr || !apply_loss_function) {
    *cost = 0.5 * squared_norm;
    return true;
  }

  double rho[3];
  loss_function_->Evaluate(squared_norm, rho);
  *cost = 0.5 * rho[0];

  // No jacobians and not outputting residuals? All done. Doing an early exit
  // here avoids constructing the "Corrector" object below in a common case.
  if (jacobians == nullptr && !outputting_residuals) {
    return true;
  }

  // Correct for the effects of the loss function. The jacobians need to be
  // corrected before the residuals, since they use the uncorrected residuals.
  Corrector correct(squared_norm, rho);
  if (jacobians != nullptr) {
    for (int i = 0; i < num_parameter_blocks; ++i) {
      if (jacobians[i] != nullptr) {
        const ParameterBlock* parameter_block = parameter_blocks_[i];

        // Correct the jacobians for the loss function.
        correct.CorrectJacobian(num_residuals,
                                parameter_block->TangentSize(),
                                residuals,
                                jacobians[i]);
      }
    }
  }

  // Correct the residuals with the loss function.
  if (outputting_residuals) {
    correct.CorrectResiduals(num_residuals, residuals);
  }
  return true;
}

int ResidualBlock::NumScratchDoublesForEvaluate() const {
  // Compute the amount of scratch space needed to store the full-sized
  // jacobians. For parameters that have no manifold no storage is needed and
  // the passed-in jacobian array is used directly. Also include space to store
  // the residuals, which is needed for cost-only evaluations.  This is slightly
  // pessimistic, since both won't be needed all the time, but the amount of
  // excess should not cause problems for the caller.
  int num_parameters = NumParameterBlocks();
  int scratch_doubles = 1;
  for (int i = 0; i < num_parameters; ++i) {
    const ParameterBlock* parameter_block = parameter_blocks_[i];
    if (parameter_block->PlusJacobian() != nullptr) {
      scratch_doubles += parameter_block->Size();
    }
  }
  scratch_doubles *= NumResiduals();
  return scratch_doubles;
}

}  // namespace ceres::internal
