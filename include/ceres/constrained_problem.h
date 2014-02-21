// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2014 Google Inc. All rights reserved.
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

#ifndef CERES_PUBLIC_CONSTRAINED_PROBLEM_H_
#define CERES_PUBLIC_CONSTRAINED_PROBLEM_H_

#include <cstddef>
#include <map>

#include "ceres/internal/macros.h"
#include "ceres/internal/port.h"
#include "ceres/problem.h"
#include "ceres/types.h"

namespace ceres {

class CostFunction;
class LossFunction;
class LocalParameterization;

namespace internal {
class ResidualBlock;
}  // namespace internal

typedef internal::ResidualBlock* ResidualBlockId;
typedef internal::ResidualBlock* ConstraintBlockId;

namespace experimental {

enum ConstraintType {
  LINEAR,
  NONLINEAR
};

// Experimental code.
class ConstrainedProblem {
 public:
  ~ConstrainedProblem();

  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   const vector<double*>& parameter_blocks);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4, double* x5);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4, double* x5,
                                   double* x6);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4, double* x5,
                                   double* x6, double* x7);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4, double* x5,
                                   double* x6, double* x7, double* x8);
  ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                   LossFunction* loss_function,
                                   double* x0, double* x1, double* x2,
                                   double* x3, double* x4, double* x5,
                                   double* x6, double* x7, double* x8,
                                   double* x9);

  // WARNING WARNING WARNING
  // DO NOT REUSE CostFunction objects across constraints.
  // WARNING WARNING WARNING
  //
  // Add an equality constraint cost_function(x) = 0.
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       const vector<double*>& parameter_blocks);

  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4, double* x5);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4, double* x5,
                                       double* x6);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4, double* x5,
                                       double* x6, double* x7);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4, double* x5,
                                       double* x6, double* x7, double* x8);
  ConstraintBlockId AddConstraintBlock(CostFunction* cost_function,
                                       ConstraintType constraint_type,
                                       double* x0, double* x1, double* x2,
                                       double* x3, double* x4, double* x5,
                                       double* x6, double* x7, double* x8,
                                       double* x9);


  // Add a parameter block with appropriate size to the problem.
  // Repeated calls with the same arguments are ignored. Repeated
  // calls with the same double pointer but a different size results
  // in undefined behaviour.
  void AddParameterBlock(double* values, int size);

  // Add a parameter block with appropriate size and parameterization
  // to the problem. Repeated calls with the same arguments are
  // ignored. Repeated calls with the same double pointer but a
  // different size results in undefined behaviour.
  void AddParameterBlock(double* values,
                         int size,
                         LocalParameterization* local_parameterization);

  // Hold the indicated parameter block constant during optimization.
  void SetParameterBlockConstant(double* values);

  // Allow the indicated parameter block to vary during optimization.
  void SetParameterBlockVariable(double* values);

  // Set the local parameterization for one of the parameter blocks.
  // The local_parameterization is owned by the Problem by default. It
  // is acceptable to set the same parameterization for multiple
  // parameters; the destructor is careful to delete local
  // parameterizations only once. The local parameterization can only
  // be set once per parameter, and cannot be changed once set.
  void SetParameterization(double* values,
                           LocalParameterization* local_parameterization);

  // Get the local parameterization object associated with this
  // parameter block. If there is no parameterization object
  // associated then NULL is returned.
  const LocalParameterization* GetParameterization(double* values) const;

  // Set the lower/upper bound for the parameter with position "index".
  void SetParameterLowerBound(double* values, int index, double lower_bound);
  void SetParameterUpperBound(double* values, int index, double upper_bound);

  // Number of parameter blocks in the problem. Always equals
  // parameter_blocks().size() and parameter_block_sizes().size().
  int NumParameterBlocks() const;

  // The size of the parameter vector obtained by summing over the
  // sizes of all the parameter blocks.
  int NumParameters() const;

  // Number of residual blocks in the problem. Always equals
  // residual_blocks().size().
  int NumResidualBlocks() const;

  // The size of the residual vector obtained by summing over the
  // sizes of all of the residual blocks.
  int NumResiduals() const;

  // The size of the parameter block.
  int ParameterBlockSize(const double* values) const;

  // The size of local parameterization for the parameter block. If
  // there is no local parameterization associated with this parameter
  // block, then ParameterBlockLocalSize = ParameterBlockSize.
  int ParameterBlockLocalSize(const double* values) const;

  Problem* mutable_problem() { return &problem_; };
  const map<ConstraintBlockId, CostFunction*>& constraints() const { return constraints_; }
  map<ConstraintBlockId, CostFunction*>* mutable_constraints() { return &constraints_; }

 private:
  Problem problem_;
  map<ConstraintBlockId, CostFunction*> constraints_;
  CERES_DISALLOW_COPY_AND_ASSIGN(ConstrainedProblem);
};

}  // namespace experimental
}  // namespace ceres

#endif  // CERES_PUBLIC_CONSTRAINED_PROBLEM_H_
