// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
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
// Author: keir@google.com (Keir Mierle)

#ifndef CERES_INTERNAL_PROGRAM_H_
#define CERES_INTERNAL_PROGRAM_H_

#include <vector>
#include "ceres/internal/port.h"

namespace ceres {

class CRSMatrix;

namespace internal {

class ParameterBlock;
class ProblemImpl;
class ResidualBlock;

// A nonlinear least squares optimization problem. This is different from the
// similarly-named "Problem" object, which offers a mutation interface for
// adding and modifying parameters and residuals. The Program contains the core
// part of the Problem, which is the parameters and the residuals, stored in a
// particular ordering. The ordering is critical, since it defines the mapping
// between (residual, parameter) pairs and a position in the jacobian of the
// objective function. Various parts of Ceres transform one Program into
// another; for example, the first stage of solving involves stripping all
// constant parameters and residuals. This is in contrast with Problem, which is
// not built for transformation.
class Program {
 public:
  Program();
  explicit Program(const Program& program);

  // The ordered parameter and residual blocks for the program.
  const vector<ParameterBlock*>& parameter_blocks() const;
  const vector<ResidualBlock*>& residual_blocks() const;
  vector<ParameterBlock*>* mutable_parameter_blocks();
  vector<ResidualBlock*>* mutable_residual_blocks();

  // Serialize to/from the program and update states.
  //
  // NOTE: Setting the state of a parameter block can trigger the
  // computation of the Jacobian of its local parameterization. If
  // this computation fails for some reason, then this method returns
  // false and the state of the parameter blocks cannot be trusted.
  bool StateVectorToParameterBlocks(const double *state);
  void ParameterBlocksToStateVector(double *state) const;

  // Copy internal state to the user's parameters.
  void CopyParameterBlockStateToUserState();

  // Set the parameter block pointers to the user pointers. Since this
  // runs parameter block set state internally, which may call local
  // parameterizations, this can fail. False is returned on failure.
  bool SetParameterBlockStatePtrsToUserStatePtrs();

  // Update a state vector for the program given a delta.
  bool Plus(const double* state,
            const double* delta,
            double* state_plus_delta) const;

  // Set the parameter indices and offsets. This permits mapping backward
  // from a ParameterBlock* to an index in the parameter_blocks() vector. For
  // any parameter block p, after calling SetParameterOffsetsAndIndex(), it
  // is true that
  //
  //   parameter_blocks()[p->index()] == p
  //
  // If a parameter appears in a residual but not in the parameter block, then
  // it will have an index of -1.
  //
  // This also updates p->state_offset() and p->delta_offset(), which are the
  // position of the parameter in the state and delta vector respectively.
  void SetParameterOffsetsAndIndex();

  // See problem.h for what these do.
  int NumParameterBlocks() const;
  int NumParameters() const;
  int NumEffectiveParameters() const;
  int NumResidualBlocks() const;
  int NumResiduals() const;

  int MaxScratchDoublesNeededForEvaluate() const;
  int MaxDerivativesPerResidualBlock() const;
  int MaxParametersPerResidualBlock() const;

  // Program evaluator. This is used for computing the cost, residual
  // and Jacobian for returning to the user. For actually solving the
  // optimization problem, the optimization algorithm uses the
  // ProgramEvaluator objects directly.
  //
  // The residual and jacobian pointers can be NULL, in which case
  // they will not be evaluated. cost cannot be NULL.
  //
  // The parallelism of the evaluator is controlled by num_threads, it
  // should be at least 1.
  //
  // Note: This function assumes that the state of the parameter
  // blocks is already set to the value that is of interest to the
  // user. In practice what this means is that
  // Program::SetParameterBlockStatePtrsToUserStatePtrs() has been
  // called before calling Evaluate, so that the state pointers of all
  // the parameter blocks point to the user state pointer.
  //
  // Also worth noting is that this function mutates program by
  // calling, Program::SetParameterOffsetsAndIndex() on it, so that an
  // evaluator object can be constructed. Further, Evaluate calls
  // program->SetParameterBlockStatePtrsToUserStatePtrs() on the way
  // out. So that the state of the ParameterBlocks does not point to
  // some vector that doesn exist in memory anymore.
  static bool Evaluate(Program* program,
                       int num_threads,
                       double* cost,
                       vector<double>* residuals,
                       CRSMatrix* jacobian);

 private:
  // The Program does not own the ParameterBlock or ResidualBlock objects.
  vector<ParameterBlock*> parameter_blocks_;
  vector<ResidualBlock*> residual_blocks_;

  friend class ProblemImpl;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_PROGRAM_H_
