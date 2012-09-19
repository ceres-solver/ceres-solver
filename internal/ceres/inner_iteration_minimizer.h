// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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

#include <vector>
#include "ceres/internal/scoped_ptr.h"
#include "ceres/minimizer.h"
#include "ceres/problem_impl.h"
#include "ceres/solver.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/evaluator.h"
#include "ceres/linear_solver.h"

namespace ceres {
namespace internal {

class Program;

// The InnerIterationMinimizer performs coordinate descent on a user
// specified set of parameter blocks. The user can either specify the
// set of parameter blocks for coordinate descent or have the
// minimizer choose on its own.
//
// This Minimizer when used in combination with the
// TrustRegionMinimizer is used to implement a non-linear
// generalization of Ruhe & Wedin's Algorithm II for separable
// non-linear least squares problems.
class InnerIterationMinimizer : public Minimizer {
 public:
  // Initialize the minimizer. The return value indicates success or
  // failure, and the error string contains a description of the
  // error.
  //
  // The parameter blocks for inner iterations must form an
  // independent set in the Hessian for the optimization problem.
  //
  // If this vector is empty, the minimizer will attempt to find a set
  // of parameter blocks to optimize.
  bool Init(const Program& program,
            const ProblemImpl::ParameterMap& parameter_map,
            const vector<double*>& parameter_blocks_for_inner_iterations,
            string* error);

  // Minimizer interface.
  virtual ~InnerIterationMinimizer();
  virtual void Minimize(const Minimizer::Options& options,
                        double* parameters,
                        Solver::Summary* summary);


 private:
  void MinimalSolve(Program* program, double* parameters, Solver::Summary* summary);
  void ComputeResidualBlockOffsetsParameterBlock(const int num_eliminate_blocks);

  scoped_ptr<Program> program_;
  vector<int> residual_block_offsets_;
  Evaluator::Options evaluator_options_;
  TrustRegionStrategy::Options trust_region_strategy_options_;
  scoped_ptr<LinearSolver> linear_solver_;
  scoped_ptr<TrustRegionStrategy> trust_region_strategy_;
};

}  // namespace internal
}  // namespace ceres
