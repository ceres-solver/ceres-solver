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

#include "ceres/inner_iteration_minimizer.h"

#include <iterator>
#include <numeric>
#include <vector>
#include "ceres/evaluator.h"
#include "ceres/linear_solver.h"
#include "ceres/minimizer.h"
#include "ceres/ordering.h"
#include "ceres/parameter_block.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/residual_block.h"
#include "ceres/schur_ordering.h"
#include "ceres/solver.h"
#include "ceres/solver_impl.h"
#include "ceres/trust_region_minimizer.h"
#include "ceres/trust_region_strategy.h"

namespace ceres {
namespace internal {

InnerIterationMinimizer::~InnerIterationMinimizer() {
}

bool InnerIterationMinimizer::Init(const Program& outer_program,
                                   const ProblemImpl::ParameterMap& parameter_map,
                                   const vector<double*>& parameter_blocks_for_inner_iterations,
                                   string* error) {
  program_.reset(new Program(outer_program));

  Ordering ordering;
  int num_inner_iteration_parameter_blocks = 0;

  if (parameter_blocks_for_inner_iterations.size() == 0) {
    // The user wishes for the solver to determine a set of parameter
    // blocks to descend on.
    //
    // For now use approximate maximum independent set computed by
    // ComputeSchurOrdering code. Though going forward, we want use
    // the smallest maximal independent set, rather than the largest.
    //
    // TODO(sameeragarwal): Smallest maximal independent set instead
    // of the approximate maximum independent set.
    vector<ParameterBlock*> parameter_block_ordering;
    num_inner_iteration_parameter_blocks =
        ComputeSchurOrdering(*program_, &parameter_block_ordering);
    // Decompose the Schur ordering into elimination group 0 and 1, 0
    // is the one used for inner iterations.
    for (int i = 0; i < parameter_block_ordering.size(); ++i) {
      double* ptr = parameter_block_ordering[i]->mutable_user_state();
      if (i < num_inner_iteration_parameter_blocks) {
        ordering.AddParameterBlockToGroup(ptr, 0);
      } else {
        ordering.AddParameterBlockToGroup(ptr, 1);
      }
    }
  } else {
    const vector<ParameterBlock*> parameter_blocks = program_->parameter_blocks();
    set<double*> parameter_block_ptrs(parameter_blocks_for_inner_iterations.begin(),
                                      parameter_blocks_for_inner_iterations.end());
    num_inner_iteration_parameter_blocks = 0;
    // Divide the set of parameter blocks into two groups. Group 0 is
    // the set of parameter blocks specified by the user, and the rest
    // in group 1.
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      double* ptr = parameter_blocks[i]->mutable_user_state();
      if (parameter_block_ptrs.count(ptr) != 0) {
        ordering.AddParameterBlockToGroup(ptr, 0);
      } else {
        ordering.AddParameterBlockToGroup(ptr, 1);
      }
    }

    num_inner_iteration_parameter_blocks = ordering.GroupSize(0);
    if (num_inner_iteration_parameter_blocks > 0) {
      const map<int, set<double*> >& group_id_to_parameter_blocks =
          ordering.group_id_to_parameter_blocks();
      if (!SolverImpl::IsParameterBlockSetIndependent(
              group_id_to_parameter_blocks.begin()->second,
              program_->residual_blocks())) {
        *error = "The user provided parameter_blocks_for_inner_iterations "
            "does not form an independent set";
        return false;
      }
    }
  }

  if (!SolverImpl::ApplyUserOrdering(parameter_map,
                                     &ordering,
                                     program_.get(),
                                     error)) {
    return false;
  }

  program_->SetParameterOffsetsAndIndex();

  if (!SolverImpl::LexicographicallyOrderResidualBlocks(
          num_inner_iteration_parameter_blocks,
          program_.get(),
          error)) {
    return false;
  }

  ComputeResidualBlockOffsets(num_inner_iteration_parameter_blocks);

  const_cast<Program*>(&outer_program)->SetParameterOffsetsAndIndex();

  LinearSolver::Options linear_solver_options;
  linear_solver_options.type = DENSE_QR;
  linear_solver_.reset(LinearSolver::Create(linear_solver_options));
  CHECK_NOTNULL(linear_solver_.get());

  evaluator_options_.linear_solver_type = DENSE_QR;
  evaluator_options_.num_eliminate_blocks = 0;
  evaluator_options_.num_threads = 1;

  return true;
}

void InnerIterationMinimizer::Minimize(
    const Minimizer::Options& options,
    double* parameters,
    Solver::Summary* summary) {
  const vector<ParameterBlock*>& parameter_blocks = program_->parameter_blocks();
  const vector<ResidualBlock*>& residual_blocks = program_->residual_blocks();

  const int num_inner_iteration_parameter_blocks = residual_block_offsets_.size() - 1;

  for (int i = 0; i < parameter_blocks.size(); ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    parameter_block->SetState(parameters + parameter_block->state_offset());
    if (i >=  num_inner_iteration_parameter_blocks) {
      parameter_block->SetConstant();
    }
  }

#pragma omp parallel for num_threads(options.num_threads)
  for (int i = 0; i < num_inner_iteration_parameter_blocks; ++i) {
    Solver::Summary inner_summary;
    ParameterBlock* parameter_block = parameter_blocks[i];
    const int old_index = parameter_block->index();
    const int old_delta_offset = parameter_block->delta_offset();

    parameter_block->set_index(0);
    parameter_block->set_delta_offset(0);

    Program inner_program;
    inner_program.mutable_parameter_blocks()->push_back(parameter_block);

    // This works, because we have already ordered the residual blocks
    // so that the residual blocks for each parameter block being
    // optimized over are contiguously located in the residual_blocks
    // vector.
    copy(residual_blocks.begin() + residual_block_offsets_[i],
         residual_blocks.begin() + residual_block_offsets_[i + 1],
         back_inserter(*inner_program.mutable_residual_blocks()));

    MinimalSolve(&inner_program,
                 parameters + parameter_block->state_offset(),
                 &inner_summary);

    parameter_block->set_index(old_index);
    parameter_block->set_delta_offset(old_delta_offset);
  }

  for (int i =  num_inner_iteration_parameter_blocks; i < parameter_blocks.size(); ++i) {
    parameter_blocks[i]->SetVarying();
  }
}

void InnerIterationMinimizer::MinimalSolve(Program* program,
                                           double* parameters,
                                           Solver::Summary* summary) {

  *summary = Solver::Summary();
  summary->initial_cost = 0.0;
  summary->fixed_cost = 0.0;
  summary->final_cost = 0.0;
  string error;

  scoped_ptr<Evaluator> evaluator(Evaluator::Create(evaluator_options_, program,  &error));
  CHECK_NOTNULL(evaluator.get());

  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());
  CHECK_NOTNULL(jacobian.get());

  TrustRegionStrategy::Options trust_region_strategy_options;
  trust_region_strategy_options.linear_solver = linear_solver_.get();
  scoped_ptr<TrustRegionStrategy>trust_region_strategy(
      TrustRegionStrategy::Create(trust_region_strategy_options));
  CHECK_NOTNULL(trust_region_strategy.get());

  Minimizer::Options minimizer_options;
  minimizer_options.evaluator = evaluator.get();
  minimizer_options.jacobian = jacobian.get();
  minimizer_options.trust_region_strategy = trust_region_strategy.get();

  TrustRegionMinimizer minimizer;
  minimizer.Minimize(minimizer_options, parameters, summary);
}


void InnerIterationMinimizer::ComputeResidualBlockOffsets(
    const int num_eliminate_blocks) {
  vector<int> counts(num_eliminate_blocks, 0);
  const vector<ResidualBlock*>& residual_blocks = program_->residual_blocks();
  for (int i = 0; i < residual_blocks.size(); ++i) {
    ResidualBlock* residual_block = residual_blocks[i];
    const int num_parameter_blocks = residual_block->NumParameterBlocks();
    for (int j = 0; j < num_parameter_blocks; ++j) {
      ParameterBlock* parameter_block = residual_block->parameter_blocks()[j];
      if (!parameter_block->IsConstant() &&
          parameter_block->index() < num_eliminate_blocks) {
        counts[parameter_block->index()] += 1;
      }
    }
  }

  residual_block_offsets_.resize(num_eliminate_blocks + 1);
  residual_block_offsets_[0] = 0;
  partial_sum(counts.begin(), counts.end(), residual_block_offsets_.begin() + 1);
}


}  // namespace internal
}  // namespace ceres
