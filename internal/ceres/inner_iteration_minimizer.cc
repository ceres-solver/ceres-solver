#include "ceres/inner_iteration_minimizer.h"

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

bool InnerIterationMinimizer::Init(const Program& program,
                                   const ProblemImpl::ParameterMap& parameter_map,
                                   const vector<double*>& parameter_blocks_for_inner_iterations,
                                   string* error) {
  program_.reset(new Program(program));

  Ordering ordering;
  int num_inner_iteration_parameter_blocks = 0;

  if (parameter_blocks_for_inner_iterations.size() == 0) {
    vector<ParameterBlock*> parameter_block_ordering;
    const int num_eliminate_blocks =
        ComputeSchurOrdering(*program_, &parameter_block_ordering);
    for (int i = 0; i < parameter_block_ordering.size(); ++i) {
      ordering.AddParameterBlockToGroup(
          parameter_block_ordering[i]->mutable_user_state(),
          i >= num_inner_iteration_parameter_blocks);
    }

    num_inner_iteration_parameter_blocks =
        parameter_block_ordering.size() - num_eliminate_blocks;
  } else {
    const vector<ParameterBlock*> parameter_blocks = program_->parameter_blocks();
    set<double*> parameter_block_ptrs(parameter_blocks_for_inner_iterations.begin(),
                                      parameter_blocks_for_inner_iterations.end());
    num_inner_iteration_parameter_blocks = 0;
    for (int i = 0; i < parameter_blocks.size(); ++i) {
      if (parameter_block_ptrs.count(parameter_blocks[i]->mutable_user_state()) > 0) {
        ++num_inner_iteration_parameter_blocks;
        ordering.AddParameterBlockToGroup(parameter_blocks[i]->mutable_user_state(), 0);
      } else {
        ordering.AddParameterBlockToGroup(parameter_blocks[i]->mutable_user_state(), 1);
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

  if (!SolverImpl::LexicographicallyOrderResidualBlocks(num_inner_iteration_parameter_blocks,
                                            program_.get(),
                                            error)) {
    return false;
  }

  CountResidualBlocksPerParameterBlock(num_inner_iteration_parameter_blocks);

  const_cast<Program*>(&program)->SetParameterOffsetsAndIndex();
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

  Solver::Summary inner_summary;
  for (int i = 0; i < num_inner_iteration_parameter_blocks; ++i) {
    ParameterBlock* parameter_block = parameter_blocks[i];
    const int old_index = parameter_block->index();
    const int old_delta_offset = parameter_block->delta_offset();

    parameter_block->set_index(0);
    parameter_block->set_delta_offset(0);

    Program inner_program;
    inner_program.mutable_parameter_blocks()->push_back(parameter_block);

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

  LinearSolver::Options linear_solver_options;
  linear_solver_options.type = DENSE_QR;
  scoped_ptr<LinearSolver>
      linear_solver(LinearSolver::Create(linear_solver_options));

  Evaluator::Options evaluator_options;
  evaluator_options.linear_solver_type = DENSE_QR;
  evaluator_options.num_eliminate_blocks = 0;
  evaluator_options.num_threads = 1;
  scoped_ptr<Evaluator> evaluator(Evaluator::Create(evaluator_options, program,  &error));
  CHECK_NOTNULL(evaluator.get());

  scoped_ptr<SparseMatrix> jacobian(evaluator->CreateJacobian());
  CHECK_NOTNULL(jacobian.get());

  TrustRegionStrategy::Options trust_region_strategy_options;
  trust_region_strategy_options.linear_solver = linear_solver.get();
  scoped_ptr<TrustRegionStrategy> strategy(
      TrustRegionStrategy::Create(trust_region_strategy_options));
  CHECK_NOTNULL(strategy.get());

  Minimizer::Options minimizer_options;
  minimizer_options.evaluator = evaluator.get();
  minimizer_options.jacobian = jacobian.get();
  minimizer_options.trust_region_strategy = strategy.get();

  TrustRegionMinimizer minimizer;
  minimizer.Minimize(minimizer_options, parameters, summary);
}


void InnerIterationMinimizer::CountResidualBlocksPerParameterBlock(const int num_eliminate_blocks) {
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
  for (int i = 0; i < num_eliminate_blocks; ++i) {
    residual_block_offsets_[i+1] = residual_block_offsets_[i] + counts[i];
    CHECK_GT(counts[i], 0);
  }
}


}  // namespace internal
}  // namespace ceres
