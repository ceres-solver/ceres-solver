// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <memory>

#include "ceres/casts.h"
#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cost_function.h"
#include "ceres/evaluator.h"
#include "ceres/inner_product_computer.h"
#include "ceres/internal/eigen.h"
#include "ceres/invert_psd_matrix.h"
#include "ceres/linear_cost_function.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/reorder_program.h"
#include "ceres/residual_block.h"
#include "ceres/residual_block_utils.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

using std::set;
using std::vector;

static int SumGlobalSize(const vector<ParameterBlock*>& parameter_blocks) {
  int sum = 0;
  for (int b = 0; b < parameter_blocks.size(); b++) {
    auto* p = parameter_blocks[b];
    sum += p->Size();
  }
  return sum;
}

static int SumLocalSize(const vector<ParameterBlock*>& parameter_blocks) {
  int sum = 0;
  for (int b = 0; b < parameter_blocks.size(); b++) {
    auto* p = parameter_blocks[b];
    sum += p->LocalSize();
  }
  return sum;
}

// Compute the matrix S such that S * S^T reconstructs the symmetric positive
// semidefinite matrix from which ldlt was computed.
static Matrix ComputeSqrtFactor(const Eigen::LDLT<Matrix>& ldlt) {
  const Matrix L = ldlt.matrixL();
  Matrix result = ldlt.transpositionsP().transpose() * L;
  Vector D = ldlt.vectorD();
  // Diagonal elements can be very small and negative.
  for (int i = 0; i < D.size(); ++i) {
    D(i) = std::max(D(i), 0.0);
  }
  result *= D.array().sqrt().matrix().asDiagonal();
  return result;
}

// Compute A^T * A from a sparse matrix A.
static Matrix GramMatrix(const SparseMatrix* A) {
  CHECK_NOTNULL(A);
  Matrix ATA_dense(A->num_cols(), A->num_cols());
  std::unique_ptr<InnerProductComputer> inner_product_computer;
  const auto* A_bs = dynamic_cast<const BlockSparseMatrix*>(A);
  CHECK(A_bs) << "Not block sparse!\n";

  inner_product_computer.reset(InnerProductComputer::Create(
      *A_bs, CompressedRowSparseMatrix::UPPER_TRIANGULAR));
  inner_product_computer->Compute();
  const CompressedRowSparseMatrix& ATA = inner_product_computer->result();

  ATA.ToDenseMatrix(&ATA_dense);
  // Symmetrize from upper part.
  for (int i = 0; i < ATA_dense.rows(); i++) {
    for (int j = i + 1; j < ATA_dense.cols(); j++) {
      ATA_dense(j, i) = ATA_dense(i, j);
    }
  }
  return ATA_dense;
}

// Copy a parameter block from the old problem to the new problem. The new
// problem should not take ownership of the local parameterization.
static void AddParameterBlock(const Problem& external_problem,
                              const bool add_parameterization,
                              ProblemImpl* new_problem,
                              double* parameter_block) {
  CHECK_NOTNULL(new_problem);
  if (new_problem->HasParameterBlock(parameter_block)) {
    return;
  }

  CHECK(!external_problem.IsParameterBlockConstant(parameter_block))
      << "Constant parameter blocks cannot be marginalized";

  const int size = external_problem.ParameterBlockSize(parameter_block);
  const LocalParameterization* parameterization =
      add_parameterization
          ? external_problem.GetParameterization(parameter_block)
          : NULL;

  new_problem->AddParameterBlock(
      parameter_block, size,
      const_cast<LocalParameterization*>(parameterization));
}

// Build a problem consisting of the parameter blocks to be marginalized,
// their Markov blanket, and error terms involving the parameter blocks to
// marginalize.
static ProblemImpl* BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  Problem::Options options;
  options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.enable_fast_removal = true;
  set<ResidualBlockId> marginalized_blocks_residual_ids;

  ProblemImpl* new_problem = new ProblemImpl(options);
  for (auto it = parameter_blocks_to_marginalize.begin();
       it != parameter_blocks_to_marginalize.end(); ++it) {
    auto* parameter_block = *it;
    vector<ResidualBlockId> residual_blocks;
    external_problem.GetResidualBlocksForParameterBlock(parameter_block,
                                                        &residual_blocks);

    for (size_t j = 0; j < residual_blocks.size(); ++j) {
      // Add this residual block if we have not already.
      const ResidualBlockId& residual_block_id = residual_blocks[j];
      if (marginalized_blocks_residual_ids.count(residual_block_id)) {
        continue;
      }
      marginalized_blocks_residual_ids.insert(residual_block_id);
      const CostFunction* cost_function =
          external_problem.GetCostFunctionForResidualBlock(residual_block_id);
      const LossFunction* loss_function =
          external_problem.GetLossFunctionForResidualBlock(residual_block_id);

      vector<double*> parameter_blocks;
      external_problem.GetParameterBlocksForResidualBlock(residual_block_id,
                                                          &parameter_blocks);

      for (size_t k = 0; k < parameter_blocks.size(); ++k) {
        // For blocks to marginalize, add local parameterization.
        const bool add_parameterization = static_cast<bool>(
            parameter_blocks_to_marginalize.count(parameter_blocks[k]));
        AddParameterBlock(external_problem, add_parameterization, new_problem,
                          parameter_blocks[k]);
      }

      const ResidualBlockId new_block_id = new_problem->AddResidualBlock(
          const_cast<CostFunction*>(cost_function),
          const_cast<LossFunction*>(loss_function), parameter_blocks.data(),
          static_cast<int>(parameter_blocks.size()));
    }
  }
  return new_problem;
}

static constexpr int kMarginalizedGroupId = 0;
static constexpr int kMarkovBlanketGroupId = 1;

// Get an ordering where the parameter blocks to be marginalized are followed by
// parameter blocks in the Markov blanket.
static ParameterBlockOrdering GetOrderingForMarginalizedBlocksAndMarkovBlanket(
    const ProblemImpl& problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  ParameterBlockOrdering ordering;
  vector<double*> added_parameter_blocks;
  problem.GetParameterBlocks(&added_parameter_blocks);
  for (int b = 0; b < added_parameter_blocks.size(); b++) {
    auto* added_parameter_block = added_parameter_blocks[b];
    if (parameter_blocks_to_marginalize.count(added_parameter_block)) {
      ordering.AddElementToGroup(added_parameter_block, kMarginalizedGroupId);
    } else {
      ordering.AddElementToGroup(added_parameter_block, kMarkovBlanketGroupId);
    }
  }
  return ordering;
}

static void EvaluateProblem(const int num_elimination_blocks,
                            ProblemImpl* problem, Matrix* jtj, Vector* gradient,
                            Vector* state_vector) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(gradient);
  CHECK_NOTNULL(state_vector);
  CHECK_NOTNULL(jtj);

  gradient->resize(problem->program().NumEffectiveParameters());
  state_vector->resize(problem->NumParameters());

  std::string error;
  Evaluator::Options evaluator_options;
  evaluator_options.linear_solver_type = DENSE_SCHUR;
  evaluator_options.num_eliminate_blocks = num_elimination_blocks;
  evaluator_options.context = problem->context();

  std::unique_ptr<Evaluator> evaluator;
  evaluator.reset(
      Evaluator::Create(evaluator_options, problem->mutable_program(), &error));
  CHECK(evaluator.get() != NULL) << "Failed creating evaluator";

  std::unique_ptr<SparseMatrix> sparse_jacobian;
  sparse_jacobian.reset(evaluator->CreateJacobian());

  problem->program().ParameterBlocksToStateVector(state_vector->data());

  double cost;
  CHECK(evaluator->Evaluate(state_vector->data(), &cost, NULL, gradient->data(),
                            sparse_jacobian.get()));

  *jtj = GramMatrix(sparse_jacobian.get());
}

// Compute the Schur complement of the first block of the system jtj, gradient.
static void SchurComplement(const Matrix& jtj, const Vector& gradient,
                            const int size_marginalized, Matrix* jtj_marginal,
                            Vector* gradient_marginal) {
  CHECK_NOTNULL(jtj_marginal);
  CHECK_NOTNULL(gradient_marginal);
  CHECK(size_marginalized < gradient.size());
  CHECK(jtj.rows() == gradient.size() && jtj.rows() == jtj.cols());
  const int global_size_blanket = gradient.size() - size_marginalized;
  const Vector gradient1 = gradient.head(size_marginalized);
  const Vector gradient2 = gradient.tail(global_size_blanket);
  const auto jtj_22 =
      jtj.bottomRightCorner(global_size_blanket, global_size_blanket);
  const auto jtj_12 =
      jtj.topRightCorner(size_marginalized, global_size_blanket);
  const Matrix jtj_11 = jtj.topLeftCorner(size_marginalized, size_marginalized);
  const Matrix jtj_11_inv = InvertPSDMatrix<Eigen::Dynamic>(false, jtj_11);
  *jtj_marginal = jtj_22 - jtj_12.transpose() * jtj_11_inv * jtj_12;
  *gradient_marginal = gradient2 - jtj_12.transpose() * jtj_11_inv * gradient1;
}

// Compute the parameters of a linear cost function, and return the blocks
// in the Markov blanket of the parameter blocks to marginalize.
static bool ComputeMarginalFactorData(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize, Matrix* jacobian,
    Vector* b, vector<double*>* markov_blanket_parameter_blocks,
    vector<int>* parameter_block_sizes) {
  std::unique_ptr<ProblemImpl> problem;
  problem.reset(BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
      external_problem, parameter_blocks_to_marginalize));

  const ParameterBlockOrdering ordering =
      GetOrderingForMarginalizedBlocksAndMarkovBlanket(
          *problem, parameter_blocks_to_marginalize);

  std::string error;
  CHECK(ApplyOrdering(problem->parameter_map(), ordering,
                      problem->mutable_program(), &error))
      << "Failed to apply ordering required for marginalization";

  problem->mutable_program()->SetParameterOffsetsAndIndex();

  CHECK(LexicographicallyOrderResidualBlocks(
      parameter_blocks_to_marginalize.size(), problem->mutable_program(),
      &error))
      << "Failed to order residual blocks";

  const vector<ParameterBlock*>& ordered_parameter_blocks =
      problem->program().parameter_blocks();
  const auto ordered_parameter_blocks_blanket =
      vector<ParameterBlock*>(ordered_parameter_blocks.end() -
                                  ordering.GroupSize(kMarkovBlanketGroupId),
                              ordered_parameter_blocks.end());
  const auto ordered_parameter_blocks_marginalized =
      vector<ParameterBlock*>(ordered_parameter_blocks.begin(),
                              ordered_parameter_blocks.begin() +
                                  ordering.GroupSize(kMarginalizedGroupId));

  const int global_size_blanket =
      SumGlobalSize(ordered_parameter_blocks_blanket);
  if (global_size_blanket == 0) {
    return false;
  }
  const int local_size_marginalized =
      SumLocalSize(ordered_parameter_blocks_marginalized);
  const int information_size = local_size_marginalized + global_size_blanket;

  Matrix jtj;
  Vector gradient;
  Vector state_vector;
  EvaluateProblem(parameter_blocks_to_marginalize.size(), problem.get(), &jtj,
                  &gradient, &state_vector);

  Matrix jtj_marginal_factor;
  Vector gradient_marginal_factor;
  SchurComplement(jtj, gradient, local_size_marginalized, &jtj_marginal_factor,
                  &gradient_marginal_factor);

  const Eigen::LDLT<Matrix> ldlt = jtj_marginal_factor.ldlt();
  if (ldlt.info() != Eigen::Success) {
    return false;
  }

  const Matrix sqrt_factor = ComputeSqrtFactor(ldlt);
  const Vector state_vector_blanket = state_vector.tail(global_size_blanket);

  *jacobian = sqrt_factor.transpose();
  *b = sqrt_factor.transpose() *
       (ldlt.solve(gradient_marginal_factor) - state_vector_blanket);

  markov_blanket_parameter_blocks->clear();
  for (auto* parameter_block : ordered_parameter_blocks_blanket) {
    markov_blanket_parameter_blocks->push_back(
        parameter_block->mutable_user_state());
    parameter_block_sizes->push_back(parameter_block->Size());
  }

  return true;
}

static bool Compute(const set<double*>& parameter_blocks_to_marginalize,
                    Problem* problem,
                    vector<double*>* markov_blanket_parameter_blocks,
                    LinearCostFunction** marginal_factor_cost_function) {
  CHECK_NOTNULL(marginal_factor_cost_function);
  CHECK_NOTNULL(markov_blanket_parameter_blocks);

  vector<int> marginal_factor_parameter_block_sizes;
  Matrix marginal_factor_jacobian;
  Vector marginal_factor_b;

  if (!ComputeMarginalFactorData(*problem, parameter_blocks_to_marginalize,
                                 &marginal_factor_jacobian, &marginal_factor_b,
                                 markov_blanket_parameter_blocks,
                                 &marginal_factor_parameter_block_sizes)) {
    return false;
  }

  *marginal_factor_cost_function =
      new LinearCostFunction(marginal_factor_jacobian, marginal_factor_b,
                             marginal_factor_parameter_block_sizes);
  return true;
}

static bool MarginalizeOutVariables(
    const set<double*>& parameter_blocks_to_marginalize, Problem* problem) {
  vector<double*> markov_blanket_ordered_parameter_blocks;
  LinearCostFunction* marginal_factor_cost_function;
  if (!Compute(parameter_blocks_to_marginalize, problem,
               &markov_blanket_ordered_parameter_blocks,
               &marginal_factor_cost_function)) {
    return false;
  }

  // Remove marginalized blocks.
  for (auto it = parameter_blocks_to_marginalize.begin();
       it != parameter_blocks_to_marginalize.end(); ++it) {
    problem->RemoveParameterBlock(*it);
  }

  // Add the cost function for the marginal factor to the problem, attaching it
  // to the Markov blanket of the marginalized blocks.
  problem->AddResidualBlock(
      marginal_factor_cost_function, NULL,
      markov_blanket_ordered_parameter_blocks.data(),
      static_cast<int>(markov_blanket_ordered_parameter_blocks.size()));

  return true;
}

}  // namespace internal

bool MarginalizeOutVariables(
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem) {
  return internal::MarginalizeOutVariables(parameter_blocks_to_marginalize,
                                           problem);
}

}  // namespace ceres
