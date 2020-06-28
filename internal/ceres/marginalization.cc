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
#include "ceres/marginalizable_parameterization.h"
#include "ceres/marginalization_prior_cost_function.h"
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

using std::map;
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


static bool ComputeSqrtFactorWithEigendecomposition(const Matrix& P, Matrix* S)
{
  Eigen::SelfAdjointEigenSolver<Matrix> es(P);
  Vector v = es.eigenvalues();
  Matrix U = es.eigenvectors();
  int rank = v.rows();
  for (int i = 0 ; i < v.rows() ; i++ )
  {
    if (v[i] <= 0.0 )
    {
      --rank;
    }
  }
  if (rank == 0)
  {
    return false;
  }
  *S = U.rightCols(rank) * v.tail(rank).array().sqrt().matrix().asDiagonal();

  return true;
}

// Compute A^T * A from a sparse matrix A.
static Matrix GramMatrix(const BlockSparseMatrix* A) {
  CHECK_NOTNULL(A);
  Matrix ATA_dense(A->num_cols(), A->num_cols());
  std::unique_ptr<InnerProductComputer> inner_product_computer;

  inner_product_computer.reset(InnerProductComputer::Create(
      *A, CompressedRowSparseMatrix::UPPER_TRIANGULAR));
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
          external_problem.GetParameterization(parameter_block);

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
        // For blocks to marginalize, add the local parameterization.
        // If the marginalization prior is formulated in the tangent space,
        // add the local parameterization for the blocks in the Markov blanket.
        AddParameterBlock(external_problem, new_problem,
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

static void EvaluateProblem(
    const int num_elimination_blocks,
    const map<double*, const double*>* parameter_block_linearization_states,
    ProblemImpl* problem, Matrix* jtj, Vector* gradient, Vector* state_vector) {
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

  if (parameter_block_linearization_states) {
    // Compute the Jacobian using the linearization states.
    for (ParameterBlock* pb : problem->program().parameter_blocks()) {
      const auto it =
          parameter_block_linearization_states->find(pb->mutable_user_state());
      CHECK(it != parameter_block_linearization_states->end())
          << "Could not find linearization state for parameter block!";
      pb->SetState(it->second);
    }

    Vector linearization_state_vector(problem->NumParameters());
    problem->program().ParameterBlocksToStateVector(
        linearization_state_vector.data());

    CHECK(evaluator->Evaluate(linearization_state_vector.data(), &cost, NULL,
                              NULL, sparse_jacobian.get()));

    Matrix J;
    sparse_jacobian->ToDenseMatrix(&J);

    Vector residuals(problem->NumResiduals());

    // Compute the residuals, restoring the state vector.
    CHECK(evaluator->Evaluate(state_vector->data(), &cost, residuals.data(),
                              NULL, NULL));

    // Compute the gradient.
    gradient->setZero();
    sparse_jacobian->LeftMultiply(residuals.data(), gradient->data());
  } else {
    CHECK(evaluator->Evaluate(state_vector->data(), &cost, NULL,
                              gradient->data(), sparse_jacobian.get()));
  }

  *jtj =
      GramMatrix(dynamic_cast<const BlockSparseMatrix*>(sparse_jacobian.get()));
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
    const MarginalizationOptions& options,
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    std::unique_ptr<ProblemImpl>& local_problem,
    Matrix* jacobian, Vector* b,
    vector<ParameterBlock*>* blanket_parameter_blocks) {
  CHECK_NOTNULL(jacobian);
  CHECK_NOTNULL(b);
  CHECK_NOTNULL(blanket_parameter_blocks);

  local_problem.reset(BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
      external_problem, parameter_blocks_to_marginalize));

  const ParameterBlockOrdering ordering =
      GetOrderingForMarginalizedBlocksAndMarkovBlanket(
          *local_problem, parameter_blocks_to_marginalize);

  std::string error;
  CHECK(ApplyOrdering(local_problem->parameter_map(), ordering,
                      local_problem->mutable_program(), &error))
      << "Failed to apply ordering required for marginalization";

  local_problem->mutable_program()->SetParameterOffsetsAndIndex();

  CHECK(LexicographicallyOrderResidualBlocks(
      parameter_blocks_to_marginalize.size(), local_problem->mutable_program(),
      &error))
      << "Failed to order residual blocks";

  const vector<ParameterBlock*>& ordered_parameter_blocks =
      local_problem->program().parameter_blocks();
  *blanket_parameter_blocks =
      vector<ParameterBlock*>(ordered_parameter_blocks.end() -
                                  ordering.GroupSize(kMarkovBlanketGroupId),
                              ordered_parameter_blocks.end());
  const auto ordered_parameter_blocks_marginalized =
      vector<ParameterBlock*>(ordered_parameter_blocks.begin(),
                              ordered_parameter_blocks.begin() +
                                  ordering.GroupSize(kMarginalizedGroupId));

  const int size_blanket = SumLocalSize(*blanket_parameter_blocks);

  if (size_blanket == 0)
  {
    return false;
  }
    
  const int local_size_marginalized =
      SumLocalSize(ordered_parameter_blocks_marginalized);
  const int information_size = local_size_marginalized + size_blanket;

  Matrix jtj;
  Vector gradient;
  Vector state_vector;
  EvaluateProblem(parameter_blocks_to_marginalize.size(),
                  parameter_block_linearization_states, local_problem.get(), &jtj,
                  &gradient, &state_vector);

  Matrix jtj_marginal;
  Vector gradient_marginal_factor;
  SchurComplement(jtj, gradient, local_size_marginalized, &jtj_marginal,
                  &gradient_marginal_factor);

  const Eigen::LDLT<Matrix> ldlt = jtj_marginal.ldlt();
  if (ldlt.info() != Eigen::Success) {
    return false;
  }
  
  Matrix sqrt_factor;
  if (options.assume_full_rank)
  {
    sqrt_factor = ComputeSqrtFactor(ldlt);
  }
  else
  {
    // Reduce the dimension of the residual. See
    // Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
    // node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
    // (2014): 1371-1385.
    if (!ComputeSqrtFactorWithEigendecomposition(jtj_marginal, &sqrt_factor))
    {
      // Rank 0, unexpected.
      return false;
    }
  }

  *jacobian = sqrt_factor.transpose();
  *b = *jacobian * ldlt.solve(gradient_marginal_factor);
  return true;
}

static bool ComputeWithTangentSpaceFormulation(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    Problem* problem, vector<double*>* blanket_parameter_blocks_state,
    CostFunction** marginal_factor_cost_function) {
  CHECK_NOTNULL(marginal_factor_cost_function);
  CHECK_NOTNULL(blanket_parameter_blocks_state);

  Matrix marginal_factor_jacobian;
  Vector marginal_factor_b;
  vector<ParameterBlock*> blanket_parameter_block_ptrs;
  std::unique_ptr<ProblemImpl> localProblem;

  if (!ComputeMarginalFactorData(options,
                                 *problem, parameter_blocks_to_marginalize,
                                 parameter_block_linearization_states,
                                 localProblem,
                                 &marginal_factor_jacobian, &marginal_factor_b,
                                 &blanket_parameter_block_ptrs)) {
    return false;
  }

  const size_t numMarkovBlanketBlocks = blanket_parameter_block_ptrs.size();
  vector<const MarginalizableParameterization*> blanket_parameterizations(numMarkovBlanketBlocks);
  vector<vector<double>> blanket_parameter_block_states(numMarkovBlanketBlocks);
  vector<int> blanket_local_sizes(numMarkovBlanketBlocks);
  blanket_parameter_blocks_state->resize(numMarkovBlanketBlocks);
  for (size_t i = 0; i < numMarkovBlanketBlocks; i++) {
    ParameterBlock* pb = blanket_parameter_block_ptrs[i];
    double* pb_state = pb->mutable_user_state();
    int pb_global_size = pb->Size();
    int pb_local_size = pb->LocalSize();
    (*blanket_parameter_blocks_state)[i] = pb_state;
    blanket_parameterizations[i] =
        dynamic_cast<const MarginalizableParameterization*>(
            localProblem->GetParameterization(pb_state));
    CHECK(blanket_parameterizations[i]) << "Parameterization is not a MarginalizableParameterization";

    blanket_parameter_block_states[i].assign(pb_state, pb_state + pb_global_size);
    blanket_local_sizes[i] = pb_local_size;
  }

  *marginal_factor_cost_function = new MarginalizationPriorCostFunction(
      marginal_factor_jacobian, marginal_factor_b,
      blanket_parameter_block_states, 
      blanket_local_sizes, blanket_parameterizations);

  return true;
}

static bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize, Problem* problem,
    ResidualBlockId* marginal_factor_id,
    const map<double*, const double*>* parameter_block_linearization_states) {
  vector<double*> markov_blanket_ordered_parameter_blocks;

  CostFunction* marginal_factor_cost_function;

  if (!ComputeWithTangentSpaceFormulation(options,
                    parameter_blocks_to_marginalize,
                    parameter_block_linearization_states, problem,
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
  ResidualBlockId mf_id = problem->AddResidualBlock(
      marginal_factor_cost_function, NULL,
      markov_blanket_ordered_parameter_blocks.data(),
      static_cast<int>(markov_blanket_ordered_parameter_blocks.size()));

  if (marginal_factor_id) {
    *marginal_factor_id = mf_id;
  }

  return true;
}

}  // namespace internal

bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const std::set<double*>& parameter_blocks_to_marginalize, Problem* problem,
    ResidualBlockId* marginal_factor_id,
    const std::map<double*, const double*>*
        parameter_block_linearization_states) {
  return internal::MarginalizeOutVariables(
      options,
      parameter_blocks_to_marginalize, problem, marginal_factor_id,
      parameter_block_linearization_states);
}

}  // namespace ceres
