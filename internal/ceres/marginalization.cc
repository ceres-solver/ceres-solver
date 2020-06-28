#include "ceres/marginalization.h"

#include <memory>
#include <numeric>

#include "ceres/compressed_row_sparse_matrix.h"
#include "ceres/cost_function.h"
#include "ceres/evaluator.h"
#include "ceres/inner_product_computer.h"
#include "ceres/internal/eigen.h"
#include "ceres/invert_psd_matrix.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/marginalization_prior_cost_function.h"
#include "ceres/ordered_groups.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/problem_impl.h"
#include "ceres/program.h"
#include "ceres/reorder_program.h"
#include "ceres/residual_block.h"
#include "ceres/sparse_matrix.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {

using std::map;
using std::set;
using std::vector;

namespace {

int SumTangentSize(const vector<ParameterBlock*>& parameter_blocks) {
  return std::accumulate(
      parameter_blocks.begin(),
      parameter_blocks.end(),
      0,
      [](int sz, const ParameterBlock* pb) { return sz + pb->TangentSize(); });
}

// Compute the matrix S such that S * S^T reconstructs the symmetric positive
// definite matrix from which ldlt was computed.
bool CholeskyFactor(const Eigen::LDLT<Matrix>& ldlt, Matrix* S) {
  const Matrix L = ldlt.matrixL();
  const Vector D = ldlt.vectorD();
  // Check full rank
  for (int i = 0; i < D.size(); ++i) {
    if (D(i) <= 0.0) {
      return false;
    }
  }
  *S = ldlt.transpositionsP().transpose() * L *
       D.array().sqrt().matrix().asDiagonal();
  return true;
}

bool CholeskyFactorWithEigendecomposition(const Matrix& P, Matrix* S) {
  Eigen::SelfAdjointEigenSolver<Matrix> es(P);
  Vector v = es.eigenvalues();
  Matrix U = es.eigenvectors();
  int rank = v.rows();
  for (int i = 0; i < v.rows(); ++i) {
    if (v[i] <= 0.0) {
      --rank;
    }
  }
  if (rank == 0) {
    return false;
  }
  *S = U.rightCols(rank) * v.tail(rank).array().sqrt().matrix().asDiagonal();

  return true;
}

// Compute A^T * A from a sparse matrix A.
Matrix GramMatrix(const BlockSparseMatrix* A) {
  CHECK_NOTNULL(A);
  Matrix ATA_dense(A->num_cols(), A->num_cols());
  std::unique_ptr<InnerProductComputer> inner_product_computer(
      InnerProductComputer::Create(
          *A, CompressedRowSparseMatrix::UPPER_TRIANGULAR));
  inner_product_computer->Compute();
  const CompressedRowSparseMatrix& ATA = inner_product_computer->result();

  ATA.ToDenseMatrix(&ATA_dense);
  // Symmetrize from the upper part.
  for (int i = 0; i < ATA_dense.rows(); ++i) {
    for (int j = i + 1; j < ATA_dense.cols(); ++j) {
      ATA_dense(j, i) = ATA_dense(i, j);
    }
  }
  return ATA_dense;
}

// Build a problem consisting of the parameter blocks to be marginalized,
// their Markov blanket, and error terms involving the parameter blocks to
// marginalize. The lifetime of external_problem must be longer than that of the
// returned problem.
ProblemImpl* BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  Problem::Options options;
  options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.manifold_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.enable_fast_removal = true;
  set<ResidualBlockId> marginalized_blocks_residual_ids;

  ProblemImpl* new_problem = new ProblemImpl(options);
  for (auto it = parameter_blocks_to_marginalize.begin();
       it != parameter_blocks_to_marginalize.end();
       ++it) {
    double* parameter_block = *it;
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

      vector<double*> parameter_blocks;
      external_problem.GetParameterBlocksForResidualBlock(residual_block_id,
                                                          &parameter_blocks);

      // Add parameter blocks to marginalize.
      for (size_t k = 0; k < parameter_blocks.size(); ++k) {
        if (new_problem->HasParameterBlock(parameter_blocks[k])) {
          continue;
        }

        CHECK(!external_problem.IsParameterBlockConstant(parameter_blocks[k]))
            << "Constant parameter blocks cannot be marginalized";

        const int size =
            external_problem.ParameterBlockSize(parameter_blocks[k]);
        const Manifold* manifold =
            external_problem.GetManifold(parameter_blocks[k]);

        new_problem->AddParameterBlock(
            parameter_blocks[k], size, const_cast<Manifold*>(manifold));
      }

      const CostFunction* cost_function =
          external_problem.GetCostFunctionForResidualBlock(residual_block_id);
      const LossFunction* loss_function =
          external_problem.GetLossFunctionForResidualBlock(residual_block_id);
      new_problem->AddResidualBlock(const_cast<CostFunction*>(cost_function),
                                    const_cast<LossFunction*>(loss_function),
                                    parameter_blocks.data(),
                                    static_cast<int>(parameter_blocks.size()));
    }
  }
  return new_problem;
}

static constexpr int kMarginalizedGroupId = 0;
static constexpr int kMarkovBlanketGroupId = 1;

// Get an ordering where the parameter blocks to be marginalized are followed by
// parameter blocks in the Markov blanket.
ParameterBlockOrdering GetOrderingForMarginalizedBlocksAndMarkovBlanket(
    const ProblemImpl& problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  ParameterBlockOrdering ordering;
  vector<double*> added_parameter_blocks;
  problem.GetParameterBlocks(&added_parameter_blocks);
  for (int b = 0; b < added_parameter_blocks.size(); ++b) {
    double* added_parameter_block = added_parameter_blocks[b];
    if (parameter_blocks_to_marginalize.count(added_parameter_block)) {
      ordering.AddElementToGroup(added_parameter_block, kMarginalizedGroupId);
    } else {
      ordering.AddElementToGroup(added_parameter_block, kMarkovBlanketGroupId);
    }
  }
  return ordering;
}

void EvaluateProblem(
    const int num_elimination_blocks,
    const map<double*, const double*>* parameter_block_linearization_states,
    ProblemImpl* problem,
    Matrix* jtj,
    Vector* gradient,
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

  std::unique_ptr<Evaluator> evaluator(
      Evaluator::Create(evaluator_options, problem->mutable_program(), &error));
  CHECK(evaluator.get() != nullptr) << "Failed creating evaluator";

  std::unique_ptr<SparseMatrix> sparse_jacobian(evaluator->CreateJacobian());

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

    CHECK(evaluator->Evaluate(linearization_state_vector.data(),
                              &cost,
                              nullptr,
                              nullptr,
                              sparse_jacobian.get()));

    Matrix J;
    sparse_jacobian->ToDenseMatrix(&J);

    Vector residuals(problem->NumResiduals());

    // Compute the residuals, restoring the state vector.
    CHECK(evaluator->Evaluate(
        state_vector->data(), &cost, residuals.data(), nullptr, nullptr));

    // Compute the gradient.
    gradient->setZero();
    sparse_jacobian->LeftMultiply(residuals.data(), gradient->data());
  } else {
    CHECK(evaluator->Evaluate(state_vector->data(),
                              &cost,
                              nullptr,
                              gradient->data(),
                              sparse_jacobian.get()));
  }

  *jtj =
      GramMatrix(dynamic_cast<const BlockSparseMatrix*>(sparse_jacobian.get()));
}

// Compute the Schur complement of the first block of the system jtj, gradient.
void SchurComplement(const Matrix& jtj,
                     const Vector& gradient,
                     const int size_marginalized,
                     Matrix* jtj_marginal,
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
  const Matrix jtj_11_inv = InvertPSDMatrix<Eigen::Dynamic>(/*assume_full_rank=*/ false, jtj_11);
  *jtj_marginal = jtj_22 - jtj_12.transpose() * jtj_11_inv * jtj_12;
  *gradient_marginal = gradient2 - jtj_12.transpose() * jtj_11_inv * gradient1;
}

// Compute the parameters of a linear cost function, and return the blocks
// in the Markov blanket of the parameter blocks to marginalize.
bool ComputeMarginalizationPriorData(
    const MarginalizationOptions& options,
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    std::unique_ptr<ProblemImpl>& local_problem,
    Matrix* jacobian,
    Vector* b,
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
  CHECK(ApplyOrdering(local_problem->parameter_map(),
                      ordering,
                      local_problem->mutable_program(),
                      &error))
      << "Failed to apply ordering required for marginalization";

  local_problem->mutable_program()->SetParameterOffsetsAndIndex();

  if (!local_problem->program().IsFeasible(&error)) {
    return false;
  }

  for (const ParameterBlock* pb :
       local_problem->mutable_program()->parameter_blocks()) {
    vector<ResidualBlockId> residual_blocks;
    local_problem->GetResidualBlocksForParameterBlock(pb->state(),
                                                      &residual_blocks);
  }
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

  if (SumTangentSize(*blanket_parameter_blocks) == 0) {
    return false;
  }

  const int tan_size_marginalized =
      SumTangentSize(ordered_parameter_blocks_marginalized);

  Matrix jtj;
  Vector gradient;
  Vector state_vector;
  EvaluateProblem(parameter_blocks_to_marginalize.size(),
                  parameter_block_linearization_states,
                  local_problem.get(),
                  &jtj,
                  &gradient,
                  &state_vector);

  Matrix jtj_marginal;
  Vector gradient_marginalization_prior;
  SchurComplement(jtj,
                  gradient,
                  tan_size_marginalized,
                  &jtj_marginal,
                  &gradient_marginalization_prior);

  const Eigen::LDLT<Matrix> ldlt = jtj_marginal.ldlt();
  if (ldlt.info() != Eigen::Success) {
    return false;
  }

  Matrix sqrt_factor;
  if (options.assume_full_rank) {
    if (!CholeskyFactor(ldlt, &sqrt_factor)) {
      return false;
    }
  } else {
    // Reduce the dimension of the residual. See
    // Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
    // node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
    // (2014): 1371-1385.
    if (!CholeskyFactorWithEigendecomposition(jtj_marginal, &sqrt_factor)) {
      // Rank 0, unexpected.
      return false;
    }
  }

  *jacobian = sqrt_factor.transpose();
  *b = *jacobian * ldlt.solve(gradient_marginalization_prior);
  return true;
}

bool ComputeMarginalizationPrior(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    Problem* problem,
    vector<double*>* blanket_parameter_blocks_state,
    CostFunction** marginalization_prior_cost_function) {
  CHECK_NOTNULL(marginalization_prior_cost_function);
  CHECK_NOTNULL(blanket_parameter_blocks_state);

  Matrix marginalization_prior_jacobian;
  Vector marginalization_prior_b;
  vector<ParameterBlock*> blanket_parameter_block_ptrs;
  std::unique_ptr<ProblemImpl> localProblem;

  if (!ComputeMarginalizationPriorData(options,
                                       *problem,
                                       parameter_blocks_to_marginalize,
                                       parameter_block_linearization_states,
                                       localProblem,
                                       &marginalization_prior_jacobian,
                                       &marginalization_prior_b,
                                       &blanket_parameter_block_ptrs)) {
    return false;
  }

  const size_t numMarkovBlanketBlocks = blanket_parameter_block_ptrs.size();
  vector<const Manifold*> blanket_manifolds(numMarkovBlanketBlocks);
  vector<vector<double>> blanket_parameter_block_states(numMarkovBlanketBlocks);
  vector<int> blanket_tan_sizes(numMarkovBlanketBlocks);
  blanket_parameter_blocks_state->resize(numMarkovBlanketBlocks);
  for (size_t i = 0; i < numMarkovBlanketBlocks; ++i) {
    ParameterBlock* pb = blanket_parameter_block_ptrs[i];
    double* pb_state = pb->mutable_user_state();
    int pb_global_size = pb->Size();
    int pb_tan_size = pb->TangentSize();
    (*blanket_parameter_blocks_state)[i] = pb_state;
    blanket_manifolds[i] = localProblem->GetManifold(pb_state);

    blanket_parameter_block_states[i].assign(pb_state,
                                             pb_state + pb_global_size);
    blanket_tan_sizes[i] = pb_tan_size;
  }

  *marginalization_prior_cost_function =
      new MarginalizationPriorCostFunction(marginalization_prior_jacobian,
                                           marginalization_prior_b,
                                           blanket_parameter_block_states,
                                           blanket_tan_sizes,
                                           blanket_manifolds);

  return true;
}

bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    ResidualBlockId* marginalization_prior_id,
    const map<double*, const double*>* parameter_block_linearization_states) {
  vector<double*> markov_blanket_ordered_parameter_blocks;

  CostFunction* marginalization_prior_cost_function;

  if (!ComputeMarginalizationPrior(options,
                                   parameter_blocks_to_marginalize,
                                   parameter_block_linearization_states,
                                   problem,
                                   &markov_blanket_ordered_parameter_blocks,
                                   &marginalization_prior_cost_function)) {
    return false;
  }

  // Remove marginalized blocks.
  for (auto it = parameter_blocks_to_marginalize.begin();
       it != parameter_blocks_to_marginalize.end();
       ++it) {
    problem->RemoveParameterBlock(*it);
  }

  // Add the cost function for the marginal factor to the problem, attaching it
  // to the Markov blanket of the marginalized blocks.
  const ResidualBlockId mp_id = problem->AddResidualBlock(
      marginalization_prior_cost_function,
      nullptr,
      markov_blanket_ordered_parameter_blocks.data(),
      static_cast<int>(markov_blanket_ordered_parameter_blocks.size()));

  if (marginalization_prior_id) {
    *marginalization_prior_id = mp_id;
  }

  return true;
}
}  // anonymous namespace

}  // namespace internal

bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    ResidualBlockId* marginalization_prior_id,
    const std::map<double*, const double*>*
        parameter_block_linearization_states) {
  return internal::MarginalizeOutVariables(
      options,
      parameter_blocks_to_marginalize,
      problem,
      marginalization_prior_id,
      parameter_block_linearization_states);
}

}  // namespace ceres
         