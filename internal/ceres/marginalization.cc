#include "ceres/marginalization.h"

#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/dense_sparse_matrix.h"
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
    if (D(i) <= std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  *S = ldlt.transpositionsP().transpose() * L *
       D.array().sqrt().matrix().asDiagonal();
  return true;
}

// Get U and D such that U * D * U^T = P, D containing the nonzero eigenvalues
// of P on the diagonal.
bool GetEigendecomposition(const Matrix& P, Matrix* U, Vector* D) {
  CHECK_NOTNULL(U);
  CHECK_NOTNULL(D);

  Eigen::SelfAdjointEigenSolver<Matrix> es(P);
  Vector v = es.eigenvalues();
  *U = es.eigenvectors();
  int rank = v.rows();
  for (int i = 0; i < v.rows(); ++i) {
    if (v[i] <= std::numeric_limits<double>::epsilon()) {
      --rank;
    }
  }
  if (rank == 0) {
    return false;
  }

  *D = v.tail(rank);
  *U = U->rightCols(rank);
  return true;
}

// Returns whether there are non-trivial bounds constraints for any parameter
// blocks in a set.
bool HasBoundsConstraintsForAnyParameterBlocks(
    const Problem& problem, const set<double*>& parameter_blocks) {
  map<double*, ParameterBlock*> parameter_map;
  for (double* pb : parameter_blocks) {
    const int size = problem.ParameterBlockSize(pb);
    for (int i = 0; i < size; i++) {
      if (problem.GetParameterLowerBound(pb, i) >
          -std::numeric_limits<double>::max()) {
        return true;
      }
      if (problem.GetParameterUpperBound(pb, i) <
          std::numeric_limits<double>::max()) {
        return true;
      }
    }
  }
  return false;
}

// Build a problem consisting of the parameter blocks to be marginalized,
// their Markov blanket, and error terms involving the parameter blocks to
// marginalize. The lifetime of external_problem must be longer than that of the
// returned problem.
std::unique_ptr<ProblemImpl> BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize) {
  // Input validated previously. The external problem contains all parameter
  // blocks to marginalize.

  Problem::Options options;
  options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.manifold_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.enable_fast_removal = true;
  set<ResidualBlockId> marginalized_blocks_residual_ids;

  auto new_problem = std::make_unique<ProblemImpl>(options);

  for (double* parameter_block : parameter_blocks_to_marginalize) {
    vector<ResidualBlockId> residual_blocks;
    external_problem.GetResidualBlocksForParameterBlock(parameter_block,
                                                        &residual_blocks);

    for (const ResidualBlockId& residual_block_id : residual_blocks) {
      // Add this residual block if we have not already.
      if (marginalized_blocks_residual_ids.count(residual_block_id)) {
        continue;
      }
      marginalized_blocks_residual_ids.insert(residual_block_id);

      vector<double*> parameter_blocks;
      external_problem.GetParameterBlocksForResidualBlock(residual_block_id,
                                                          &parameter_blocks);

      // Add parameter blocks for this residual block.
      for (double* parameter_block : parameter_blocks) {
        if (new_problem->HasParameterBlock(parameter_block)) {
          continue;
        }

        CHECK(!external_problem.IsParameterBlockConstant(parameter_block))
            << "Constant parameter blocks cannot be marginalized";

        const int size = external_problem.ParameterBlockSize(parameter_block);
        const Manifold* manifold =
            external_problem.GetManifold(parameter_block);

        new_problem->AddParameterBlock(
            parameter_block, size, const_cast<Manifold*>(manifold));
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
    Vector* gradient) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(gradient);
  CHECK_NOTNULL(jtj);

  gradient->resize(problem->program().NumEffectiveParameters());
  Vector state_vector(problem->NumParameters());

  std::string error;
  Evaluator::Options evaluator_options;
  // Use DENSE_NORMAL_CHOLESKY, which uses the DenseJacobianWriter required for
  // the cast to DenseSparseMatrix below.
  evaluator_options.linear_solver_type = DENSE_NORMAL_CHOLESKY;
  evaluator_options.num_eliminate_blocks = num_elimination_blocks;
  evaluator_options.context = problem->context();

  std::unique_ptr<Evaluator> evaluator(
      Evaluator::Create(evaluator_options, problem->mutable_program(), &error));
  CHECK(evaluator.get() != nullptr) << "Failed creating evaluator";

  std::unique_ptr<SparseMatrix> sparse_jacobian(evaluator->CreateJacobian());

  problem->program().ParameterBlocksToStateVector(state_vector.data());

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

    Vector residuals(problem->NumResiduals());

    // Compute the residuals, restoring the state vector.
    CHECK(evaluator->Evaluate(
        state_vector.data(), &cost, residuals.data(), nullptr, nullptr));

    // Compute the gradient.
    gradient->setZero();
    sparse_jacobian->LeftMultiply(residuals.data(), gradient->data());
  } else {
    CHECK(evaluator->Evaluate(state_vector.data(),
                              &cost,
                              nullptr,
                              gradient->data(),
                              sparse_jacobian.get()));
  }

  // This cast is valid if DenseJacobianWriter is used, ensured by the linear
  // solver choice DENSE_NORMAL_CHOLESKY.
  const Matrix& jac =
      static_cast<const DenseSparseMatrix*>(sparse_jacobian.get())->matrix();
  *jtj = jac.transpose() * jac;
}

// Compute the Schur complement of the first block of the system jtj, gradient.
void SchurComplement(const Matrix& jtj,
                     const Vector& gradient,
                     const int tan_size_marginalized,
                     Matrix* jtj_marginal,
                     Vector* gradient_marginal) {
  CHECK_NOTNULL(jtj_marginal);
  CHECK_NOTNULL(gradient_marginal);
  CHECK(tan_size_marginalized < gradient.size());
  CHECK(jtj.rows() == gradient.size() && jtj.rows() == jtj.cols());
  const int tan_size_blanket = gradient.size() - tan_size_marginalized;
  const Vector gradient1 = gradient.head(tan_size_marginalized);
  const Vector gradient2 = gradient.tail(tan_size_blanket);
  const auto jtj_22 = jtj.bottomRightCorner(tan_size_blanket, tan_size_blanket);
  const auto jtj_12 =
      jtj.topRightCorner(tan_size_marginalized, tan_size_blanket);
  const Matrix jtj_11 =
      jtj.topLeftCorner(tan_size_marginalized, tan_size_marginalized);
  const Matrix jtj_11_inv =
      InvertPSDMatrix<Eigen::Dynamic>(/*assume_full_rank=*/false, jtj_11);
  *jtj_marginal = jtj_22 - jtj_12.transpose() * jtj_11_inv * jtj_12;
  *gradient_marginal = gradient2 - jtj_12.transpose() * jtj_11_inv * gradient1;
}

// Compute the parameters of cost function that will be associated with the
// marginalization prior. If successful, return the marginalization prior
// parameters marginalization_prior_A and marginalization_prior_b and pointers
// to the blocks in the Markov blanket of the parameter blocks to marginalize.
// Return value indicates whether the operation was successful or not.
bool ComputeMarginalizationPriorData(
    const MarginalizationOptions& options,
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    Matrix* marginalization_prior_A,
    Vector* marginalization_prior_b,
    vector<double*>* blanket_parameter_blocks_state) {
  CHECK_NOTNULL(marginalization_prior_A);
  CHECK_NOTNULL(marginalization_prior_b);
  CHECK_NOTNULL(blanket_parameter_blocks_state);

  // Marginalizing a block analytically minimizes it from the problem by
  // neglecting bounds constraints. In generla, there is no way to guarantee
  // that the bounds constraints are not violated. The user should remove
  // bounds constraints on variables before marginalizing them out.
  if (HasBoundsConstraintsForAnyParameterBlocks(
          external_problem, parameter_blocks_to_marginalize)) {
    return false;
  }

  std::unique_ptr<ProblemImpl> local_problem =
      BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
          external_problem, parameter_blocks_to_marginalize);

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

  const vector<ParameterBlock*>& ordered_parameter_blocks =
      local_problem->program().parameter_blocks();
  const auto blanket_parameter_blocks =
      vector<ParameterBlock*>(ordered_parameter_blocks.end() -
                                  ordering.GroupSize(kMarkovBlanketGroupId),
                              ordered_parameter_blocks.end());
  const auto ordered_parameter_blocks_marginalized =
      vector<ParameterBlock*>(ordered_parameter_blocks.begin(),
                              ordered_parameter_blocks.begin() +
                                  ordering.GroupSize(kMarginalizedGroupId));

  // Now we can validate that if linearization states are valid, all required
  // states have been provided.
  if (parameter_block_linearization_states) {
    for (ParameterBlock* pb : ordered_parameter_blocks) {
      if (parameter_block_linearization_states->find(
              pb->mutable_user_state()) ==
          parameter_block_linearization_states->end()) {
        return false;
      }
    }
  }

  if (SumTangentSize(blanket_parameter_blocks) == 0) {
    return false;
  }

  const int tan_size_marginalized =
      SumTangentSize(ordered_parameter_blocks_marginalized);

  if (tan_size_marginalized == 0) {
    return false;
  }

  Matrix jtj;
  Vector gradient;
  EvaluateProblem(parameter_blocks_to_marginalize.size(),
                  parameter_block_linearization_states,
                  local_problem.get(),
                  &jtj,
                  &gradient);

  Matrix jtj_marginal;
  Vector gradient_marginalization_prior;
  SchurComplement(jtj,
                  gradient,
                  tan_size_marginalized,
                  &jtj_marginal,
                  &gradient_marginalization_prior);

  blanket_parameter_blocks_state->resize(blanket_parameter_blocks.size());
  for (int i = 0; i < blanket_parameter_blocks.size(); ++i) {
    (*blanket_parameter_blocks_state)[i] =
        blanket_parameter_blocks[i]->mutable_user_state();
  }

  if (options.assume_full_rank) {
    const Eigen::LDLT<Matrix> ldlt = jtj_marginal.ldlt();
    if (ldlt.info() != Eigen::Success) {
      return false;
    }
    Matrix sqrt_factor;
    if (!CholeskyFactor(ldlt, &sqrt_factor)) {
      return false;
    }
    *marginalization_prior_A = sqrt_factor.transpose();
    *marginalization_prior_b =
        *marginalization_prior_A * ldlt.solve(gradient_marginalization_prior);
  } else {
    Matrix U;
    Vector D;
    // Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
    // node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
    // (2014): 1371-1385.
    if (!GetEigendecomposition(jtj_marginal, &U, &D)) {
      // Rank 0, unexpected.
      return false;
    }
    const Vector sqrtD = D.array().sqrt();
    *marginalization_prior_A = sqrtD.matrix().asDiagonal() * U.transpose();
    *marginalization_prior_b = sqrtD.cwiseInverse().matrix().asDiagonal() *
                               U.transpose() * gradient_marginalization_prior;
  }
  return true;
}

bool ComputeMarginalizationPrior(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<double*, const double*>* parameter_block_linearization_states,
    const Problem* problem,
    vector<double*>* blanket_parameter_blocks_state,
    CostFunction** marginalization_prior_cost_function) {
  CHECK_NOTNULL(marginalization_prior_cost_function);
  CHECK_NOTNULL(blanket_parameter_blocks_state);

  Matrix marginalization_prior_A;
  Vector marginalization_prior_b;
  if (!ComputeMarginalizationPriorData(options,
                                       *problem,
                                       parameter_blocks_to_marginalize,
                                       parameter_block_linearization_states,
                                       &marginalization_prior_A,
                                       &marginalization_prior_b,
                                       blanket_parameter_blocks_state)) {
    return false;
  }

  const int num_markov_blanket_blocks = blanket_parameter_blocks_state->size();
  vector<const Manifold*> blanket_manifolds(num_markov_blanket_blocks);
  vector<Vector> blanket_parameter_block_states(num_markov_blanket_blocks);
  vector<int> blanket_tan_sizes(num_markov_blanket_blocks);
  for (int i = 0; i < num_markov_blanket_blocks; ++i) {
    const double* pb_state = (*blanket_parameter_blocks_state)[i];
    blanket_manifolds[i] = problem->GetManifold(pb_state);
    const int pb_ambient_size = problem->ParameterBlockSize(pb_state);
    blanket_tan_sizes[i] = problem->ParameterBlockTangentSize(pb_state);
    blanket_parameter_block_states[i] =
        ConstVectorRef(pb_state, pb_ambient_size);
  }

  *marginalization_prior_cost_function =
      new MarginalizationPriorCostFunction(marginalization_prior_A,
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
    vector<ResidualBlockId>* marginalization_prior_ids,
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

  if (marginalization_prior_ids) {
    *marginalization_prior_ids = {mp_id};
  }

  return true;
}
}  // anonymous namespace

}  // namespace internal

bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<double*, const double*>*
        parameter_block_linearization_states) {
  // Validate the input. parameter_block_linearization_states will be validated
  // later.
  CHECK_NOTNULL(problem);
  for (double* parameter_block : parameter_blocks_to_marginalize) {
    if (!problem->HasParameterBlock(parameter_block)) {
      return false;
    }
  }

  return internal::MarginalizeOutVariables(
      options,
      parameter_blocks_to_marginalize,
      problem,
      marginalization_prior_ids,
      parameter_block_linearization_states);
}

}  // namespace ceres
