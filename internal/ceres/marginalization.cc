// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/evaluator.h"
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

// Compute U,D such that U D U^T = P, where U is n x r, D is r x r, and
// r = rank(P).
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
  for (const double* pb : parameter_blocks) {
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

// Build a problem consisting of the parameter blocks to be marginalized, their
// Markov blanket, and error terms involving the parameter blocks to
// marginalize. The lifetime of external_problem must exceed that of the
// returned problem. If problem_pb_to_storage_map is provided, it is used to
// store copies of the parameter blocks in the new problem, use store pointers
// to the parameter blocks in the original problem in keys. Otherwise, just use
// pointers to parameter blocks in the original problem.
std::unique_ptr<ProblemImpl> BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states,
    std::map<const double*, double*>* storage_to_problem_pb_map,
    std::map<const double*, Vector>* problem_pb_to_storage_map) {
  // Input validated previously. The external problem contains all parameter
  // blocks to marginalize.
  if (parameter_block_linearization_states) {
    CHECK_NOTNULL(problem_pb_to_storage_map);
    CHECK_NOTNULL(storage_to_problem_pb_map);
    problem_pb_to_storage_map->clear();
    storage_to_problem_pb_map->clear();
  }

  Problem::Options options;
  options.cost_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.loss_function_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.local_parameterization_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.manifold_ownership = DO_NOT_TAKE_OWNERSHIP;
  options.enable_fast_removal = true;
  set<ResidualBlockId> marginalized_blocks_residual_ids;

  auto new_problem = std::make_unique<ProblemImpl>(options);

  auto maybeAddParameterBlockToNewProblem = [&](double* pb,
                                                int size) -> double* {
    double* pb_in_new_problem;
    if (problem_pb_to_storage_map) {
      // If linearization states are different from the current states, we
      // must use new storage for states in the local problem.
      auto it = problem_pb_to_storage_map->find(pb);
      if (it == problem_pb_to_storage_map->end()) {
        // Allocate new storage.
        const auto it_lin = parameter_block_linearization_states->find(pb);
        CHECK(it_lin != parameter_block_linearization_states->end())
            << "If linearization states are provided, all should be "
               "provided";

        // Set the state in the local problem to the linearization state.
        auto p = problem_pb_to_storage_map->emplace(pb, size);
        CHECK(p.second) << "Failed to insert into problem_pb_to_storage_map";
        Vector& pb_state = p.first->second;
        pb_state = ConstVectorRef(it_lin->second, size);
        pb_in_new_problem = pb_state.data();
        storage_to_problem_pb_map->emplace(pb_in_new_problem, pb);
      } else {
        // We have allocated storage for a copy of this parameter block.
        pb_in_new_problem = it->second.data();
      }
    } else {
      // Re-use storage for the parameter block.
      pb_in_new_problem = pb;
    }

    // Add the parameter block to the new problem if it has not been added
    // already.
    if (new_problem->HasParameterBlock(pb_in_new_problem)) {
      return pb_in_new_problem;
    }
    CHECK(!external_problem.GetParameterization(pb))
        << "LocalParameterizations are not supproted in marginalization! Use "
           "Manifold instead.";
    const Manifold* manifold = external_problem.GetManifold(pb);
    new_problem->AddParameterBlock(
        pb_in_new_problem, size, const_cast<Manifold*>(manifold));
    return pb_in_new_problem;
  };

  // Add the parameter blocks to marginalize to the new problem. Any that have
  // no residuals will not be added in the next step.
  for (double* pb : parameter_blocks_to_marginalize) {
    CHECK(!external_problem.IsParameterBlockConstant(pb))
        << "Constant marginalized blocks are not currently supported in "
           "marginalization";
    const int size = external_problem.ParameterBlockSize(pb);
    maybeAddParameterBlockToNewProblem(pb, size);
  }

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

      vector<double*> parameter_blocks_for_res;
      external_problem.GetParameterBlocksForResidualBlock(
          residual_block_id, &parameter_blocks_for_res);

      vector<double*> parameter_blocks_in_new_problem;
      parameter_blocks_in_new_problem.reserve(parameter_blocks_for_res.size());

      // Add any new parameter blocks for this residual block.
      for (double* pb_for_res : parameter_blocks_for_res) {
        CHECK(!external_problem.IsParameterBlockConstant(pb_for_res))
            << "Constant parameter blocks are not currently supported in "
               "marginalization";
        const int size = external_problem.ParameterBlockSize(pb_for_res);
        double*& pb_in_new_problem =
            parameter_blocks_in_new_problem.emplace_back();
        pb_in_new_problem =
            maybeAddParameterBlockToNewProblem(pb_for_res, size);
      }

      const CostFunction* cost_function =
          external_problem.GetCostFunctionForResidualBlock(residual_block_id);
      const LossFunction* loss_function =
          external_problem.GetLossFunctionForResidualBlock(residual_block_id);
      // The new problem is for evaluation. The manifold, loss function, and
      // cost function metadata will not be modified.
      new_problem->AddResidualBlock(const_cast<CostFunction*>(cost_function),
                                    const_cast<LossFunction*>(loss_function),
                                    parameter_blocks_in_new_problem.data(),
                                    parameter_blocks_in_new_problem.size());
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

void EvaluateProblem(const int num_elimination_blocks,
                     ProblemImpl* problem,
                     Matrix* jtj,
                     Vector* gradient) {
  CHECK_NOTNULL(problem);
  CHECK_NOTNULL(gradient);
  CHECK_NOTNULL(jtj);

  gradient->resize(problem->program().NumEffectiveParameters());

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

  Vector state_vector(problem->NumParameters());
  problem->program().ParameterBlocksToStateVector(state_vector.data());

  double cost;
  std::unique_ptr<SparseMatrix> sparse_jacobian(evaluator->CreateJacobian());
  CHECK(evaluator->Evaluate(state_vector.data(),
                            &cost,
                            nullptr,
                            gradient->data(),
                            sparse_jacobian.get()));

  // This cast is valid if DenseJacobianWriter is used, ensured by the linear
  // solver choice DENSE_NORMAL_CHOLESKY.
  const Matrix& jac =
      static_cast<const DenseSparseMatrix*>(sparse_jacobian.get())->matrix();
  *jtj = jac.transpose() * jac;
}

// Compute the Schur complement of the first block of the system jtj, gradient.
void SchurComplement(const Matrix& jtj,
                     const Vector& gradient,
                     bool assume_jtj11_full_rank,
                     int tan_size_marginalized,
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
  // The information block corresponding to the states to marginalize is not
  // assumed to be full rank, so a pseudoinverse is used.
  const Matrix jtj_11_pinv =
      InvertPSDMatrix<Eigen::Dynamic>(assume_jtj11_full_rank, jtj_11);
  *jtj_marginal = jtj_22 - jtj_12.transpose() * jtj_11_pinv * jtj_12;
  *gradient_marginal = gradient2 - jtj_12.transpose() * jtj_11_pinv * gradient1;
}

// Compute the parameters of cost function that will be associated with the
// marginalization prior. If successful, return the marginalization prior
// parameters marginalization_prior_A and marginalization_prior_b and pointers
// to the blocks in the Markov blanket of the parameter blocks to marginalize
// in external_problem. Return value indicates whether the operation was
// successful or not.
CostFunction* ComputeMarginalizationPrior(
    const MarginalizationOptions& options,
    const Problem& external_problem,
    const set<double*>& parameter_blocks_to_marginalize,
    const map<const double*, const double*>*
        parameter_block_linearization_states,
    vector<double*>* blanket_parameter_blocks_problem_states) {
  CHECK_NOTNULL(blanket_parameter_blocks_problem_states);

  // Marginalizing a block analytically minimizes it from the problem by
  // neglecting bounds constraints. In general, there is no way to guarantee
  // that the bounds constraints are not violated. The user should remove
  // bounds constraints on variables before marginalizing them out.
  if (HasBoundsConstraintsForAnyParameterBlocks(
          external_problem, parameter_blocks_to_marginalize)) {
    return nullptr;
  }

  const bool copy_parameter_blocks =
      parameter_block_linearization_states != nullptr;

  // If linearization states are used, keep copies of the parameter blocks in
  // a map from the parameter block in the original problem to the copy.
  std::map<const double*, Vector> problem_pb_to_storage_map;
  // If linearization states are used, we also need the inverse map.
  std::map<const double*, double*> storage_to_problem_pb_map;
  std::unique_ptr<ProblemImpl> local_problem =
      BuildProblemWithMarginalizedBlocksAndMarkovBlanket(
          external_problem,
          parameter_blocks_to_marginalize,
          parameter_block_linearization_states,
          copy_parameter_blocks ? &storage_to_problem_pb_map : nullptr,
          copy_parameter_blocks ? &problem_pb_to_storage_map : nullptr);

  set<double*> parameter_blocks_to_marginalize_in_new_prob;
  const set<double*>* marginalized_parameter_blocks_in_loc_prob_ptr = nullptr;
  if (copy_parameter_blocks) {
    for (double* pb : parameter_blocks_to_marginalize) {
      Vector& pb_storage = problem_pb_to_storage_map.at(pb);
      parameter_blocks_to_marginalize_in_new_prob.insert(pb_storage.data());
    }
    marginalized_parameter_blocks_in_loc_prob_ptr =
        &parameter_blocks_to_marginalize_in_new_prob;
  } else {
    marginalized_parameter_blocks_in_loc_prob_ptr =
        &parameter_blocks_to_marginalize;
  }
  const ParameterBlockOrdering ordering =
      GetOrderingForMarginalizedBlocksAndMarkovBlanket(
          *local_problem, *marginalized_parameter_blocks_in_loc_prob_ptr);
  std::string error;
  CHECK(ApplyOrdering(local_problem->parameter_map(),
                      ordering,
                      local_problem->mutable_program(),
                      &error))
      << "Failed to apply ordering required for marginalization";

  local_problem->mutable_program()->SetParameterOffsetsAndIndex();

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

  if (SumTangentSize(blanket_parameter_blocks) == 0) {
    return nullptr;
  }
  const int num_marginalized_blocks =
      marginalized_parameter_blocks_in_loc_prob_ptr->size();
  const int num_blanket_blocks = blanket_parameter_blocks.size();
  const int tan_size_marginalized =
      SumTangentSize(ordered_parameter_blocks_marginalized);

  if (tan_size_marginalized == 0) {
    return nullptr;
  }

  Matrix jtj;
  Vector gradient;
  EvaluateProblem(
      num_marginalized_blocks, local_problem.get(), &jtj, &gradient);

  Matrix jtj_marginal;
  Vector gradient_marginalization_prior;
  SchurComplement(jtj,
                  gradient,
                  options.assume_marginalized_block_is_full_rank,
                  tan_size_marginalized,
                  &jtj_marginal,
                  &gradient_marginalization_prior);

  // Get tangent space sizes, states, and manifolds required for the
  // marginalization prior.
  blanket_parameter_blocks_problem_states->resize(
      num_blanket_blocks);  // in the original problem
  vector<Vector> blanket_reference_points(
      num_blanket_blocks);  // in the local problem
  vector<const Manifold*> blanket_manifolds(num_blanket_blocks);
  vector<int> blanket_tan_sizes(num_blanket_blocks);
  for (int i = 0; i < num_blanket_blocks; ++i) {
    const int size = blanket_parameter_blocks[i]->Size();
    double* pb_in_new_problem =
        blanket_parameter_blocks[i]->mutable_user_state();
    double* pb_in_ext_problem =
        copy_parameter_blocks ? storage_to_problem_pb_map.at(pb_in_new_problem)
                              : pb_in_new_problem;
    (*blanket_parameter_blocks_problem_states)[i] = pb_in_ext_problem;
    blanket_reference_points[i] = ConstVectorRef(pb_in_new_problem, size);
    blanket_manifolds[i] = external_problem.GetManifold(pb_in_ext_problem);
    blanket_tan_sizes[i] =
        external_problem.ParameterBlockTangentSize(pb_in_ext_problem);
  }

  Matrix marginalization_prior_A;
  Vector marginalization_prior_b;
  if (options.assume_marginal_information_is_full_rank) {
    const Eigen::LLT<Matrix> llt = jtj_marginal.llt();
    if (llt.info() != Eigen::Success) {
      return nullptr;
    }
    const Matrix matL = llt.matrixL();
    marginalization_prior_A = matL.transpose();
    marginalization_prior_b = matL.inverse() * gradient_marginalization_prior;
  } else {
    Matrix U;
    Vector D;
    // Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
    // node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
    // (2014): 1371-1385.
    if (!GetEigendecomposition(jtj_marginal, &U, &D)) {
      // Rank 0, unexpected.
      return nullptr;
    }
    const Vector sqrtD = D.array().sqrt();
    marginalization_prior_A = sqrtD.matrix().asDiagonal() * U.transpose();
    marginalization_prior_b = sqrtD.cwiseInverse().matrix().asDiagonal() *
                              U.transpose() * gradient_marginalization_prior;
  }

  return new MarginalizationPriorCostFunction(marginalization_prior_A,
                                              marginalization_prior_b,
                                              blanket_reference_points,
                                              blanket_tan_sizes,
                                              blanket_manifolds);
}

bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    vector<ResidualBlockId>* marginalization_prior_ids,
    const map<const double*, const double*>*
        parameter_block_linearization_states) {
  vector<double*> blanket_ordered_parameter_blocks;
  CostFunction* new_cost_function =
      ComputeMarginalizationPrior(options,
                                  *problem,
                                  parameter_blocks_to_marginalize,
                                  parameter_block_linearization_states,
                                  &blanket_ordered_parameter_blocks);
  if (!new_cost_function) {
    return false;
  }

  // Remove marginalized blocks.
  for (double* pb : parameter_blocks_to_marginalize) {
    problem->RemoveParameterBlock(pb);
  }

  // Add the cost function for the marginalization prior to the problem.
  const ResidualBlockId mp_id =
      problem->AddResidualBlock(new_cost_function,
                                nullptr,
                                blanket_ordered_parameter_blocks.data(),
                                blanket_ordered_parameter_blocks.size());

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
    const std::map<const double*, const double*>*
        parameter_block_linearization_states) {
  // Validate the input. parameter_block_linearization_states will be validated
  // later.
  CHECK_NOTNULL(problem);
  for (double* parameter_block : parameter_blocks_to_marginalize) {
    CHECK(problem->HasParameterBlock(parameter_block))
        << "Parameter block to marginalize is not in the problem. Did you "
           "forget to add it?";
  }

  return internal::MarginalizeOutVariables(
      options,
      parameter_blocks_to_marginalize,
      problem,
      marginalization_prior_ids,
      parameter_block_linearization_states);
}

}  // namespace ceres
