#include "ceres/marginalization.h"

#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/loss_function.h"
#include "ceres/manifold.h"
#include "ceres/manifold_test_utils.h"
#include "ceres/map_util.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/program.h"
#include "ceres/random.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "gtest/gtest.h"
#include "marginalization_prior_cost_function.h"

namespace ceres {
namespace internal {
using std::vector;

namespace {
// residual = 2/3 * p0 ^ 1.5 - 0.5 * p1^2
class NonlinearBinaryCostFunction : public SizedCostFunction<1, 1, 1> {
 public:
  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    const double p0 = parameters[0][0];
    const double p1 = parameters[1][0];
    residuals[0] = 2.0 / 3.0 * pow(p0, 1.5) - 0.5 * pow(p1, 2.0);
    if (jacobians) {
      if (jacobians[0]) {
        jacobians[0][0] = sqrt(p0);
      }
      if (jacobians[1]) {
        jacobians[1][0] = -p1;
      }
    }
    return true;
  }
};

class LinearCostFunction : public CostFunction {
 public:
  LinearCostFunction(vector<Matrix> a, const Vector& b)
      : a_(std::move(a)), b_(b) {
    set_num_residuals(b_.size());
    for (size_t i = 0; i < a_.size(); ++i) {
      mutable_parameter_block_sizes()->push_back(a_[i].cols());
      CHECK(a_[i].rows() == b_.size()) << "Dimensions mismatch";
    }
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    VectorRef res(residuals, b_.rows());
    res = b_;
    for (size_t i = 0; i < a_.size(); ++i) {
      ConstVectorRef pi(parameters[i], parameter_block_sizes()[i]);
      res += a_[i] * pi;
    }

    if (jacobians) {
      for (size_t i = 0; i < a_.size(); ++i) {
        if (jacobians[i]) {
          MatrixRef(jacobians[i], a_[i].rows(), a_[i].cols()) = a_[i];
        }
      }
    }
    return true;
  }

 private:
  const vector<Matrix> a_;
  const Vector b_;
};

LinearCostFunction* MakeRandomLinearCostFunction(
    int num_residuals, const vector<int>& parameter_block_sizes) {
  vector<Matrix> a;
  a.reserve(parameter_block_sizes.size());
  for (int size : parameter_block_sizes) {
    a.push_back(Matrix::Random(num_residuals, size));
  }
  return new LinearCostFunction(std::move(a), Vector::Random(num_residuals));
}

void TestLinearizationState(const MarginalizationOptions& options) {
  // States
  const vector<double> final_states = {9.0, 2.0};
  vector<double> states = final_states;
  // States for Jacobian.
  const vector<double> linearization_states = {12.0, 4.0};
  /*
   * This toy problem consists of the residuals
   * r1(x0, x1) = 2/3 * x0^1.5 - 0.5 * x1^2
   * r2(x0) = 2.0 * x0 - 2.0
   *
   * At the linearization state (12,4),
   *  J     = [ 12^0.5  -4.0 ]
   *          [ 2.0      0.0 ]
   *
   * One can show that marginalizing out x yields the prior
   *
   * prior(x1) = 2 * (x1 - 2.0) + 8 * (sqrt(3) - 1),
   */
  Problem problem;
  problem.AddParameterBlock(&states[0], 1);
  problem.AddParameterBlock(&states[1], 1);
  problem.SetManifold(&states[0], new EuclideanManifold<1>());
  problem.SetManifold(&states[1], new EuclideanManifold<1>());
  problem.AddResidualBlock(
      new NonlinearBinaryCostFunction(), nullptr, &states[0], &states[1]);
  problem.AddResidualBlock(
      new LinearCostFunction({2.0 * Eigen::Matrix<double, 1, 1>::Identity()},
                             Eigen::Matrix<double, 1, 1>(-2.0)),
      nullptr,
      &states[0]);
  const std::map<double*, const double*> parameter_block_linearization_states =
      {{&states[0], &linearization_states[0]},
       {&states[1], &linearization_states[1]}};
  vector<ResidualBlockId> marginalization_prior_ids;
  EXPECT_TRUE(MarginalizeOutVariables(
      options,
      std::set<double*>{&states[0]},  // Marginalize the first state
      &problem,
      &marginalization_prior_ids,
      &parameter_block_linearization_states));
  const auto* marginalization_prior =
      static_cast<const MarginalizationPriorCostFunction*>(
          problem.GetCostFunctionForResidualBlock(
              marginalization_prior_ids.front()));
  EXPECT_TRUE(marginalization_prior);
  const vector<Matrix> J = marginalization_prior->GetJacobianWrtIncrement();
  const Vector b = marginalization_prior->GetB();
  const double b_expected = 8.0 * (sqrt(3.0) - 1.0);
  const double J_expected = 2.0;
  EXPECT_EQ(states[0], final_states[0]);
  EXPECT_EQ(states[1], final_states[1]);
  EXPECT_NEAR(b[0], b_expected, 1e-9);
  EXPECT_NEAR(J[0](0, 0), J_expected, 1e-9);
}
TEST(Marginalization, LinearizationState) {
  MarginalizationOptions options;
  options.assume_full_rank = false;
  ASSERT_NO_FATAL_FAILURE(TestLinearizationState(options));
  options.assume_full_rank = true;
  ASSERT_NO_FATAL_FAILURE(TestLinearizationState(options));
}
class TestGraphState {
  // Class that operates on the graph
  //
  //  +--x0-----x1----prior
  //  |  |      |
  //  |  |      |
  //  |  x2     x3
  //  |   \    /
  //  |    \  /
  //  |     \/
  //  +-----x4
  //
  // Mixed DOF parameter blocks.
 public:
  static constexpr int kNumBlocks = 5;
  TestGraphState() {
    // Calculate block offsets
    cum_parameter_sizes_.resize(parameter_sizes_.size(), 0);
    std::partial_sum(parameter_sizes_.begin(),
                     parameter_sizes_.end() - 1,
                     cum_parameter_sizes_.begin() + 1,
                     std::plus<int>());

    // Get block pointers.
    for (int k = 0; k < kNumBlocks; ++k) {
      ordered_parameter_blocks_.push_back(
          &state_vector_(cum_parameter_sizes_[k]));
    }

    // Add parameters.
    for (int k = 0; k < kNumBlocks; ++k) {
      int block_size = parameter_sizes_[k];
      problem_.AddParameterBlock(ordered_parameter_blocks_[k], block_size);
      Manifold* manifold = nullptr;
      if (k <= 1) {
        manifold = static_cast<Manifold*>(new SubsetManifold(block_size, {1}));
      } else if (k <= 3) {
        manifold = static_cast<Manifold*>(
            new EuclideanManifold<ceres::DYNAMIC>(block_size));
      }  // else nullptr => Euclidean manifold
      problem_.SetManifold(ordered_parameter_blocks_[k], manifold);
    }

    // Add residuals.
    problem_.AddResidualBlock(
        new LinearCostFunction({Matrix::Random(4, 2)}, {Vector::Random(4)}),
        nullptr,
        ordered_parameter_blocks_[1]);
    vector<std::pair<int, int>> edges = {
        {0, 1}, {2, 0}, {3, 1}, {2, 4}, {3, 4}, {0, 4}};
    int residuals_for_edge = 3;
    for (const auto e : edges) {
      const int i = e.first;
      const int j = e.second;
      const std::vector<int> sizes = {parameter_sizes_[i], parameter_sizes_[j]};
      problem_.AddResidualBlock(
          MakeRandomLinearCostFunction(residuals_for_edge++, sizes),
          nullptr,
          ordered_parameter_blocks_[i],
          ordered_parameter_blocks_[j]);
    }

    // Get tangent space size information.
    for (int k = 0; k < ordered_parameter_blocks_.size(); ++k) {
      parameter_tan_sizes_.push_back(
          problem_.ParameterBlockTangentSize(ordered_parameter_blocks_[k]));
    }
    cum_parameter_tan_size_.resize(parameter_tan_sizes_.size() + 1);
    cum_parameter_tan_size_[0] = 0;
    std::partial_sum(parameter_tan_sizes_.begin(),
                     parameter_tan_sizes_.end(),
                     cum_parameter_tan_size_.begin() + 1,
                     std::plus<int>());
  }
  void Perturb() {
    vector<double*> parameter_blocks;
    problem_.GetParameterBlocks(&parameter_blocks);
    for (int b = 0; b < parameter_blocks.size(); ++b) {
      double* pb = parameter_blocks[b];
      const int tan_size = problem_.ParameterBlockTangentSize(pb);
      const int size = problem_.ParameterBlockSize(pb);
      const Vector tan_perturbation = Vector::Random(tan_size);
      // Apply perturbation to this parameter block.
      const Manifold* manifold = problem_.GetManifold(pb);
      Vector pb_perturbed(size);
      if (manifold) {
        manifold->Plus(pb, tan_perturbation.data(), pb_perturbed.data());
      } else {
        pb_perturbed = VectorRef(pb, size) + tan_perturbation;
      }
      VectorRef(pb, size) = pb_perturbed;
    }
  }
  Vector GetStateVector(const std::bitset<kNumBlocks> selection) const {
    Vector res(state_vector_.size());
    int offset = 0;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        for (int j = 0; j < parameter_sizes_[i]; j++, offset++) {
          res(offset) = state_vector_(cum_parameter_sizes_[i] + j);
        }
      }
    }
    return res.head(offset);
  }
  void SolveProblem() {
    Solver::Options options;
    options.max_num_iterations = 1;
    options.max_lm_diagonal = options.min_lm_diagonal;
    Solver::Summary summary;
    Solve(options, &problem_, &summary);
  }
  void MarginalizeOutVariableSubset(
      const std::bitset<kNumBlocks> parameter_blocks_to_marginalize_mask) {
    const std::set<double*> parameter_blocks_to_marginalize =
        GetParameterBlockSubset(parameter_blocks_to_marginalize_mask);
    EXPECT_TRUE(MarginalizeOutVariables(
        options_, parameter_blocks_to_marginalize, &problem_, nullptr));
  }
  std::set<double*> GetParameterBlockSubset(
      const std::bitset<kNumBlocks> selection) {
    std::set<double*> subset;
    for (int i = 0; i < ordered_parameter_blocks_.size(); i++) {
      if (selection.test(i)) {
        subset.insert(ordered_parameter_blocks_[i]);
      }
    }
    return subset;
  }
  int GetProblemTangentSize(
      const vector<const double*>& problem_parameter_blocks) {
    return std::accumulate(problem_parameter_blocks.begin(),
                           problem_parameter_blocks.end(),
                           0,
                           [this](int sz, const double* pb) {
                             return sz + problem_.ParameterBlockTangentSize(pb);
                           });
  }
  Matrix GetCovarianceMatrixInTangentSpace(
      vector<const double*>& covariance_blocks) {
    Covariance::Options options;
    Covariance covariance(options);
    const int tan_size = GetProblemTangentSize(covariance_blocks);
    Matrix covariance_matrix(tan_size, tan_size);
    CHECK(covariance.Compute(covariance_blocks, &problem_));
    CHECK(covariance.GetCovarianceMatrixInTangentSpace(
        covariance_blocks, covariance_matrix.data()));
    return covariance_matrix;
  }
  Matrix GetCovarianceMatrixInTangentSpace() {
    vector<double*> problem_parameter_blocks;
    problem_.GetParameterBlocks(&problem_parameter_blocks);
    vector<const double*> covariance_blocks;
    for (const double* pb : problem_parameter_blocks) {
      covariance_blocks.push_back(pb);
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }
  Matrix GetCovarianceMatrixInTangentSpace(
      const std::bitset<kNumBlocks> selection) {
    vector<const double*> covariance_blocks;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        covariance_blocks.push_back(ordered_parameter_blocks_[i]);
      }
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

 private:
  const std::array<int, 8> parameter_sizes_ = {2, 2, 1, 1, 2};
  Vector state_vector_ = Vector::Random(8);
  vector<double*> ordered_parameter_blocks_;
  vector<int> cum_parameter_sizes_;
  vector<int> parameter_tan_sizes_;
  vector<int> cum_parameter_tan_size_;
  Problem problem_;
  MarginalizationOptions options_;
};
void TestMarginalization(
    std::bitset<TestGraphState::kNumBlocks> is_marginalized) {
  TestGraphState state;
  const Matrix marginal_covariance_expected =
      state.GetCovarianceMatrixInTangentSpace(~is_marginalized);
  state.SolveProblem();
  const Vector state_after_first_solve = state.GetStateVector(~is_marginalized);
  state.Perturb();
  state.SolveProblem();
  state.Perturb();
  state.MarginalizeOutVariableSubset(is_marginalized);
  const Matrix marginal_covariance_actual =
      state.GetCovarianceMatrixInTangentSpace();
  const double cov_error =
      (marginal_covariance_expected - marginal_covariance_actual).norm();
  EXPECT_LT(cov_error, 1e-6);
  // Solve the new problem to compute the marginal mean.
  state.Perturb();
  state.SolveProblem();
  const Vector state_after_marginalization =
      state.GetStateVector(~is_marginalized);
  const double state_error =
      (state_after_marginalization - state_after_first_solve)
          .lpNorm<Eigen::Infinity>();
  EXPECT_LT(state_error, 1e-6);
}
}  //  anonymous namespace

TEST(Marginalization, MarginalizationSuccess) {
  /*
   * Construct a linear least squares problem with 5 variables. Compute the
   * mean and covariance of the likelihood. Marginalize out a subset of
   * variables, producing a smaller problem. Verify that the marginal
   * mean and covariance computed from this problem matches the corresponding
   * entries in the joint mean and covariance computed previously. Perform this
   * test for all subsets of the variables.
   */
  srand(5);
  for (unsigned int m = 1; m < (1u << TestGraphState::kNumBlocks) - 1; m++) {
    ASSERT_NO_FATAL_FAILURE(TestMarginalization(m));
  }
}

TEST(Marginalization, MarginalizationFailureNoMarkovBlanket) {
  // No markov blanket
  double x[2] = {0, 0};
  Problem problem;
  problem.AddParameterBlock(x, 2);
  problem.AddResidualBlock(
      new LinearCostFunction({Eigen::Matrix<double, 2, 2>::Identity()},
                             Eigen::Vector2d(1.0, 2.0)),
      nullptr,
      x);
  const std::set<double*> to_marginalize = {x};
  MarginalizationOptions options;
  EXPECT_FALSE(
      MarginalizeOutVariables(options, to_marginalize, &problem, nullptr));
}

TEST(Marginalization, MarginalizationFailureIncorrectlyAssumeFullRank) {
  // Assumed full rank, but system is not full rank.
  double x[2] = {1, 2};
  double y[2] = {3, 4};
  Problem problem;
  problem.AddParameterBlock(x, 2);
  problem.AddParameterBlock(y, 2);
  std::vector<Matrix> jacobians{Matrix::Zero(3, 2), Matrix::Zero(3, 2)};
  problem.AddResidualBlock(
      new LinearCostFunction(jacobians, Vector::Random(3)), nullptr, x, y);
  const std::set<double*> to_marginalize = {x};
  MarginalizationOptions options;

  options.assume_full_rank = true;
  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr));
}

TEST(Marginalization, MarginalizationFailureRankZero) {
  // Do not assume full rank, but rank 0.
  double x[2] = {1, 2};
  double y[2] = {3, 4};
  Problem problem;
  problem.AddParameterBlock(x, 2);
  problem.AddParameterBlock(y, 2);
  problem.AddResidualBlock(
      new LinearCostFunction({Matrix::Zero(3, 2), Matrix::Zero(3, 2)},
                             Vector::Random(3)),
      nullptr,
      x,
      y);
  const std::set<double*> to_marginalize = {x};
  MarginalizationOptions options;

  options.assume_full_rank = false;
  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr));
}

TEST(Marginalization, MarginalizationFailureMarkovBlanketIsInfeasible) {
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  // Bounds constraints are allowed only for the Markov blanket variables.
  problem.SetParameterUpperBound(&y, 0, 3.0);
  problem.SetParameterLowerBound(&y, 0, 1.0);
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), nullptr, &x, &y);
  const std::set<double*> to_marginalize = {&x};
  MarginalizationOptions options;
  // Markov blanket is infeasible.
  y = 4.0;
  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr));
}

TEST(Marginalization, MarginalizationFailureBlockNotInProblem) {
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  const std::set<double*> to_marginalize = {&y};
  MarginalizationOptions options;
  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr));
}

TEST(Marginalization,
     MarginalizationFailureOnlySomeLinearizationStatesProvided) {
  // If linearization states are provided, they must include all variables.
  double x = 4.0;
  double x_linearization_state = 4.2;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), nullptr, &x, &y);
  const std::set<double*> to_marginalize = {&x};

  MarginalizationOptions options;
  const std::map<double*, const double*> parameter_block_linearization_states =
      {{&x, &x_linearization_state}};

  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr,
                              &parameter_block_linearization_states));
}

TEST(Marginalization,
     MarginalizationFailureVariableToMarginalizeHasBoundsConstraints) {
  double x = 4.0;
  double y = 2.0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);
  problem.AddParameterBlock(&y, 1);
  problem.SetParameterUpperBound(&x, 0, 3.0);
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), nullptr, &x, &y);
  const std::set<double*> to_marginalize = {&x};
  MarginalizationOptions options;
  // Variable to marginalize is infeasible.
  EXPECT_FALSE(
      MarginalizeOutVariables(options,
                              to_marginalize,
                              &problem,
                              /* marginalization_prior_ids = */ nullptr));
}

TEST(LinearCostFunction, JacobianTest) {
  const int num_residuals = 4;
  const vector<int> parameter_block_sizes = {1, 2, 3};

  const int state_dim = std::accumulate(
      parameter_block_sizes.begin(), parameter_block_sizes.end(), 0);
  const int num_parameter_blocks = parameter_block_sizes.size();

  vector<Matrix> jacobians_expected;
  jacobians_expected.reserve(parameter_block_sizes.size());
  for (int size : parameter_block_sizes) {
    jacobians_expected.push_back(Matrix::Random(num_residuals, size));
  }

  Vector b = Vector::Random(num_residuals);
  Vector x = Vector::Random(state_dim);
  const vector<const double*> parameters = {&x(0), &x(1), &x(3)};

  Vector residual_expected = b;
  for (size_t i = 0; i < parameter_block_sizes.size(); ++i) {
    residual_expected +=
        jacobians_expected[i] *
        ConstVectorRef(parameters[i], parameter_block_sizes[i]);
  }

  LinearCostFunction linear_cost_function(jacobians_expected, b);
  Vector residual_actual(num_residuals);

  vector<Matrix> jacobians(num_parameter_blocks);
  double* jacobian_ptrs[num_parameter_blocks];

  for (int i = 0; i < num_parameter_blocks; i++) {
    jacobians[i].resize(num_residuals, parameter_block_sizes[i]);
    jacobian_ptrs[i] = jacobians[i].data();
  }

  linear_cost_function.Evaluate(
      parameters.data(), residual_actual.data(), jacobian_ptrs);

  for (size_t i = 0; i < parameters.size(); ++i) {
    EXPECT_DOUBLE_EQ(
        (jacobians[i] - jacobians_expected[i]).lpNorm<Eigen::Infinity>(), 0.0);
  }
  EXPECT_DOUBLE_EQ(
      (residual_actual - residual_expected).lpNorm<Eigen::Infinity>(), 0.0);

  EXPECT_EQ(linear_cost_function.num_residuals(), num_residuals);
  EXPECT_EQ(linear_cost_function.parameter_block_sizes(),
            parameter_block_sizes);
}
}  // namespace internal
}  // namespace ceres
