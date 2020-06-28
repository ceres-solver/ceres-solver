// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"
#include "ceres/marginalizable_parameterization.h"
#include "ceres/loss_function.h"
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
#include "linear_cost_function.h"
#include "marginalization_prior_cost_function.h"

namespace ceres {
namespace internal {

using std::bitset;
using std::set;
using std::vector;

// residual = 2/3 * p0 ^ 1.5 - 0.5 * p1^2
class NonlinearBinaryCostFunction : public SizedCostFunction<1, 1, 1> {
 public:
  bool Evaluate(double const* const* parameters, double* residuals,
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

// Relative linear constraint with D degrees of freedom.
// residual = p1 - b * p0 - c
template <int D>
class BinaryCostFunction : public SizedCostFunction<D, D, D> {
 public:
  typedef Eigen::Matrix<double, D, 1> VectorType;
  BinaryCostFunction(const Eigen::Matrix<double, D, D>& b, const VectorType& c)
      : b_(b), c_(c) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    Eigen::Map<const VectorType> p0(parameters[0]);
    Eigen::Map<const VectorType> p1(parameters[1]);
    Eigen::Map<VectorType> res(residuals);

    res = p1 - b_ * p0 - c_;

    if (jacobians) {
      typedef Eigen::Matrix<double, D, D, Eigen::RowMajor> JacobianType;

      if (jacobians[0]) {
        Eigen::Map<JacobianType> J(jacobians[0]);
        J = JacobianType::Identity() * -1;
      }

      if (jacobians[1]) {
        Eigen::Map<JacobianType> J(jacobians[1]);
        J = JacobianType::Identity();
      }
    }

    return true;
  }

 private:
  VectorType c_;
  Eigen::Matrix<double, D, D> b_;
};

// Mixed DOF relative linear constraint, residual = p1 - p0 - value.
class BinaryCostFunctionMixed : public SizedCostFunction<2, 1, 2> {
 public:
  BinaryCostFunctionMixed(const Eigen::Vector2d& value) : value_(value) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    const Eigen::Vector2d p0(parameters[0][0], parameters[0][0]);
    Eigen::Map<const Eigen::Vector2d> p1(parameters[1]);

    Eigen::Map<Eigen::Vector2d> res(residuals);
    res = p1 - p0 - value_;

    if (jacobians) {
      if (jacobians[0]) {
        Eigen::Map<Eigen::Vector2d> J(jacobians[0]);
        J(0) = -1;
        J(1) = -1;
      }

      if (jacobians[1]) {
        typedef Eigen::Matrix<double, 2, 2, Eigen::RowMajor> JacobianType;
        Eigen::Map<JacobianType> J(jacobians[1]);
        J = JacobianType::Identity();
      }
    }

    return true;
  }

 private:
  Eigen::Vector2d value_;
};

// This is a toy linear parameterization where the local space dimension is 1
// and the global space dimension is user specified.
class PureOverparameterization : public MarginalizableParameterization,
                                 public LocalParameterization {
 public:
  explicit PureOverparameterization(int size) : size_(size) {
    CHECK_GT(size_, 0);
    v_ = Vector::Ones(size_);
    for (int i = 0; i < size; ++i) {
      v_[i] = 1 + i * i;
    }
  }

  virtual ~PureOverparameterization() {}

  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const override {
    VectorRef(x_plus_delta, size_) = ConstVectorRef(x, size_) + *delta * v_;
    return true;
  }

  virtual bool Minus(const double* x_plus_delta, const double* x,
                     double* delta) const override {
    *delta = (x_plus_delta[0] - x[0]) / v_[0];
    return true;
  }

  virtual bool ComputeMinusJacobian(const double* x,
                                    const double* x0,
                                    double* jacobian) const override {
    (void)x;
    (void)x0;
    jacobian[0] = v_[0];
    for (int i = 1 ; i < size_ ; ++i )
    {
      jacobian[i] = 0;
    }
    return true;
  }

  virtual bool ComputeJacobian(const double* x,
                               double* jacobian) const override {
    VectorRef(jacobian, size_) = v_;
    return true;
  }

  virtual int GlobalSize() const { return size_; }
  virtual int LocalSize() const { return 1; }

 private:
  const int size_;
  Vector v_;
};

template <typename ParameterizationType>
static void TestMarginalizableParameterization(const ParameterizationType* par, const Vector& x, const Vector& delta)
{
  int global_size = x.rows();
  int local_size = delta.rows();

  Vector x_plus_delta(global_size);
  par->Plus(x.data(), delta.data(), x_plus_delta.data());

  // Consistency of Plus and Minus
  {
    Vector delta_from_minus(local_size);
    par->Minus(x_plus_delta.data(), x.data(), delta_from_minus.data());
    const double residual = (delta_from_minus - delta).norm() / delta.norm();
    GTEST_ASSERT_LT(residual, 1e-10);
  }

  // Minus(x, x) = 0
  {
    Vector delta_from_minus(local_size);
    par->Minus(x.data(), x.data(), delta_from_minus.data());
    const double residual = delta_from_minus.norm() / delta.norm();
    GTEST_ASSERT_LT(residual, 1e-10);
  }

  {
    Matrix jacobian_minus_analytical(local_size, global_size);
    par->ComputeMinusJacobian(x.data(), x.data(), jacobian_minus_analytical.data());
    
    Matrix jacobian_plus_analytical(global_size, local_size);
    par->ComputeJacobian(x.data(), jacobian_plus_analytical.data());

    Matrix shouldBeIdentity = jacobian_minus_analytical * jacobian_plus_analytical;
    const double residual = (shouldBeIdentity - Matrix::Identity(local_size, local_size)).norm();
    GTEST_ASSERT_LT(residual, 1e-10);
  }
}

TEST(Marginalization, TestPureOverparameterization) {
  PureOverparameterization par(4);
  Vector x = Vector::Random(4);
  x = x / x.norm();
  Vector delta = Vector::Random(1);
  ASSERT_NO_FATAL_FAILURE(TestMarginalizableParameterization(&par, x, delta));
}

TEST(Marginalization, TestQuaternionParameterization) {
  MarginalizableQuaternionParameterization par;
  Vector q = Vector::Random(4);
  q = q / q.norm();
  Vector delta = Vector::Random(3);
  ASSERT_NO_FATAL_FAILURE(TestMarginalizableParameterization(&par, q, delta));
}

TEST(Marginalization, TestIdentityParameterization) {
  const int size = 3;
  MarginalizableIdentityParameterization par(size);
  Vector x = Vector::Random(size);
  Vector delta = Vector::Random(size);
  ASSERT_NO_FATAL_FAILURE(TestMarginalizableParameterization(&par, x, delta));
}

static void TestLinearizationState(const MarginalizationOptions& options)
{
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
    problem.SetParameterization(&states[0], new MarginalizableIdentityParameterization(1));
    problem.SetParameterization(&states[1], new MarginalizableIdentityParameterization(1));
    problem.AddResidualBlock(new NonlinearBinaryCostFunction(), NULL, &states[0],
                            &states[1]);

    problem.AddResidualBlock(
        new LinearCostFunction(2.0 * Eigen::Matrix<double, 1, 1>::Identity(),
                                Eigen::Matrix<double, 1, 1>(-2.0),
                                vector<int>({1})),
        NULL,
        &states[0]);

    const set<double*> parameter_blocks_to_marginalize = {&states[0]};
    const std::map<double*, const double*> parameter_block_linearization_states =
        {{&states[0], &linearization_states[0]},
        {&states[1], &linearization_states[1]}};

    ResidualBlockId marginal_factor_id;
    EXPECT_TRUE(MarginalizeOutVariables(
        options, parameter_blocks_to_marginalize, &problem,
        &marginal_factor_id, &parameter_block_linearization_states));

    const auto* marginal_factor = dynamic_cast<const MarginalizationPriorCostFunction*>(
        problem.GetCostFunctionForResidualBlock(marginal_factor_id));

    EXPECT_TRUE(marginal_factor);

    const vector<Matrix> J = marginal_factor->GetJacobianWrtIncrement();
    const Vector b = marginal_factor->GetB();
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
  constexpr static int kNumBlocks = 5;

  TestGraphState() {
    state_vector_ << 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 5.0, 5.5;
    parameter_block_sizes_ = {2, 2, 1, 1, 2};

    parameter_block_offsets_.resize(parameter_block_sizes_.size(), 0);
    std::partial_sum(parameter_block_sizes_.begin(),
                     parameter_block_sizes_.end() - 1,
                     parameter_block_offsets_.begin() + 1,
                     std::plus<int>());

    for (int k = 0; k < kNumBlocks; k++) {
      ordered_parameter_blocks_.push_back(
          &state_vector_(parameter_block_offsets_[k]));
    }

    for (int k = 0; k < kNumBlocks; k++) {
      int block_size = parameter_block_sizes_[k];
      problem_.AddParameterBlock(ordered_parameter_blocks_[k], block_size);
      LocalParameterization* parameterization =
          k != 1 ? static_cast<LocalParameterization*>(
                       new MarginalizableIdentityParameterization(block_size))
                 : static_cast<LocalParameterization*>(
                       new PureOverparameterization(block_size));
      problem_.SetParameterization(ordered_parameter_blocks_[k],
                                   parameterization);
    }

    const auto I2x2 = Eigen::Matrix<double, 2, 2>::Identity();

    ResidualBlockId prior_x1 = problem_.AddResidualBlock(
        new LinearCostFunction(I2x2, Eigen::Vector2d(2, 2), vector<int>({2})), NULL,
        ordered_parameter_blocks_[1]);
    ResidualBlockId rel_x0_x1 = problem_.AddResidualBlock(
        new BinaryCostFunction<2>(I2x2, Eigen::Vector2d(1, 1)), NULL,
        ordered_parameter_blocks_[0], ordered_parameter_blocks_[1]);
    ResidualBlockId rel_x0_x2 = problem_.AddResidualBlock(
        new BinaryCostFunctionMixed(Eigen::Vector2d(2, 2)), NULL,
        ordered_parameter_blocks_[2], ordered_parameter_blocks_[0]);
    ResidualBlockId rel_x1_x3 = problem_.AddResidualBlock(
        new BinaryCostFunctionMixed(Eigen::Vector2d(2, 2)), NULL,
        ordered_parameter_blocks_[3], ordered_parameter_blocks_[1]);
    ResidualBlockId rel_x2_x4 = problem_.AddResidualBlock(
        new BinaryCostFunctionMixed(Eigen::Vector2d(2, 2)), NULL,
        ordered_parameter_blocks_[2], ordered_parameter_blocks_[4]);
    ResidualBlockId rel_x3_x4 = problem_.AddResidualBlock(
        new BinaryCostFunctionMixed(Eigen::Vector2d(1, 1)), NULL,
        ordered_parameter_blocks_[3], ordered_parameter_blocks_[4]);
    ResidualBlockId rel_x0_x4 = problem_.AddResidualBlock(
        new BinaryCostFunction<2>(I2x2, Eigen::Vector2d(4, 4)), NULL,
        ordered_parameter_blocks_[0], ordered_parameter_blocks_[4]);

    for (int k = 0; k < ordered_parameter_blocks_.size(); k++) {
      parameter_block_local_sizes_.push_back(
          problem_.ParameterBlockLocalSize(ordered_parameter_blocks_[k]));
    }
    parameter_block_local_offsets_.resize(parameter_block_local_sizes_.size(),
                                          0);
    std::partial_sum(parameter_block_local_sizes_.begin(),
                     parameter_block_local_sizes_.end() - 1,
                     parameter_block_local_offsets_.begin() + 1,
                     std::plus<int>());
  }

  void Perturb() {
    vector<double*> parameter_blocks;
    problem_.GetParameterBlocks(&parameter_blocks);
    for (int b = 0; b < parameter_blocks.size(); ++b) {
      double* pb = parameter_blocks[b];
      const int local_size = problem_.ParameterBlockLocalSize(pb);
      const int size = problem_.ParameterBlockSize(pb);
      Vector local_perturbation(local_size);
      RandomVector(&local_perturbation);

      // Apply perturbation to this parameter block.
      const LocalParameterization* parameterization =
          problem_.GetParameterization(pb);
      Vector pb_perturbed(size);
      if (parameterization) {
        parameterization->Plus(pb, local_perturbation.data(),
                               pb_perturbed.data());
      } else {
        pb_perturbed = VectorRef(pb, size) + local_perturbation;
      }
      VectorRef(pb, size) = pb_perturbed;
    }
  }

  Vector GetStateVector(const bitset<kNumBlocks> selection) const {
    Vector res(state_vector_.size());
    int offset = 0;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        for (int j = 0; j < parameter_block_sizes_[i]; j++, offset++) {
          res(offset) = state_vector_(parameter_block_offsets_[i] + j);
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
      const bitset<kNumBlocks> parameter_blocks_to_marginalize_mask) {
    const set<double*> parameter_blocks_to_marginalize =
        GetParameterBlockSubset(parameter_blocks_to_marginalize_mask);

    EXPECT_TRUE(MarginalizeOutVariables(options_,
                                        parameter_blocks_to_marginalize,
                                        &problem_, nullptr));
  }

  set<double*> GetParameterBlockSubset(const bitset<kNumBlocks> selection) {
    set<double*> subset;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        subset.insert(ordered_parameter_blocks_[i]);
      }
    }
    return subset;
  }

  int GetProblemLocalSize(
      const vector<const double*>& problem_parameter_blocks) {
    int local_size = 0;
    for (int i = 0; i < problem_parameter_blocks.size(); i++) {
      local_size +=
          problem_.ParameterBlockLocalSize(problem_parameter_blocks[i]);
    }
    return local_size;
  }

  Matrix GetCovarianceMatrixInTangentSpace(
      vector<const double*>& covariance_blocks) {
    Covariance::Options options;
    Covariance covariance(options);
    const int local_size = GetProblemLocalSize(covariance_blocks);
    Matrix covariance_matrix(local_size, local_size);

    CHECK(covariance.Compute(covariance_blocks, &problem_));
    CHECK(covariance.GetCovarianceMatrixInTangentSpace(
        covariance_blocks, covariance_matrix.data()));
    return covariance_matrix;
  }

  Matrix GetCovarianceMatrixInTangentSpace() {
    vector<double*> problem_parameter_blocks;
    problem_.GetParameterBlocks(&problem_parameter_blocks);
    vector<const double*> covariance_blocks;
    for (double* pb : problem_parameter_blocks) {
      covariance_blocks.push_back(pb);
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

  Matrix GetCovarianceMatrixInTangentSpace(const bitset<kNumBlocks> selection) {
    vector<const double*> covariance_blocks;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        covariance_blocks.push_back(ordered_parameter_blocks_[i]);
      }
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

 private:
  Eigen::Matrix<double, 8, 1> state_vector_;
  vector<double*> ordered_parameter_blocks_;
  vector<int> parameter_block_sizes_;
  vector<int> parameter_block_offsets_;
  vector<int> parameter_block_local_sizes_;
  vector<int> parameter_block_local_offsets_;
  Problem problem_;
  MarginalizationOptions options_;

  static void RandomVector(Vector* v) {
    for (int r = 0; r < v->rows(); ++r) (*v)[r] = 2 * RandDouble() - 1;
  }
};

static void TestMarginalization(unsigned int m) {
  bitset<TestGraphState::kNumBlocks> current_mask(m);
  TestGraphState state;

  const Matrix marginal_covariance_expected =
      state.GetCovarianceMatrixInTangentSpace(~current_mask);

  state.SolveProblem();

  const Vector state_after_first_solve = state.GetStateVector(~current_mask);

  state.Perturb();

  state.SolveProblem();

  state.Perturb();

  state.MarginalizeOutVariableSubset(current_mask);

  const Matrix marginal_covariance_actual =
      state.GetCovarianceMatrixInTangentSpace();

  const double cov_error =
      (marginal_covariance_expected - marginal_covariance_actual).norm();
  EXPECT_LT(cov_error, 1e-6);

  // Solve the new problem to compute the marginal mean.
  state.Perturb();
  state.SolveProblem();

  const Vector state_after_marginalization =
      state.GetStateVector(~current_mask);
  const double state_error =
      (state_after_marginalization - state_after_first_solve)
          .lpNorm<Eigen::Infinity>();
  EXPECT_LT(state_error, 1e-6);
}

static void TestMarginalization()
{
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
    TestMarginalization(m);
  }
}

TEST(Marginalization, MarginalizationSuccess) {
  ASSERT_NO_FATAL_FAILURE(TestMarginalization());
}

TEST(Marginalization, MarginalizationFailure) {
  // No markov blanket
  double x[2] = {0, 0};
  Problem problem;
  problem.AddParameterBlock(x, 2);

  problem.AddResidualBlock(
      new LinearCostFunction(Eigen::Matrix<double, 2, 2>::Identity(),
                             Eigen::Vector2d(1.0, 2.0),
                             vector<int>({2})),
      NULL,
      x);
  const set<double*> to_marginalize = {x};
  MarginalizationOptions options;
  EXPECT_FALSE(
      MarginalizeOutVariables(options, to_marginalize, &problem, nullptr));
}

}  // namespace internal
}  // namespace ceres
