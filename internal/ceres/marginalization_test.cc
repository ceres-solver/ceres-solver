// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"
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

namespace ceres {
namespace internal {

using std::bitset;
using std::set;
using std::vector;

// Unary linear constraint with D degrees of freedom.
template <int D>
class UnaryCostFunction : public SizedCostFunction<D, D> {
 public:
  typedef Eigen::Matrix<double, D, 1> VectorType;
  UnaryCostFunction(const VectorType& value) : value_(value) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    Eigen::Map<const VectorType> p0(parameters[0]);
    const double a = 2.0;

    Eigen::Map<VectorType> res(residuals);
    res = a * p0 - value_;

    if (jacobians) {
      typedef Eigen::Matrix<double, D, D, Eigen::RowMajor> JacobianType;

      if (jacobians[0]) {
        Eigen::Map<JacobianType> J(jacobians[0]);
        J = a * JacobianType::Identity();
      }
    }

    return true;
  }

 private:
  VectorType value_;
};

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
class PureOverparameterization : public LocalParameterization {
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
                    double* x_plus_delta) const {
    VectorRef(x_plus_delta, size_) = ConstVectorRef(x, size_) + *delta * v_;
    return true;
  }

  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    VectorRef(jacobian, size_) = v_;
    return true;
  }

  virtual int GlobalSize() const { return size_; }
  virtual int LocalSize() const { return 1; }

 private:
  const int size_;
  Vector v_;
};

TEST(Marginalization, LinearizationState) {
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
   * prior(x1) = 2 * x1 + (-12 + 8 * sqrt(3)),
   */

  Problem problem;
  problem.AddParameterBlock(&states[0], 1);
  problem.AddParameterBlock(&states[1], 1);
  problem.SetParameterization(&states[0], new IdentityParameterization(1));
  problem.SetParameterization(&states[1], new IdentityParameterization(1));
  problem.AddResidualBlock(new NonlinearBinaryCostFunction(), NULL, &states[0],
                           &states[1]);
  problem.AddResidualBlock(
      new UnaryCostFunction<1>(Eigen::Matrix<double, 1, 1>(2.0)), NULL,
      &states[0]);

  const set<double*> parameter_blocks_to_marginalize = {&states[0]};
  const std::map<double*, const double*> parameter_block_linearization_states =
      {{&states[0], &linearization_states[0]},
       {&states[1], &linearization_states[1]}};

  ResidualBlockId marginal_factor_id;
  EXPECT_TRUE(MarginalizeOutVariables(parameter_blocks_to_marginalize, &problem,
                                      &marginal_factor_id,
                                      &parameter_block_linearization_states));

  const auto* marginal_factor = dynamic_cast<const LinearCostFunction*>(
      problem.GetCostFunctionForResidualBlock(marginal_factor_id));

  EXPECT_TRUE(marginal_factor);

  const Matrix J = marginal_factor->GetJacobian();
  const Vector b = marginal_factor->GetB();
  const double b_expected = -12.0 + 8.0 * sqrt(3.0);
  const double J_expected = 2.0;

  EXPECT_EQ(states[0], final_states[0]);
  EXPECT_EQ(states[1], final_states[1]);
  EXPECT_NEAR(b[0], b_expected, 1e-9);
  EXPECT_NEAR(J(0, 0), J_expected, 1e-9);
}

TEST(Marginalization, SlidingWindowOptimization) {
  // Solve a toy problem with a linear Gaussian model
  //
  //  x_k+1 = F * x_k + N(0, Q),
  //  z_k = H * x_k + N(0, R).
  //
  // by computing for time t = 0..T - 1
  // argmax p(x_t | z_0, z_1, ..., z_t).
  //
  // This can be done efficiently in an incremental fashion with a Kalman
  // filter or information filter, which are based on marginalization.
  // Alternatively, this can be implemented with the marginalization API. Here,
  // the solution with marginalization is compared to joint optimization of the
  // entire time history of states.

  const double sqrtQ = 0.5;
  const double sqrtR = 0.5;
  const unsigned int T = 20;

  const double F = 1.2;
  const double H = 2.5;
  const auto Fmtx = F * Eigen::Matrix<double, 1, 1>::Identity();

  double x_sim[T];
  double meas[T];
  x_sim[0] = 1.1;
  meas[0] = H * x_sim[0];

  for (unsigned int t = 1; t < T; t++) {
    x_sim[t] = F * x_sim[t - 1] + sqrtQ * RandNormal();
    meas[t] = H * x_sim[t] + sqrtR * RandNormal();
  }

  // Simulate scenario where a problem grows but marginalization is not used.
  double x_bo_hist[T];
  double x_bo[T];
  x_bo[0] = 1.1;
  for (unsigned int t = 0; t < T; t++) {
    if (t > 0) {
      x_bo[t] = F * x_bo_hist[t - 1];
    }

    // Create problem
    Problem problem;
    for (unsigned int k = 0; k <= t; k++) {
      problem.AddParameterBlock(&x_bo[k], 1);
      problem.SetParameterization(&x_bo[k], new IdentityParameterization(1));
      problem.AddResidualBlock(
          new UnaryCostFunction<1>(Eigen::Matrix<double, 1, 1>(meas[k])), NULL,
          &x_bo[k]);
      if (k > 0) {
        problem.AddResidualBlock(
            new BinaryCostFunction<1>(Fmtx, Eigen::Matrix<double, 1, 1>(0.0)),
            NULL, &x_bo[k - 1], &x_bo[k]);
      }
    }

    // Run the solver!
    Solver::Options options;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    x_bo_hist[t] = x_bo[t];
  }

  // Simulate sliding window optimization for the same problem.
  const int window_size = 1;
  double x_sw_hist[T];
  double x_sw[T];
  x_sw[0] = 1.1;
  std::unique_ptr<Problem> pProblem(new Problem());
  for (unsigned int t = 0; t < T; t++) {
    // Augment the problem with a new state.
    pProblem->AddParameterBlock(&x_sw[t], 1);
    pProblem->SetParameterization(&x_sw[t], new IdentityParameterization(1));
    pProblem->AddResidualBlock(
        new UnaryCostFunction<1>(Eigen::Matrix<double, 1, 1>(meas[t])), NULL,
        &x_sw[t]);
    if (t > 0) {
      x_sw[t] = F * x_sw_hist[t - 1];
      pProblem->AddResidualBlock(
          new BinaryCostFunction<1>(Fmtx, Eigen::Matrix<double, 1, 1>(0.0)),
          NULL, &x_sw[t - 1], &x_sw[t]);
    }

    // Marginalize out the oldest state.
    if (t >= window_size) {
      const set<double*> parameter_blocks_to_marginalize = {
          &x_sw[t - window_size]};

      EXPECT_TRUE(MarginalizeOutVariables(parameter_blocks_to_marginalize,
                                          pProblem.get(), nullptr));
    }

    // Run the solver!
    Solver::Options options;
    Solver::Summary summary;
    Solve(options, pProblem.get(), &summary);
    x_sw_hist[t] = x_sw[t];
  }

  // Verify that results are the same.
  for (unsigned int t = 0; t < T; t++) {
    EXPECT_NEAR(x_sw_hist[t], x_bo_hist[t], 1e-3);
  }
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
    state_vector_.resize(8);
    state_vector_(0) = 1.0;  // Block 0
    state_vector_(1) = 1.5;  // Block 0
    state_vector_(2) = 2.0;  // Block 1
    state_vector_(3) = 2.0;  // Block 1
    state_vector_(4) = 3.0;  // Block 2
    state_vector_(5) = 4.0;  // Block 3
    state_vector_(6) = 5.0;  // Block 4
    state_vector_(7) = 5.5;  // Block 4
    parameter_block_sizes_.push_back(2);
    parameter_block_sizes_.push_back(2);
    parameter_block_sizes_.push_back(1);
    parameter_block_sizes_.push_back(1);
    parameter_block_sizes_.push_back(2);

    parameter_block_offsets_.resize(parameter_block_sizes_.size(), 0);
    std::partial_sum(parameter_block_sizes_.begin(),
                     parameter_block_sizes_.end() - 1,
                     parameter_block_offsets_.begin() + 1, std::plus<int>());

    for (int k = 0; k < kNumBlocks; k++) {
      ordered_parameter_blocks_.push_back(
          &state_vector_(parameter_block_offsets_[k]));
    }

    for (int k = 0; k < kNumBlocks; k++) {
      problem_.AddParameterBlock(ordered_parameter_blocks_[k],
                                 parameter_block_sizes_[k]);
    }

    problem_.SetParameterization(
        ordered_parameter_blocks_[1],
        new PureOverparameterization(parameter_block_sizes_[1]));

    const auto I2x2 = Eigen::Matrix<double, 2, 2>::Identity();

    ResidualBlockId prior_x1 = problem_.AddResidualBlock(
        new UnaryCostFunction<2>(Eigen::Vector2d(2, 2)), NULL,
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

    EXPECT_TRUE(MarginalizeOutVariables(parameter_blocks_to_marginalize,
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
  Vector state_vector_;
  vector<double*> ordered_parameter_blocks_;
  vector<int> parameter_block_sizes_;
  vector<int> parameter_block_offsets_;
  vector<int> parameter_block_local_sizes_;
  vector<int> parameter_block_local_offsets_;
  Problem problem_;

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

TEST(Marginalization, MarkovBlanketMixed) {
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

TEST(Marginalization, MarginalizationFailure) {
  // No markov blanket
  double x = 0;
  Problem problem;
  problem.AddParameterBlock(&x, 1);

  problem.AddResidualBlock(
      new UnaryCostFunction<1>(Eigen::Matrix<double, 1, 1>(1.0)), NULL, &x);
  const set<double*> to_marginalize = {&x};
  EXPECT_FALSE(
      ceres::MarginalizeOutVariables(to_marginalize, &problem, nullptr));

  double y = 0;
  problem.AddParameterBlock(&y, 1);
  problem.AddResidualBlock(
      new UnaryCostFunction<1>(Eigen::Matrix<double, 1, 1>(1.0)), NULL, &x);
  EXPECT_FALSE(MarginalizeOutVariables(to_marginalize, &problem, nullptr));
}

}  // namespace internal
}  // namespace ceres
