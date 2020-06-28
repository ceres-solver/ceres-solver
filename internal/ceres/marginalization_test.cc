// Author: evanlevine138e@gmail.com (Evan Levine)

#include <bitset>
#include <memory>
#include <numeric>

#include "ceres/cost_function.h"
#include "ceres/covariance.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"
#include "ceres/loss_function.h"
#include "ceres/map_util.h"
#include "ceres/marginalization.h"
#include "ceres/parameter_block.h"
#include "ceres/problem.h"
#include "ceres/program.h"
#include "ceres/random.h"
#include "ceres/residual_block.h"
#include "ceres/sized_cost_function.h"
#include "ceres/solver.h"
#include "ceres/types.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Unary linear constraint with D degrees of freedom.
template <int D>
class UnaryCostFunction : public SizedCostFunction<D, D> {
 public:
  typedef Eigen::Matrix<double, D, 1> VectorType;
  UnaryCostFunction(const VectorType& value) : value_(value) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    Eigen::Map<const VectorType> p0(parameters[0]);

    Eigen::Map<VectorType> res(residuals);
    res = p0 - value_;

    if (jacobians) {
      typedef Eigen::Matrix<double, D, D, Eigen::RowMajor> JacobianType;

      if (jacobians[0]) {
        Eigen::Map<JacobianType> J(jacobians[0]);
        J = JacobianType::Identity();
      }
    }

    return true;
  }

 private:
  VectorType value_;
};

// Relative linear constraint with D degrees of freedom.
template <int D>
class BinaryCostFunction : public SizedCostFunction<D, D, D> {
 public:
  typedef Eigen::Matrix<double, D, 1> VectorType;
  BinaryCostFunction(const VectorType& value) : value_(value) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    Eigen::Map<const VectorType> p0(parameters[0]);
    Eigen::Map<const VectorType> p1(parameters[1]);
    const VectorType diff = p1 - p0;

    Eigen::Map<VectorType> res(residuals);
    res = diff - value_;

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
  VectorType value_;
};

// Mixed DOF relative linear constraint
class BinaryCostFunctionMixed : public SizedCostFunction<2, 1, 2> {
 public:
  BinaryCostFunctionMixed(const Eigen::Vector2d& value) : value_(value) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const {
    const Eigen::Vector2d p0(parameters[0][0], parameters[0][0]);
    Eigen::Map<const Eigen::Vector2d> p1(parameters[1]);
    const Eigen::Vector2d diff = p1 - p0;

    Eigen::Map<Eigen::Vector2d> res(residuals);
    res = diff - value_;

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

// Convert a CRSMatrix to a dense Eigen matrix.
static Matrix CRSToDenseMatrix(const CRSMatrix& input) {
  Matrix m;
  m.resize(input.num_rows, input.num_cols);
  m.setZero();
  for (int row = 0; row < input.num_rows; ++row) {
    for (int j = input.rows[row]; j < input.rows[row + 1]; ++j) {
      const int col = input.cols[j];
      m(row, col) = input.values[j];
    }
  }
  return m;
}

TEST(Marginalization, SlidingWindowOptimization) {
  // 1 DOF relative constraint
  class TestProcessModelCostFunction : public SizedCostFunction<1, 1, 1> {
   public:
    TestProcessModelCostFunction() {}
    static constexpr double f(double x) { return 1.2 * x; }
    static constexpr double df(double x) { return 1.2; }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const {
      residuals[0] = f(parameters[0][0]) - parameters[1][0];
      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = df(parameters[0][0]);
        }

        if (jacobians[1]) {
          jacobians[1][0] = -1.0;
        }
      }

      return true;
    }

   private:
    double value_;
  };

  // 1 DOF prior
  class TestObservationModelCostFunction : public SizedCostFunction<1, 1> {
   public:
    static constexpr double h(double x) { return 10.0 * x; }
    static constexpr double dh(double x) { return 10.0; }

    TestObservationModelCostFunction(double measurement)
        : measurement_(measurement) {}

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const {
      residuals[0] = h(parameters[0][0]) - measurement_;

      if (jacobians) {
        if (jacobians[0]) {
          jacobians[0][0] = dh(parameters[0][0]);
        }
      }

      return true;
    }

   private:
    double measurement_;
  };

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.

  const double sqrtQ = 0.5;
  const double sqrtR = 0.5;
  const unsigned int T = 20;

  double x_sim[T];
  double z_meas[T];
  x_sim[0] = 1.1;
  z_meas[0] = TestObservationModelCostFunction::h(x_sim[0]);
  for (unsigned int t = 1; t < T; t++) {
    x_sim[t] =
        TestProcessModelCostFunction::f(x_sim[t - 1]) + sqrtR * RandNormal();
    z_meas[t] =
        TestObservationModelCostFunction::h(x_sim[t]) + sqrtQ * RandNormal();
  }

  // Build the problem.
  double x_bo_causal[T];
  double x_bo[T];
  x_bo[0] = 1.1;
  for (unsigned int t = 0; t < T; t++) {
    if (t > 0) {
      x_bo[t] = TestProcessModelCostFunction::f(x_bo_causal[t - 1]);
    }

    // Create problem
    Problem problem;
    for (unsigned int k = 0; k <= t; k++) {
      problem.AddParameterBlock(&x_bo[k], 1);
      problem.SetParameterization(&x_bo[k], new IdentityParameterization(1));
      problem.AddResidualBlock(new TestObservationModelCostFunction(z_meas[k]),
                               NULL, &x_bo[k]);
      if (k > 0) {
        problem.AddResidualBlock(new TestProcessModelCostFunction(), NULL,
                                 &x_bo[k - 1], &x_bo[k]);
      }
    }

    // Run the solver!
    Solver::Options options;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    x_bo_causal[t] = x_bo[t];
  }

  // Simulate sliding window optimization
  const int window_size = 1;
  double x_sw_causal[T];
  double x_sw[T];
  x_sw[0] = 1.1;
  std::unique_ptr<Problem> pProblem(new Problem());
  for (unsigned int t = 0; t < T; t++) {
    // Augment the problem with a new state.
    pProblem->AddParameterBlock(&x_sw[t], 1);
    pProblem->SetParameterization(&x_sw[t], new IdentityParameterization(1));
    pProblem->AddResidualBlock(new TestObservationModelCostFunction(z_meas[t]),
                               NULL, &x_sw[t]);
    if (t > 0) {
      x_sw[t] = TestProcessModelCostFunction::f(x_sw_causal[t - 1]);
      pProblem->AddResidualBlock(new TestProcessModelCostFunction(), NULL,
                                 &x_sw[t - 1], &x_sw[t]);
    }

    // Marginalize out the oldest state.
    if (t >= window_size) {
      const std::set<double*> parameter_blocks_to_marginalize = {
          &x_sw[t - window_size]};
      MarginalizeOutVariables(parameter_blocks_to_marginalize, pProblem.get());
    }

    // Run the solver!
    Solver::Options options;
    Solver::Summary summary;
    Solve(options, pProblem.get(), &summary);
    x_sw_causal[t] = x_sw[t];
  }

  // Verify that results are the same.
  for (unsigned int t = 0; t < T; t++) {
    EXPECT_NEAR(x_sw_causal[t], x_bo_causal[t], 1e-3);
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

    ResidualBlockId prior_x1 = problem_.AddResidualBlock(
        new UnaryCostFunction<2>(Eigen::Vector2d(2, 2)), NULL,
        ordered_parameter_blocks_[1]);
    ResidualBlockId rel_x0_x1 = problem_.AddResidualBlock(
        new BinaryCostFunction<2>(Eigen::Vector2d(1, 1)), NULL,
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
        new BinaryCostFunction<2>(Eigen::Vector2d(4, 4)), NULL,
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
    std::vector<double*> parameter_blocks;
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

  Vector GetStateVector(const std::bitset<kNumBlocks> selection) const {
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

  void MarginalizeOutVariables(
      const std::bitset<kNumBlocks> parameter_blocks_to_marginalize_mask) {
    const std::set<double*> parameter_blocks_to_marginalize =
        GetParameterBlockSubset(parameter_blocks_to_marginalize_mask);
    ceres::MarginalizeOutVariables(parameter_blocks_to_marginalize, &problem_);
  }

  std::set<double*> GetParameterBlockSubset(
      const std::bitset<kNumBlocks> selection) {
    std::set<double*> subset;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        subset.insert(ordered_parameter_blocks_[i]);
      }
    }
    return subset;
  }

  int GetProblemLocalSize(
      const std::vector<const double*>& problem_parameter_blocks) {
    int local_size = 0;
    for (int i = 0; i < problem_parameter_blocks.size(); i++) {
      local_size +=
          problem_.ParameterBlockLocalSize(problem_parameter_blocks[i]);
    }
    return local_size;
  }

  Matrix GetCovarianceMatrixInTangentSpace(
      std::vector<const double*>& covariance_blocks) {
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
    std::vector<double*> problem_parameter_blocks;
    problem_.GetParameterBlocks(&problem_parameter_blocks);
    std::vector<const double*> covariance_blocks;
    for (double* pb : problem_parameter_blocks) {
      covariance_blocks.push_back(pb);
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

  Matrix GetCovarianceMatrixInTangentSpace(
      const std::bitset<kNumBlocks> selection) {
    std::vector<const double*> covariance_blocks;
    for (int i = 0; i < kNumBlocks; i++) {
      if (selection.test(i)) {
        covariance_blocks.push_back(ordered_parameter_blocks_[i]);
      }
    }
    return GetCovarianceMatrixInTangentSpace(covariance_blocks);
  }

 private:
  Vector state_vector_;
  std::vector<double*> ordered_parameter_blocks_;
  std::vector<int> parameter_block_sizes_;
  std::vector<int> parameter_block_offsets_;
  std::vector<int> parameter_block_local_sizes_;
  std::vector<int> parameter_block_local_offsets_;
  Problem problem_;

  static void RandomVector(Vector* v) {
    for (int r = 0; r < v->rows(); ++r) (*v)[r] = 2 * RandDouble() - 1;
  }
};

static void TestMarginalization(unsigned int m) {
  std::bitset<TestGraphState::kNumBlocks> current_mask(m);
  TestGraphState state;

  const Matrix marginal_covariance_expected =
      state.GetCovarianceMatrixInTangentSpace(~current_mask);

  state.SolveProblem();

  const Vector state_after_first_solve = state.GetStateVector(~current_mask);

  state.Perturb();

  state.SolveProblem();

  state.Perturb();

  state.MarginalizeOutVariables(current_mask);

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

}  // namespace internal
}  // namespace ceres
