#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {
namespace experimental {

class AugmentedLagrangianCostFunction : public CostFunction {
 public:
  AugmentedLagrangianCostFunction(CostFunction* constraint)
      : constraint_(constraint) {
    set_num_residuals(constraint_->num_residuals());

    slack_.resize(num_residuals());
    slack_.setZero();
    lambda_.resize(num_residuals());
    lambda_.setZero();
    mu_ = 1.0;

    vector<int32>* parameter_block_sizes = mutable_parameter_block_sizes();
    *parameter_block_sizes = constraint->parameter_block_sizes();
    parameter_block_sizes->push_back(num_residuals());
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    if (!constraint_->Evaluate(parameters, residuals, jacobians)) {
      return false;
    }

    const double scale = 1.0 / sqrt(2 * mu_);
    VectorRef residuals_ref(residuals, num_residuals());
    residuals_ref = (residuals_ref - slack_)  * scale + lambda_ * sqrt(mu_ / 2.0);
    if (jacobians == NULL) {
      return true;
    }

    const vector<int32>& block_sizes = parameter_block_sizes();
    const int num_parameter_blocks = block_sizes.size();
    for (int i = 0; i < num_parameter_blocks - 1; ++i) {
      if (jacobians[i] != NULL) {
        MatrixRef(jacobians[i], num_residuals(), block_sizes[i]) *= scale;
      }
    }

    if (jacobians[num_parameter_blocks - 1] != NULL) {
      MatrixRef(jacobians[num_parameter_blocks - 1],
                num_residuals(),
                num_residuals()) =
          -scale * Matrix::Identity(num_residuals(), num_residuals());
    }
    return true;
  }

  double* mutable_slack() { return slack_.data(); }
  const double* slack() const { return slack_.data(); }
  double* mutable_lambda() { return lambda_.data(); }
  const double* lambda() const { return lambda_.data(); }
  double mu() const { return mu_; }
  void set_mu(double mu) { mu_ = mu; }

 private:
  ::ceres::internal::scoped_ptr<CostFunction> constraint_;
  Vector slack_;
  Vector lambda_;
  double mu_;
};

}  // namespace experimental
}  // namespace ceres
