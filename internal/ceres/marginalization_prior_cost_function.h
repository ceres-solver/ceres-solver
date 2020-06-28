// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_INTERNAL_MARGINALIZATION_FACTOR_COST_FUNCTION_H_
#define CERES_INTERNAL_MARGINALIZATION_FACTOR_COST_FUNCTION_H_

#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/marginalizable_parameterization.h"

namespace ceres {
namespace internal {

// Class for the linear cost function
//  residual = A * (x (-) x0) + b.
//
// A is "jacobian_wrt_increment"
//
class MarginalizationPriorCostFunction : public CostFunction {
 public:
  MarginalizationPriorCostFunction(
      const Matrix& jacobian_wrt_increment,
      const Matrix& b,
      const std::vector<std::vector<double>>& parameter_block_reference_points,
      const std::vector<int>& parameter_block_local_sizes,
      const std::vector<const MarginalizableParameterization*>
          parameter_block_parameterizations)
      : b_(b),
        parameter_block_local_sizes_(parameter_block_local_sizes),
        parameter_block_reference_points_(parameter_block_reference_points),
        parameter_block_parameterizations_(parameter_block_parameterizations) {
    set_num_residuals(b_.size());

    jacobian_wrt_increment_blocks_.resize(
        parameter_block_reference_points.size());
    int jacobian_column_offset = 0;
    for (size_t i = 0; i < parameter_block_parameterizations.size(); i++) {
      int local_size = parameter_block_local_sizes[i];
      jacobian_wrt_increment_blocks_[i] = jacobian_wrt_increment.block(
          0, jacobian_column_offset, jacobian_wrt_increment.rows(), local_size);
      jacobian_column_offset += local_size;
    }

    for (const auto& r : parameter_block_reference_points) {
      mutable_parameter_block_sizes()->push_back(r.size());
    }
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override {
    VectorRef(residuals, num_residuals()) = b_;
    const int num_parameter_blocks = parameter_block_reference_points_.size();

    for (int j = 0; j < num_parameter_blocks; ++j) {
      const int local_size = static_cast<int>(parameter_block_local_sizes_[j]);

      Vector delta(local_size);
      parameter_block_parameterizations_[j]->Minus(
          parameters[j],
          parameter_block_reference_points_[j].data(),
          delta.data());

      VectorRef(residuals, num_residuals()) +=
          jacobian_wrt_increment_blocks_[j] * delta;
    }

    if (jacobians == NULL) {
      return true;
    }

    for (int j = 0; j < num_parameter_blocks; ++j) {
      const int local_size = static_cast<int>(parameter_block_local_sizes_[j]);
      const int global_size =
          static_cast<int>(parameter_block_reference_points_[j].size());

      Matrix minus_jacobian(local_size, global_size);
      parameter_block_parameterizations_[j]->ComputeMinusJacobian(
          parameters[j],
          parameter_block_reference_points_[j].data(),
          minus_jacobian.data());

      const Matrix jacobian_block =
          jacobian_wrt_increment_blocks_[j] * minus_jacobian;

      MatrixRef(jacobians[j], num_residuals(), global_size) = jacobian_block;
    }

    return true;
  }

  std::vector<Matrix> GetJacobianWrtIncrement() const {return jacobian_wrt_increment_blocks_; }
  Vector GetB() const {return b_; }

 private:
  std::vector<const MarginalizableParameterization*>
      parameter_block_parameterizations_;
  std::vector<std::vector<double>> parameter_block_reference_points_;
  std::vector<int> parameter_block_local_sizes_;
  Vector b_;
  std::vector<Matrix> jacobian_wrt_increment_blocks_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_MARGINALIZATION_FACTOR_COST_FUNCTION_H_
