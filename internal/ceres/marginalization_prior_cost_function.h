#ifndef CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_
#define CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_

#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/manifold.h"

namespace ceres {
namespace internal {

// Class for the cost function
//  residual = a * (x (-) x0) + b.
//
class MarginalizationPriorCostFunction : public CostFunction {
 public:
  MarginalizationPriorCostFunction(
      const Matrix& a,
      const Matrix& b,
      const std::vector<Vector>&
          parameter_block_reference_points,  // x0 in the equation above
      const std::vector<int>& parameter_block_tan_sizes,
      const std::vector<const Manifold*>& parameter_block_manifolds)
      : b_(b),
        parameter_block_reference_points_(parameter_block_reference_points),
        parameter_block_tan_sizes_(parameter_block_tan_sizes),
        parameter_block_manifolds_(parameter_block_manifolds) {
    // Validate the input
    CHECK_EQ(parameter_block_manifolds.size(),
             parameter_block_tan_sizes.size());
    CHECK_EQ(parameter_block_manifolds.size(),
             parameter_block_reference_points.size());
    for (int i = 0; i < parameter_block_manifolds.size(); i++) {
      if (parameter_block_manifolds[i]) {
        CHECK_EQ(parameter_block_manifolds[i]->TangentSize(),
                 parameter_block_tan_sizes[i]);
      }  // else this is a Euclidean manifold
    }

    set_num_residuals(b_.size());
    a_blocks_.resize(parameter_block_reference_points.size());
    int jacobian_column_offset = 0;
    for (int i = 0; i < parameter_block_manifolds.size(); i++) {
      int tan_size = parameter_block_tan_sizes_[i];
      a_blocks_[i] = a.block(0, jacobian_column_offset, a.rows(), tan_size);
      jacobian_column_offset += tan_size;
    }

    for (const Vector& r : parameter_block_reference_points) {
      mutable_parameter_block_sizes()->push_back(r.size());
    }
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const override {
    VectorRef(residuals, num_residuals()) = b_;
    const int num_parameter_blocks = parameter_block_reference_points_.size();

    for (int j = 0; j < num_parameter_blocks; ++j) {
      const int tan_size = parameter_block_tan_sizes_[j];

      Vector delta(tan_size);
      if (parameter_block_manifolds_[j] != nullptr) {
        parameter_block_manifolds_[j]->Minus(
            parameters[j],
            parameter_block_reference_points_[j].data(),
            delta.data());
      } else {
        // Euclidean manifold
        delta = ConstVectorRef(parameters[j], tan_size) -
                parameter_block_reference_points_[j];
      }

      VectorRef(residuals, num_residuals()) += a_blocks_[j] * delta;
    }

    if (jacobians == nullptr) {
      return true;
    }

    for (int j = 0; j < num_parameter_blocks; ++j) {
      const int tan_size = parameter_block_tan_sizes_[j];
      const int ambient_size =
          static_cast<int>(parameter_block_reference_points_[j].size());

      Matrix minus_jacobian(tan_size, ambient_size);
      if (parameter_block_manifolds_[j] != nullptr) {
        parameter_block_manifolds_[j]->MinusJacobian(
            parameter_block_reference_points_[j].data(), minus_jacobian.data());
      } else {
        // Euclidean manifold
        minus_jacobian.setIdentity();
      }

      const Matrix jacobian_block = a_blocks_[j] * minus_jacobian;

      MatrixRef(jacobians[j], num_residuals(), ambient_size) = jacobian_block;
    }

    return true;
  }

  std::vector<Matrix> GetJacobianWrtIncrement() const { return a_blocks_; }
  Vector GetB() const { return b_; }

 private:
  std::vector<int> parameter_block_tan_sizes_;
  std::vector<const Manifold*> parameter_block_manifolds_;
  std::vector<Vector> parameter_block_reference_points_;
  Vector b_;
  std::vector<Matrix> a_blocks_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_MARGINALIZATION_PRIOR_COST_FUNCTION_H_
