// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_INTERNAL_LINEAR_COST_FUNCTION_H_
#define CERES_INTERNAL_LINEAR_COST_FUNCTION_H_

#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"

namespace ceres {
namespace internal {

// Class for the linear cost function
//  residual = jacobian * x + b.
//
// Inputs:
//
// parameter_block_sizes is a vector of sizes of parameter blocks of x,
// and the jacobian is in the same order as parameter_block_sizes.
class LinearCostFunction : public CostFunction {
 public:
  LinearCostFunction(const Matrix& jacobian, const Matrix& b,
                     const std::vector<int>& parameter_block_sizes)
      : jacobian_(jacobian), b_(b) {
    set_num_residuals(b_.size());
    for (const int size : parameter_block_sizes) {
      mutable_parameter_block_sizes()->push_back(size);
    }
  }

  Matrix GetJacobian() const { return jacobian_; }

  Vector GetB() const { return b_; }

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const override {
    VectorRef(residuals, num_residuals()) = b_;
    for (int i = 0; i < num_residuals(); ++i) {
      int parameter_block_offset = 0;
      for (int j = 0; j < parameter_block_sizes().size(); ++j) {
        const int block_size = parameter_block_sizes()[j];
        for (int k = 0; k < block_size; ++k) {
          residuals[i] +=
              jacobian_(i, parameter_block_offset + k) * parameters[j][k];
        }
        parameter_block_offset += block_size;
      }
    }

    if (jacobians == NULL) {
      return true;
    }

    for (int i = 0; i < num_residuals(); ++i) {
      int parameter_block_offset = 0;
      for (int j = 0; j < parameter_block_sizes().size(); ++j) {
        const int block_size = parameter_block_sizes()[j];
        for (int k = 0; k < block_size; ++k) {
          jacobians[j][i * block_size + k] =
              jacobian_(i, parameter_block_offset + k);
        }
        parameter_block_offset += block_size;
      }
    }

    return true;
  }

 private:
  Vector b_;
  Matrix jacobian_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_LINEAR_COST_FUNCTION_H_
