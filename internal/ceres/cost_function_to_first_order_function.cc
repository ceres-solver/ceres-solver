#include "ceres/cost_function_to_first_order_function.h"
#include "ceres/internal/eigen.h"
#include "glog/logging.h"

namespace ceres {

FirstOrderCostFunction::FirstOrderCostFunction(CostFunction* cost_function)
    : cost_function_(cost_function)
{
  CHECK_EQ(cost_function_->parameter_block_sizes().size(), 1)
      << "FirstOrderFunctions only support a single parameter block.";
}

bool FirstOrderCostFunction::Evaluate(const double* const parameters,
                                      double* cost,
                                      double* gradient) const {
  Vector residuals(cost_function_->num_residuals());
  if (NULL == gradient) {
    // Evaluate residuals only.
    if (!cost_function_->Evaluate(&parameters,
                                  residuals.data(),
                                  NULL)) {
      return false;
    }
  } else {
    // Evaluate residuals and gradient.
    Matrix jacobian(cost_function_->num_residuals(), this->NumParameters());
    double* jacobian_data = jacobian.data();
    if (!cost_function_->Evaluate(&parameters,
                                  residuals.data(),
                                  &jacobian_data)) {
      return false;
    }
    // Cost := sum(residuals), therefore sum over residuals to compute the
    // gradient from the jacobian.
    VectorRef(gradient, this->NumParameters()) =
        jacobian.colwise().sum().transpose();
  }
  cost[0] = residuals.sum();
  return true;
}

int FirstOrderCostFunction::NumParameters() const {
  return cost_function_->parameter_block_sizes().front();
}

}
