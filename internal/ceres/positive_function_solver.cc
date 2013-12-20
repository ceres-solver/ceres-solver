#include "ceres/positive_function_solver.h"

#include "ceres/ceres.h"

namespace ceres {
namespace internal {
void PositiveFunctionSolver(const Solver::Options& options,
                            const PositiveFunction& positive_function,
                            const int num_parameters,
                            double* parameters,
                            Solver::Summary* summary);

class PositiveFunctionWrapper : public CostFunction {
 public:
  PositiveFunctionWrapper(const PositiveFunction& function,
                          int num_parameters)
      : function_(function) {
    CHECK_GE(num_parameters, 1);
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(num_parameters);
  }

  virtual ~PositiveFunctionWrapper() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double cost = 0.0;
    const bool evaluate_jacobian =
        (jacobians != NULL && jacobians[0] != NULL);
    double* gradient = evaluate_jacobian ? jacobians[0] : NULL;
    if (function_(parameters[0], &cost, gradient)) {
      CHECK_GT(cost, 0.0);
      const double r = sqrt(2.0 * cost);
      if (residuals != NULL) {
        residuals[0] = r;
      }

      if (evaluate_jacobian) {
        for (int i = 0; i < parameter_block_sizes()[0]; ++i) {
          gradient[i] /= r;
        }
      }

      return true;
    }

    return false;
  }

 private:
  const PositiveFunction& function_;
};

} // namespace internal

void PositiveFunctionSolver(const Solver::Options& options,
                            const PositiveFunction& positive_function,
                            const int num_parameters,
                            double* parameters,
                            Solver::Summary* summary) {
  CHECK_EQ(options.minimizer_type, ceres::LINE_SEARCH);
  Problem problem;
  problem.AddResidualBlock(
      new internal::PositiveFunctionWrapper(positive_function,
                                            num_parameters),
      NULL,
      parameters);
  Solve(options, &problem, summary);
};

}  // namespace ceres
