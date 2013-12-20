#include "ceres/ceres.h"

#ifndef CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_
#define CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_

namespace ceres {

// TODO(sameeragarwal): This should not be a template parameter. Since
// most problems will use fairly large parameter sizes.

template <int kNumParameters>
class PositiveFunction : public SizedCostFunction<1, kNumParameters> {
 public:
  virtual ~PositiveFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double cost = 0.0;
    bool evaluate_jacobian = (jacobians != nullptr && jacobians[0] != nullptr);
    double* gradient = evaluate_jacobian ? jacobians[0] : nullptr;
    if (EvaluateCostAndGradient(parameters[0], &cost, gradient)) {
      CHECK_GT(cost, 0.0);
      const double r = sqrt(2.0 * cost);
      if (residuals != nullptr) {
        residuals[0] = r;
      }

      if (evaluate_jacobian) {
        for (int i = 0; i < kNumParameters; ++i) {
          gradient[i] /= r;
        }
      }

      return true;
    }

    return false;
  }

  virtual bool EvaluateCostAndGradient(const double* parameters,
                                       double* cost,
                                       double* gradient) const = 0;
};

// this templating will go away
template <int kNumParameters>
void PositiveFunctionSolver(const Solver::Options& options,
                            const ceres::PositiveFunction<kNumParameters>& positive_function,
                            double* parameters,
                            Solver::Summary* summary) {
  CHECK_EQ(options.minimizer_type, ceres::LINE_SEARCH);
  Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  Problem problem(problem_options);
  const CostFunction* cost_function = &positive_function;
  problem.AddResidualBlock(const_cast<CostFunction*>(cost_function),
                           NULL,
                           parameters);
  Solve(options, &problem, summary);
};

}  // namespace ceres

#endif  // CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_
