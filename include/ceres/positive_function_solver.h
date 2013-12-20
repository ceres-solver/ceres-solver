#ifndef CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_
#define CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_

#include "ceres/solver.h"

namespace ceres {

struct PositiveFunction {
  virtual bool operator()(const double* parameters, double* cost, double* gradient) const = 0;
};

void PositiveFunctionSolver(const Solver::Options& options,
                            const PositiveFunction& positive_function,
                            const int num_parameters,
                            double* parameters,
                            Solver::Summary* summary);

}  // namespace ceres

#endif  // CERES_PUBLIC_POSITIVE_FUNCTION_SOLVER_H_
