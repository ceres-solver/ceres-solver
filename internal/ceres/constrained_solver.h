#include "ceres/constrained_problem.h"
#include "ceres/solver.h"

namespace ceres {
namespace experimental {

class ConstrainedSolver {
 public:
  static void Solve(const Solver::Options& options,
                    experimental::ConstrainedProblem* problem,
                    Solver::Summary* summary);
  static double EvaluateConstraintMaxNorm(experimental::ConstrainedProblem* problem);
};

}  // namespace experimental
}  // namespace ceres
