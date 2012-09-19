#include "ceres/solver.h"
#include "ceres/program.h"

namespace ceres {
namespace internal {
class MinimalSolver {
 public:
  static Solver::Summary Solve(const Solver::Options& options, Program* program, double* parameters);
};

}  // namespace internal
}  // namespace ceres
