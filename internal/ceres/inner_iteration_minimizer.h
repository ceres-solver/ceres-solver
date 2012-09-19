
#include <vector>
#include "ceres/internal/scoped_ptr.h"
#include "ceres/minimizer.h"
#include "ceres/problem_impl.h"
#include "ceres/solver.h"

namespace ceres {
namespace internal {

class Program;

class InnerIterationMinimizer : public Minimizer {
 public:
  bool Init(const Program& program,
            const ProblemImpl::ParameterMap& parameter_map,
            const vector<double*>& parameter_blocks_for_inner_iterations,
            string* error);

  virtual ~InnerIterationMinimizer();
  virtual void Minimize(const Minimizer::Options& options,
                        double* parameters,
                        Solver::Summary* summary);

 private:
  void MinimalSolve(Program* program, double* parameters, Solver::Summary* summary);
  void CountResidualBlocksPerParameterBlock(const int num_eliminate_blocks);

  scoped_ptr<Program> program_;
  vector<int> residual_block_offsets_;
};

}  // namespace internal
}  // namespace ceres
