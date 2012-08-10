#include <string>
#include "ceres/ceres.h"

namespace ceres {
namespace tools {

void SolveAndRecordPerformance(const std::string& problem_name,
                               const std::string& output_dir,
                               const Solver::Options& options,
                               Problem* problem,
                               Solver::Summary* summary);


}  // namespace tools
}  // namespace ceres
