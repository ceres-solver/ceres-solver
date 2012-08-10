#include <string>
#include <cstdio>
#include "glog/logging.h"
#include "ceres/ceres.h"
#include "performance.h"

namespace ceres {
namespace tools {

class FileLoggingCallback : public IterationCallback {
 public:
  explicit FileLoggingCallback(const string& filename)
      : fptr_(NULL) {
    fptr_ = fopen(filename.c_str(), "w");
    CHECK_NOTNULL(fptr_);
  }

  virtual ~FileLoggingCallback() {
    if (fptr_ != NULL) {
      fclose(fptr_);
    }
  }

  virtual CallbackReturnType operator()(const IterationSummary& summary) {
    fprintf(fptr_,
            "%4d %e %e\n",
            summary.iteration,
            summary.cost,
            summary.cumulative_time_in_seconds);
    return SOLVER_CONTINUE;
  }
 private:
    FILE* fptr_;
};

void SolveAndRecordPerformance(const std::string& problem_name,
                               const std::string& output_dir,
                               const Solver::Options& in_options,
                               Problem* problem,
                               Solver::Summary* summary) {
  Solver::Options options = in_options;
  FileLoggingCallback file_logging_callback(
      output_dir + "/" + problem_name);
  options.callbacks.push_back(&file_logging_callback);
  Solve(options, problem, summary);
  return;
};

}  // namespace tools
}  // namespace ceres
