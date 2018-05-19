#include <iostream>
#include "ceres/ceres.h"

struct Trivial {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = pow(10.0 - x[0], 2.0);
    return true;
  }
};

int main(int argc, char** argv) {
  CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  double x = 0.5;

  ceres::GradientProblemSolver::Options options;
  options.minimizer_progress_to_stdout = true;

  ceres::GradientProblemSolver::Summary summary;
  ceres::GradientProblem problem(
      new ceres::FirstOrderCostFunction(
          new ceres::AutoDiffCostFunction<Trivial, 1, 1>(new Trivial)));
  std::cout << "Initial x = " << x << std::endl;
  ceres::Solve(options, problem, &x, &summary);
  std::cout << summary.FullReport() << "\n";
  std::cout << "Final x = " << x << std::endl;
  return 0;
}
