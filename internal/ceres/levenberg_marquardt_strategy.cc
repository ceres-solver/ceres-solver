#include "ceres/levenberg_marquardt_strategy.h"

#include <cmath>
#include "glog/logging.h"
#include "ceres/array_utils.h"
#include "ceres/internal/eigen.h"
#include "ceres/linear_solver.h"
#include "ceres/sparse_matrix.h"
#include "ceres/trust_region_strategy.h"
#include "ceres/types.h"
#include "Eigen/Core"

namespace ceres {
namespace internal {

LevenbergMarquardtStrategy::LevenbergMarquardtStrategy(
    const TrustRegionStrategy::Options& options)
    : linear_solver_(options.linear_solver),
      radius_(options.initial_radius),
      max_radius_(options.max_radius),
      min_diagonal_(options.lm_min_diagonal),
      max_diagonal_(options.lm_max_diagonal),
      decrease_factor_(2.0),
      reuse_diagonal_(false) {
  CHECK_NOTNULL(linear_solver_);
  CHECK_GT(min_diagonal_, 0.0);
  CHECK_LT(min_diagonal_, max_diagonal_);
  CHECK_GT(max_radius_, 0.0);
}

LevenbergMarquardtStrategy::~LevenbergMarquardtStrategy() {
}

LinearSolver::Summary LevenbergMarquardtStrategy::ComputeStep(
    const TrustRegionStrategy::PerSolveOptions& per_solve_options,
    SparseMatrix* jacobian,
    const double* residuals,
    double* step) {
  CHECK_NOTNULL(jacobian);
  CHECK_NOTNULL(residuals);
  CHECK_NOTNULL(step);

  const int num_parameters = jacobian->num_cols();
  if (!reuse_diagonal_) {
    if (diagonal_.rows() != num_parameters) {
      diagonal_.resize(num_parameters, 1);
    }

    jacobian->SquaredColumnNorm(diagonal_.data());
    for (int i = 0; i < num_parameters; ++i) {
      diagonal_[i] = min(max(diagonal_[i], min_diagonal_), max_diagonal_);
    }
  }

  lm_diagonal_ = (diagonal_ / radius_).array().sqrt();

  LinearSolver::PerSolveOptions solve_options;
  solve_options.D = lm_diagonal_.data();
  solve_options.q_tolerance = per_solve_options.eta;
  // Disable r_tolerance checking. Since we only care about
  // termination via the q_tolerance. As Nash and Sofer show,
  // r_tolerance based termination is essentially useless in
  // Truncated Newton methods.
  solve_options.r_tolerance = -1.0;

  // Invalidate the output array lm_step, so that we can detect if
  // the linear solver generated numerical garbage.  This is known
  // to happen for the DENSE_QR and then DENSE_SCHUR solver when
  // the Jacobin is severly rank deficient and mu is too small.
  InvalidateArray(num_parameters, step);
  LinearSolver::Summary linear_solver_summary =
      linear_solver_->Solve(jacobian, residuals, solve_options, step);
  if (linear_solver_summary.termination_type == FAILURE ||
      !IsArrayValid(num_parameters, step)) {
    LOG(WARNING) << "Linear solver failure. Failed to compute a finite step.";
    linear_solver_summary.termination_type = FAILURE;
  }

  reuse_diagonal_ = true;
  return linear_solver_summary;
}

void LevenbergMarquardtStrategy::StepAccepted(double step_quality) {
  CHECK_GT(step_quality, 0.0);
  radius_ = radius_ / std::max(1.0 / 3.0,
                               1.0 - pow(2.0 * step_quality - 1.0, 3));
  radius_ = std::min(max_radius_, radius_);
  decrease_factor_ = 2.0;
  reuse_diagonal_ = false;
}

void LevenbergMarquardtStrategy::StepRejected(double step_quality) {
  radius_ = radius_ / decrease_factor_;
  decrease_factor_ *= 2.0;
  reuse_diagonal_ = true;
}

double LevenbergMarquardtStrategy::Radius() const {
  return radius_;
}

}  // namespace internal
}  // namespace ceres
