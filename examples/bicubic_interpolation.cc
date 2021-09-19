// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2021 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Bicubic interpolation using with analytic and automatic differentiation
//
// Let us define f(x, y) = x * x - y * x + y * y
// And optimize cost function sum_i [f(x_i + s_x, y_i + s_y) - v_i]^2
//
// Bicubic interpolation of f(x, y) will be exact, thus we can expect close to
// perfect convergence

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "glog/logging.h"

// Problem sizes
const int kGridRowsHalf = 9;
const int kGridColsHalf = 11;
const int kGridRows = 2 * kGridRowsHalf + 1;
const int kGridCols = 2 * kGridColsHalf + 1;
const int kPoints = 4;

// Class for holding data for interpolation
//
// Note that if input array is column-major, definition of Grid type alias
// requires changing corresponding default parameters (kRowMajor)
using Array = Eigen::Array<double, kGridRows, kGridCols, Eigen::RowMajor>;
using Grid = ceres::Grid2D<double>;
using Interpolator = ceres::BiCubicInterpolator<Grid>;

using RowVector = Eigen::Matrix<double, 1, kGridCols>;
using ColVector = Eigen::Matrix<double, kGridRows, 1>;
template <typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;
using Vector2d = Eigen::Vector2d;

// Cost-function using autodiff interface of BiCubicInterpolator
struct AutoDiffBiCubicCost {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  template <typename T>
  bool operator()(const T* s, T* residual) const {
    using Vector2T = Vector2<T>;
    const Eigen::Map<const Vector2T> shift(s);
    const Vector2T point = point_ + shift;

    T v;
    interpolator_.Evaluate(point.y(), point.x(), &v);

    *residual = v - value_;
    return true;
  }

  AutoDiffBiCubicCost(const Interpolator& interpolator,
                      const Vector2d& point,
                      double value)
      : interpolator_(interpolator), point_(point), value_(value) {}

  static ceres::CostFunction* Create(const Interpolator& interpolator,
                                     const Vector2d& point,
                                     double value) {
    return new ceres::AutoDiffCostFunction<AutoDiffBiCubicCost, 1, 2>(
        new AutoDiffBiCubicCost(interpolator, point, value));
  }

  const Interpolator& interpolator_;
  const Vector2d point_;
  const double value_;
};

// Cost-function using analytic interface of BiCubicInterpolator
struct AnalyticBiCubicCost : public ceres::CostFunction {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {
    Eigen::Map<const Vector2d> shift(parameters[0]);
    const Vector2d point = point_ + shift;

    double* f = residuals;
    double* dfdr = nullptr;
    double* dfdc = nullptr;
    if (jacobians && jacobians[0]) {
      dfdc = jacobians[0];
      dfdr = dfdc + 1;
    }
    interpolator_.Evaluate(point.y(), point.x(), f, dfdr, dfdc);
    if (residuals) {
      *f -= value_;
    }
    return true;
  }

  AnalyticBiCubicCost(const Interpolator& interpolator,
                      const Vector2d& point,
                      double value)
      : interpolator_(interpolator), point_(point), value_(value) {
    set_num_residuals(1);
    *mutable_parameter_block_sizes() = {2};
  }

  static ceres::CostFunction* Create(const Interpolator& interpolator,
                                     const Vector2d& point,
                                     double value) {
    return new AnalyticBiCubicCost(interpolator, point, value);
  }

  const Interpolator& interpolator_;
  const Vector2d point_;
  const double value_;
};

// Data generation routine
template <typename T>
T f(const T& x, const T& y) {
  return x * x - y * x + y * y;
}

// Rectangular grid evaluation
Array F();  // -Wmissing-declarations
Array F() {
  const Array x = RowVector::LinSpaced(kGridCols, -kGridColsHalf, kGridColsHalf)
                      .replicate(kGridRows, 1);
  const Array y = ColVector::LinSpaced(kGridRows, -kGridRowsHalf, kGridRowsHalf)
                      .replicate(1, kGridCols);

  return f(x, y);
}

template <typename Cost>
Vector2d Solve(const Vector2d& init,
               const Vector2d& shift,
               const Interpolator& interpolator,
               const Vector2d* points) {
  Vector2d params = init;

  ceres::Problem problem;
  problem.AddParameterBlock(params.data(), 2);

  for (int i = 0; i < kPoints; ++i) {
    const Vector2d shifted = points[i] + shift;
    const double v = f(shifted.x(), shifted.y());
    problem.AddResidualBlock(
        Cost::Create(interpolator, points[i], v), nullptr, params.data());
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << '\n';
  return params;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  const Vector2d shift(1.234, 2.345);
  const Vector2d init(3.1415, 1.337);
  const Vector2d points[kPoints] = {{-2., -3.}, {-2., 3.}, {2., 3.}, {2., -3.}};

  const Array array = F();
  const Grid grid(array.data(),
                  -kGridRowsHalf,
                  kGridRowsHalf + 1,
                  -kGridColsHalf,
                  kGridColsHalf + 1);
  const Interpolator interpolator(grid);

  std::cout << "Automatic derivatives:\n";
  const Vector2d autodiff =
      Solve<AutoDiffBiCubicCost>(init, shift, interpolator, points);
  std::cout << "Result: " << autodiff.transpose()
            << " (error: " << (autodiff - shift).transpose() << ")\n"
            << std::endl;

  std::cout << "Analytic derivatives:\n";
  const Vector2d analytic =
      Solve<AnalyticBiCubicCost>(init, shift, interpolator, points);
  std::cout << "Result: " << analytic.transpose()
            << " (error: " << (analytic - shift).transpose() << ")"
            << std::endl;

  CHECK_LT((autodiff - shift).norm(), 1e-9);
  CHECK_LT((analytic - shift).norm(), 1e-9);
  return 0;
}
