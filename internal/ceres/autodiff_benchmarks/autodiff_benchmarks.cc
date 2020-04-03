// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2020 Google Inc. All rights reserved.
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
// Author: darius.rueckert@fau.de (Darius Rueckert)

#include <memory>
#include <random>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/constant_cost_function.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/photometric_error.h"
#include "ceres/autodiff_benchmarks/relative_pose_error.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"
#include "ceres/ceres.h"

namespace ceres {
namespace internal {

// If we want to use functors with both operator() and an Evaluate() method
// with AutoDiff then this wrapper class here has to be used. Autodiff doesn't
// support functors that have an Evaluate() function.
//
// CostFunctionToFunctor hides the Evaluate() function, because it doesn't
// derive from CostFunction. Autodiff sees it as a simple functor and will use
// the operator() as expected.
template <typename CostFunction>
struct CostFunctionToFunctor {
    template <typename... _Args>
    explicit CostFunctionToFunctor(_Args&&... __args)
        : cost_function(std::forward<_Args>(__args)...) {}

    template <typename... _Args>
    bool operator()(_Args&&... __args) const {
        return cost_function(std::forward<_Args>(__args)...);
    }

    CostFunction cost_function;
};

}  // namespace internal

template <int kParameterBlockSize>
static void BM_ConstantAnalytic(benchmark::State& state) {
  constexpr int num_residuals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, num_residuals> residuals;

  std::array<double, num_residuals * kParameterBlockSize> jacobian_values;
  double* jacobians[] = {jacobian_values.data()};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ConstantCostFunction<kParameterBlockSize>());

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals.data(), jacobians);
  }
}

template <int kParameterBlockSize>
static void BM_ConstantAutodiff(benchmark::State& state) {
  constexpr int num_residuals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, num_residuals> residuals;

  std::array<double, num_residuals * kParameterBlockSize> jacobian_values;
  double* jacobians[] = {jacobian_values.data()};

  using AutoDiffFunctor = ceres::internal::CostFunctionToFunctor<
      ConstantCostFunction<kParameterBlockSize>>;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<AutoDiffFunctor, 1, kParameterBlockSize>(
          new AutoDiffFunctor()));

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals.data(), jacobians);
  }
}

BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 1);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 1);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 10);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 10);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 20);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 20);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 30);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 30);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 40);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 40);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 50);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 50);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 60);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 60);

static void BM_Linear1AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<Linear1CostFunction>;

  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 1, 1>(new FunctorType()));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear1AutoDiff)->Arg(0)->Arg(1);

static void BM_Linear10AutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<Linear10CostFunction>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 10, 10>(new FunctorType()));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Linear10AutoDiff)->Arg(0)->Arg(1);

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* parameters, T* residuals) const {
    const T& b1 = parameters[0];
    const T& b2 = parameters[1];
    const T& b3 = parameters[2];
    const T& b4 = parameters[3];
    residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
    return true;
  }

 private:
  const double x_;
  const double y_;
};

static void BM_Rat43AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<Rat43CostFunctor, 1, 4>(
          new Rat43CostFunctor(x, y)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK(BM_Rat43AutoDiff)->Arg(0)->Arg(1);

static void BM_SnavelyReprojectionAutoDiff(benchmark::State& state) {
  using FunctorType =
      ceres::internal::CostFunctionToFunctor<SnavelyReprojectionError>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 2, 9, 3>(
          new FunctorType(x, y)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_SnavelyReprojectionAutoDiff)->Arg(0)->Arg(1);

static void BM_PhotometricAutoDiff(benchmark::State& state) {
  constexpr int PATCH_SIZE = 8;

  using FunctorType = PhotometricError<PATCH_SIZE>;
  using ImageType = Eigen::Matrix<uint8_t, 128, 128, Eigen::RowMajor>;

  // Prepare parameter / residual / jacobian blocks.
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  double parameter_block3[] = {1.};
  double* parameters[] = {parameter_block1, parameter_block2, parameter_block3};

  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  double jacobian1[FunctorType::PATCH_SIZE * FunctorType::POSE_SIZE];
  double jacobian2[FunctorType::PATCH_SIZE * FunctorType::POSE_SIZE];
  double jacobian3[FunctorType::PATCH_SIZE * FunctorType::POINT_SIZE];
  double residuals[FunctorType::PATCH_SIZE];
  double* jacobians[] = {jacobian1, jacobian2, jacobian3};

  // Prepare data (fixed seed for repeatability).
  std::mt19937::result_type seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  std::uniform_int_distribution<unsigned int> uniform0255(0, 255);

  FunctorType::Patch<double> intensities_host =
      FunctorType::Patch<double>::NullaryExpr(
          [&]() { return uniform0255(gen); });

  // Set bearing vector's z component to 1, i.e. pointing away from the camera,
  // to ensure they are (likely) in the domain of the projection function (given
  // a small rotation between host and target frame).
  FunctorType::PatchVectors<double> bearings_host =
      FunctorType::PatchVectors<double>::NullaryExpr(
          [&]() { return uniform01(gen); });
  bearings_host.row(2).array() = 1;
  bearings_host.colwise().normalize();

  ImageType image = ImageType::NullaryExpr(
      [&]() { return static_cast<uint8_t>(uniform0255(gen)); });
  FunctorType::Grid grid(image.data(), 0, image.rows(), 0, image.cols());
  FunctorType::Interpolator image_target(grid);

  FunctorType::Intrinsics intrinsics;
  intrinsics << 128, 128, 1, -1, 0.5, 0.5;

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType,
                                      FunctorType::PATCH_SIZE,
                                      FunctorType::POSE_SIZE,
                                      FunctorType::POSE_SIZE,
                                      FunctorType::POINT_SIZE>(new FunctorType(
          intensities_host, bearings_host, image_target, intrinsics)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_PhotometricAutoDiff)->Arg(0)->Arg(1);

static void BM_RelativePoseAutoDiff(benchmark::State& state) {
  using FunctorType = RelativePoseError;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  double* parameters[] = {parameter_block1, parameter_block2};

  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  double jacobian1[6 * 7];
  double jacobian2[6 * 7];
  double residuals[6];
  double* jacobians[] = {jacobian1, jacobian2};

  Eigen::Quaterniond q_i_j = Eigen::Quaterniond(1, 2, 3, 4).normalized();
  Eigen::Vector3d t_i_j(1, 2, 3);

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 6, 7, 7>(
          new FunctorType(q_i_j, t_i_j)));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_RelativePoseAutoDiff)->Arg(0)->Arg(1);

static void BM_BrdfAutoDiff(benchmark::State& state) {
  using FunctorType = ceres::internal::CostFunctionToFunctor<Brdf>;

  double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  auto c = Eigen::Vector3d(0.1, 0.2, 0.3);
  auto n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  auto v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  auto l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  auto x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  auto y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  double* parameters[7] = {
      material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()};

  double jacobian[(10 + 6 * 3) * 3];
  double residuals[3];
  double* jacobians[7] = {
      jacobian + 0,
      jacobian + 10 * 3,
      jacobian + 13 * 3,
      jacobian + 16 * 3,
      jacobian + 19 * 3,
      jacobian + 22 * 3,
      jacobian + 25 * 3,
  };

  std::unique_ptr<ceres::CostFunction> cost_function(
      new ceres::AutoDiffCostFunction<FunctorType, 3, 10, 3, 3, 3, 3, 3, 3>(
          new FunctorType));

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK(BM_BrdfAutoDiff)->Arg(0)->Arg(1);

}  // namespace ceres

BENCHMARK_MAIN();
