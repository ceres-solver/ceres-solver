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
#include <utility>

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/constant_cost_function.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/photometric_error.h"
#include "ceres/autodiff_benchmarks/relative_pose_error.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"
#include "ceres/ceres.h"

namespace ceres {

enum Dynamic { kNotDynamic, kDynamic };

// Transforms a static functor into a dynamic one.
template <typename CostFunctionType, int kNumParameterBlocks>
class ToDynamic {
 public:
  template <typename... _Args,
            typename = std::enable_if_t<
                std::is_constructible_v<CostFunctionType, _Args&&...>>>
  explicit ToDynamic(_Args&&... __args)
      : cost_function_(std::forward<_Args>(__args)...) {}

  template <typename T>
  bool operator()(const T* const* parameters, T* residuals) const {
    return Apply(
        parameters, residuals, std::make_index_sequence<kNumParameterBlocks>());
  }

 private:
  template <typename T, size_t... Indices>
  bool Apply(const T* const* parameters,
             T* residuals,
             std::index_sequence<Indices...>) const {
    return cost_function_(parameters[Indices]..., residuals);
  }

  CostFunctionType cost_function_;
};

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
      new AnalyticConstantCostFunction<kParameterBlockSize>());

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals.data(), jacobians);
  }
}

// Helpers for CostFunctionFactory.
template <typename DynamicCostFunctionType>
void AddParameterBlocks(DynamicCostFunctionType*) {}

template <int HeadN, int... TailNs, typename DynamicCostFunctionType>
void AddParameterBlocks(DynamicCostFunctionType* dynamic_function) {
  dynamic_function->AddParameterBlock(HeadN);
  AddParameterBlocks<TailNs...>(dynamic_function);
}

// Creates an autodiff cost function wrapping `CostFunctor`, with
// `kNumResiduals` residuals and parameter blocks with sized `Ns..`.
// Depending on `kIsDynamic`, either a static or dynamic cost function is
// created.
// `args` are forwarded to the `CostFunctor` constructor.
template <Dynamic kIsDynamic>
struct CostFunctionFactory {};

template <>
struct CostFunctionFactory<kNotDynamic> {
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Args>
  static std::unique_ptr<ceres::CostFunction> Create(Args&&... args) {
    return std::make_unique<
        ceres::AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>>(
        new CostFunctor(std::forward<Args>(args)...));
  }
};

template <>
struct CostFunctionFactory<kDynamic> {
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Args>
  static std::unique_ptr<ceres::CostFunction> Create(Args&&... args) {
    constexpr const int kNumParameterBlocks = sizeof...(Ns);
    auto dynamic_function = std::make_unique<ceres::DynamicAutoDiffCostFunction<
        ToDynamic<CostFunctor, kNumParameterBlocks>>>(
        new ToDynamic<CostFunctor, kNumParameterBlocks>(
            std::forward<Args>(args)...));
    dynamic_function->SetNumResiduals(kNumResiduals);
    AddParameterBlocks<Ns...>(dynamic_function.get());
    return dynamic_function;
  }
};

template <int kParameterBlockSize, Dynamic kIsDynamic>
static void BM_ConstantAutodiff(benchmark::State& state) {
  constexpr int num_residuals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, num_residuals> residuals;

  std::array<double, num_residuals * kParameterBlockSize> jacobian_values;
  double* jacobians[] = {jacobian_values.data()};

  std::unique_ptr<ceres::CostFunction> cost_function = CostFunctionFactory<
      kIsDynamic>::template Create<ConstantCostFunction<kParameterBlockSize>,
                                   1,
                                   kParameterBlockSize>();

  for (auto _ : state) {
    cost_function->Evaluate(parameters, residuals.data(), jacobians);
  }
}

BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 1);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 1, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 1, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 10);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 10, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 10, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 20);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 20, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 20, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 30);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 30, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 30, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 40);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 40, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 40, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 50);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 50, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 50, kDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAnalytic, 60);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 60, kNotDynamic);
BENCHMARK_TEMPLATE(BM_ConstantAutodiff, 60, kDynamic);

template <Dynamic kIsDynamic>
static void BM_Linear1AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function = CostFunctionFactory<
      kIsDynamic>::template Create<Linear1CostFunction, 1, 1>();

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK_TEMPLATE(BM_Linear1AutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_Linear1AutoDiff, kDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
static void BM_Linear10AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};

  std::unique_ptr<ceres::CostFunction> cost_function = CostFunctionFactory<
      kIsDynamic>::template Create<Linear10CostFunction, 10, 10>();

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK_TEMPLATE(BM_Linear10AutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_Linear10AutoDiff, kDynamic)->Arg(0)->Arg(1);

// From the NIST problem collection.
struct Rat43CostFunctor {
  Rat43CostFunctor(const double x, const double y) : x_(x), y_(y) {}

  template <typename T>
  inline bool operator()(const T* parameters, T* residuals) const {
    const T& b1 = parameters[0];
    const T& b2 = parameters[1];
    const T& b3 = parameters[2];
    const T& b4 = parameters[3];
    residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
    return true;
  }

  static constexpr int kNumParameterBlocks = 1;

 private:
  const double x_;
  const double y_;
};

template <Dynamic kIsDynamic>
static void BM_Rat43AutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals;
  double* jacobians[] = {jacobian1};
  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function =
      CostFunctionFactory<kIsDynamic>::template Create<Rat43CostFunctor, 1, 4>(
          x, y);

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, &residuals, state.range(0) ? jacobians : nullptr);
  }
}
BENCHMARK_TEMPLATE(BM_Rat43AutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_Rat43AutoDiff, kDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
static void BM_SnavelyReprojectionAutoDiff(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {jacobian1, jacobian2};

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<ceres::CostFunction> cost_function = CostFunctionFactory<
      kIsDynamic>::template Create<SnavelyReprojectionError, 2, 9, 3>(x, y);

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK_TEMPLATE(BM_SnavelyReprojectionAutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionAutoDiff, kDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
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

  std::unique_ptr<ceres::CostFunction> cost_function =
      CostFunctionFactory<kIsDynamic>::template Create<FunctorType,
                                                       FunctorType::PATCH_SIZE,
                                                       FunctorType::POSE_SIZE,
                                                       FunctorType::POSE_SIZE,
                                                       FunctorType::POINT_SIZE>(
          intensities_host, bearings_host, image_target, intrinsics);

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK_TEMPLATE(BM_PhotometricAutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_PhotometricAutoDiff, kDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
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

  std::unique_ptr<ceres::CostFunction> cost_function =
      CostFunctionFactory<kIsDynamic>::template Create<FunctorType, 6, 7, 7>(
          q_i_j, t_i_j);

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK_TEMPLATE(BM_RelativePoseAutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_RelativePoseAutoDiff, kDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
static void BM_BrdfAutoDiff(benchmark::State& state) {
  using FunctorType = Brdf;

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
  // clang-format off
  double* jacobians[7] = {
      jacobian + 0,      jacobian + 10 * 3, jacobian + 13 * 3,
      jacobian + 16 * 3, jacobian + 19 * 3, jacobian + 22 * 3,
      jacobian + 25 * 3,
  };
  // clang-format on

  std::unique_ptr<ceres::CostFunction> cost_function = CostFunctionFactory<
      kIsDynamic>::template Create<FunctorType, 3, 10, 3, 3, 3, 3, 3, 3>();

  for (auto _ : state) {
    cost_function->Evaluate(
        parameters, residuals, state.range(0) ? jacobians : nullptr);
  }
}

BENCHMARK_TEMPLATE(BM_BrdfAutoDiff, kNotDynamic)->Arg(0)->Arg(1);
BENCHMARK_TEMPLATE(BM_BrdfAutoDiff, kDynamic)->Arg(0)->Arg(1);

// ============================================================================
// Cold-Cache / Streaming Working-Set Benchmarks
// ============================================================================
// Instead of evaluating a single CostFunction on the same ~200 bytes of stack
// memory in a hot L1 loop, these benchmarks allocate a large pool of distinct
// CostFunction instances, shuffled parameter blocks, residual buffers, and
// Jacobian buffers exceeding 64 MB (larger than L1/L2/SLC cache) to simulate
// ProgramEvaluator streaming through thousands of ResidualBlocks.

constexpr size_t kColdWorkingSetBytes = 64 * 1024 * 1024;

template <Dynamic kIsDynamic>
static void BM_SnavelyReprojectionAutoDiff_Cold(benchmark::State& state) {
  // Each slot holds:
  // - 1 CostFunction (heap object + 2 doubles = ~64B)
  // - 9 camera doubles + 3 point doubles = 96B
  // - 2 residual doubles + 24 jacobian doubles = 208B
  // - pointer arrays (2 param ptrs + 2 jacobian ptrs = 32B)
  // Total ~400B per slot -> 131072 slots > 52 MB + heap overhead > 64 MB.
  constexpr size_t kNumSlots = 131072;
  constexpr size_t kMask = kNumSlots - 1;

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);

  std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions(kNumSlots);
  std::vector<std::array<double, 9>> cameras(kNumSlots);
  std::vector<std::array<double, 3>> points(kNumSlots);
  std::vector<std::array<double, 2>> residuals(kNumSlots);
  std::vector<std::array<double, 18>> jacobians_cam(kNumSlots);
  std::vector<std::array<double, 6>> jacobians_pt(kNumSlots);

  std::vector<size_t> cam_indices(kNumSlots);
  std::vector<size_t> pt_indices(kNumSlots);
  std::iota(cam_indices.begin(), cam_indices.end(), 0);
  std::iota(pt_indices.begin(), pt_indices.end(), 0);
  std::shuffle(cam_indices.begin(), cam_indices.end(), gen);
  std::shuffle(pt_indices.begin(), pt_indices.end(), gen);

  for (size_t i = 0; i < kNumSlots; ++i) {
    cost_functions[i] = CostFunctionFactory<kIsDynamic>::
        template Create<SnavelyReprojectionError, 2, 9, 3>(0.2 + dist(gen),
                                                           0.3 + dist(gen));
    cameras[i] = {0.1 + dist(gen),
                  0.2 + dist(gen),
                  0.3 + dist(gen),
                  0.4 + dist(gen),
                  0.5 + dist(gen),
                  2.5 + dist(gen),
                  500.0 + dist(gen),
                  0.01 + dist(gen) * 0.01,
                  0.001 + dist(gen) * 0.001};
    points[i] = {0.5 + dist(gen), -0.3 + dist(gen), -5.0 + dist(gen)};
  }

  size_t idx = 0;
  const bool compute_jacobians = state.range(0) != 0;
  for (auto _ : state) {
    const double* params[2] = {cameras[cam_indices[idx]].data(),
                               points[pt_indices[idx]].data()};
    double* jacs[2] = {jacobians_cam[idx].data(), jacobians_pt[idx].data()};
    cost_functions[idx]->Evaluate(
        params, residuals[idx].data(), compute_jacobians ? jacs : nullptr);
    idx = (idx + 1) & kMask;
  }
  benchmark::DoNotOptimize(residuals[0]);
  benchmark::DoNotOptimize(jacobians_cam[0]);
  benchmark::DoNotOptimize(jacobians_pt[0]);
}
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionAutoDiff_Cold, kNotDynamic)
    ->Arg(0)
    ->Arg(1);

template <Dynamic kIsDynamic>
static void BM_RelativePoseAutoDiff_Cold(benchmark::State& state) {
  using FunctorType = RelativePoseError;
  constexpr size_t kNumSlots = 65536;
  constexpr size_t kMask = kNumSlots - 1;

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-0.2, 0.2);

  std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions(kNumSlots);
  std::vector<std::array<double, 7>> poses_i(kNumSlots);
  std::vector<std::array<double, 7>> poses_j(kNumSlots);
  std::vector<std::array<double, 6>> residuals(kNumSlots);
  std::vector<std::array<double, 42>> jacobians_i(kNumSlots);
  std::vector<std::array<double, 42>> jacobians_j(kNumSlots);

  std::vector<size_t> idx_i(kNumSlots);
  std::vector<size_t> idx_j(kNumSlots);
  std::iota(idx_i.begin(), idx_i.end(), 0);
  std::iota(idx_j.begin(), idx_j.end(), 0);
  std::shuffle(idx_i.begin(), idx_i.end(), gen);
  std::shuffle(idx_j.begin(), idx_j.end(), gen);

  for (size_t k = 0; k < kNumSlots; ++k) {
    Eigen::Quaterniond q_i_j =
        Eigen::Quaterniond(1.0 + dist(gen), 2.0 + dist(gen), 3.0, 4.0)
            .normalized();
    Eigen::Vector3d t_i_j(1.0 + dist(gen), 2.0 + dist(gen), 3.0 + dist(gen));
    cost_functions[k] =
        CostFunctionFactory<kIsDynamic>::template Create<FunctorType, 6, 7, 7>(
            q_i_j, t_i_j);

    poses_i[k] = {1.0 + dist(gen), 2.0, 3.0, 4.0, 5.0 + dist(gen), 6.0, 7.0};
    poses_j[k] = {1.1 + dist(gen), 2.1, 3.1, 4.1, 5.1 + dist(gen), 6.1, 7.1};
    Eigen::Map<Eigen::Quaterniond>(poses_i[k].data()).normalize();
    Eigen::Map<Eigen::Quaterniond>(poses_j[k].data()).normalize();
  }

  size_t idx = 0;
  const bool compute_jacobians = state.range(0) != 0;
  for (auto _ : state) {
    const double* params[2] = {poses_i[idx_i[idx]].data(),
                               poses_j[idx_j[idx]].data()};
    double* jacs[2] = {jacobians_i[idx].data(), jacobians_j[idx].data()};
    cost_functions[idx]->Evaluate(
        params, residuals[idx].data(), compute_jacobians ? jacs : nullptr);
    idx = (idx + 1) & kMask;
  }
  benchmark::DoNotOptimize(residuals[0]);
  benchmark::DoNotOptimize(jacobians_i[0]);
}
BENCHMARK_TEMPLATE(BM_RelativePoseAutoDiff_Cold, kNotDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
static void BM_Rat43AutoDiff_Cold(benchmark::State& state) {
  constexpr size_t kNumSlots = 262144;
  constexpr size_t kMask = kNumSlots - 1;

  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-0.05, 0.05);

  std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions(kNumSlots);
  std::vector<std::array<double, 4>> params_pool(kNumSlots);
  std::vector<double> residuals(kNumSlots);
  std::vector<std::array<double, 4>> jacobians_pool(kNumSlots);

  for (size_t k = 0; k < kNumSlots; ++k) {
    cost_functions[k] = CostFunctionFactory<
        kIsDynamic>::template Create<Rat43CostFunctor, 1, 4>(0.2 + dist(gen),
                                                             0.3 + dist(gen));
    params_pool[k] = {
        1.0 + dist(gen), 2.0 + dist(gen), 3.0 + dist(gen), 4.0 + dist(gen)};
  }

  size_t idx = 0;
  const bool compute_jacobians = state.range(0) != 0;
  for (auto _ : state) {
    const double* params[1] = {params_pool[idx].data()};
    double* jacs[1] = {jacobians_pool[idx].data()};
    cost_functions[idx]->Evaluate(
        params, &residuals[idx], compute_jacobians ? jacs : nullptr);
    idx = (idx + 1) & kMask;
  }
  benchmark::DoNotOptimize(residuals[0]);
  benchmark::DoNotOptimize(jacobians_pool[0]);
}
BENCHMARK_TEMPLATE(BM_Rat43AutoDiff_Cold, kNotDynamic)->Arg(0)->Arg(1);

template <Dynamic kIsDynamic>
static void BM_Linear10AutoDiff_Cold(benchmark::State& state) {
  constexpr size_t kNumSlots = 65536;
  constexpr size_t kMask = kNumSlots - 1;

  std::vector<std::unique_ptr<ceres::CostFunction>> cost_functions(kNumSlots);
  std::vector<std::array<double, 10>> params_pool(kNumSlots);
  std::vector<std::array<double, 10>> residuals_pool(kNumSlots);
  std::vector<std::array<double, 100>> jacobians_pool(kNumSlots);

  for (size_t k = 0; k < kNumSlots; ++k) {
    cost_functions[k] = CostFunctionFactory<
        kIsDynamic>::template Create<Linear10CostFunction, 10, 10>();
    for (int i = 0; i < 10; ++i) {
      params_pool[k][i] = static_cast<double>(i + 1) + 0.001 * (k & 255);
    }
  }

  size_t idx = 0;
  const bool compute_jacobians = state.range(0) != 0;
  for (auto _ : state) {
    const double* params[1] = {params_pool[idx].data()};
    double* jacs[1] = {jacobians_pool[idx].data()};
    cost_functions[idx]->Evaluate(
        params, residuals_pool[idx].data(), compute_jacobians ? jacs : nullptr);
    idx = (idx + 1) & kMask;
  }
  benchmark::DoNotOptimize(residuals_pool[0]);
  benchmark::DoNotOptimize(jacobians_pool[0]);
}
BENCHMARK_TEMPLATE(BM_Linear10AutoDiff_Cold, kNotDynamic)->Arg(0)->Arg(1);

struct BenchmarkQuaternionFunctor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    T q_delta[4];
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }
    QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    T minus_x[4] = {x[0], -x[1], -x[2], -x[3]};
    T ambient_y_minus_x[4];
    QuaternionProduct(y, minus_x, ambient_y_minus_x);
    const T u_sq = ambient_y_minus_x[1] * ambient_y_minus_x[1] +
                   ambient_y_minus_x[2] * ambient_y_minus_x[2] +
                   ambient_y_minus_x[3] * ambient_y_minus_x[3];
    if (u_sq > T(0.0)) {
      T u_norm = sqrt(u_sq);
      T theta = atan2(u_norm, ambient_y_minus_x[0]);
      y_minus_x[0] = theta * ambient_y_minus_x[1] / u_norm;
      y_minus_x[1] = theta * ambient_y_minus_x[2] / u_norm;
      y_minus_x[2] = theta * ambient_y_minus_x[3] / u_norm;
    } else {
      y_minus_x[0] = ambient_y_minus_x[1];
      y_minus_x[1] = ambient_y_minus_x[2];
      y_minus_x[2] = ambient_y_minus_x[3];
    }
    return true;
  }
};

struct BenchmarkPose3Functor {
  template <typename T>
  bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
    BenchmarkQuaternionFunctor q;
    q.Plus(x, delta, x_plus_delta);
    x_plus_delta[4] = x[4] + delta[3];
    x_plus_delta[5] = x[5] + delta[4];
    x_plus_delta[6] = x[6] + delta[5];
    return true;
  }

  template <typename T>
  bool Minus(const T* y, const T* x, T* y_minus_x) const {
    BenchmarkQuaternionFunctor q;
    q.Minus(y, x, y_minus_x);
    y_minus_x[3] = y[4] - x[4];
    y_minus_x[4] = y[5] - x[5];
    y_minus_x[5] = y[6] - x[6];
    return true;
  }
};

static void BM_AutoDiffManifoldQuaternion_PlusJacobian(
    benchmark::State& state) {
  AutoDiffManifold<BenchmarkQuaternionFunctor, 4, 3> manifold;
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double jacobian[12];
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    manifold.PlusJacobian(x, jacobian);
    benchmark::DoNotOptimize(jacobian);
  }
}
BENCHMARK(BM_AutoDiffManifoldQuaternion_PlusJacobian);

static void BM_AutoDiffManifoldQuaternion_MinusJacobian(
    benchmark::State& state) {
  AutoDiffManifold<BenchmarkQuaternionFunctor, 4, 3> manifold;
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double jacobian[12];
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    manifold.MinusJacobian(x, jacobian);
    benchmark::DoNotOptimize(jacobian);
  }
}
BENCHMARK(BM_AutoDiffManifoldQuaternion_MinusJacobian);

static void BM_AutoDiffManifoldPose3_PlusJacobian(benchmark::State& state) {
  AutoDiffManifold<BenchmarkPose3Functor, 7, 6> manifold;
  double x[7] = {0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0};
  double jacobian[42];
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    manifold.PlusJacobian(x, jacobian);
    benchmark::DoNotOptimize(jacobian);
  }
}
BENCHMARK(BM_AutoDiffManifoldPose3_PlusJacobian);

static void BM_AutoDiffManifoldPose3_MinusJacobian(benchmark::State& state) {
  AutoDiffManifold<BenchmarkPose3Functor, 7, 6> manifold;
  double x[7] = {0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0};
  double jacobian[42];
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    manifold.MinusJacobian(x, jacobian);
    benchmark::DoNotOptimize(jacobian);
  }
}
BENCHMARK(BM_AutoDiffManifoldPose3_MinusJacobian);

template <int kNumParameters>
struct RosenbrockFirstOrderFunctor {
  template <typename T>
  bool operator()(const T* const x, T* cost) const {
    T sum = T(0.0);
    for (int i = 0; i < kNumParameters - 1; ++i) {
      const T t1 = T(1.0) - x[i];
      const T t2 = x[i + 1] - x[i] * x[i];
      sum += t1 * t1 + T(100.0) * t2 * t2;
    }
    *cost = sum;
    return true;
  }
};

template <int kNumParameters>
static void BM_AutoDiffFirstOrderFunction(benchmark::State& state) {
  AutoDiffFirstOrderFunction<RosenbrockFirstOrderFunctor<kNumParameters>,
                             kNumParameters>
      f;
  std::array<double, kNumParameters> x;
  for (int i = 0; i < kNumParameters; ++i) {
    x[i] = 0.5 + 0.01 * i;
  }
  double cost = 0.0;
  std::array<double, kNumParameters> gradient;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    f.Evaluate(x.data(), &cost, gradient.data());
    benchmark::DoNotOptimize(cost);
    benchmark::DoNotOptimize(gradient);
  }
}
BENCHMARK_TEMPLATE(BM_AutoDiffFirstOrderFunction, 4);
BENCHMARK_TEMPLATE(BM_AutoDiffFirstOrderFunction, 10);
BENCHMARK_TEMPLATE(BM_AutoDiffFirstOrderFunction, 16);
BENCHMARK_TEMPLATE(BM_AutoDiffFirstOrderFunction, 24);

struct InnerProjectionFunctor {
  template <typename T>
  bool operator()(const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    const T xp = point[0] / point[2];
    const T yp = point[1] / point[2];
    const T r2 = xp * xp + yp * yp;
    const T distortion = T(1.0) + r2 * (intrinsics[1] + intrinsics[2] * r2);
    residuals[0] = intrinsics[0] * distortion * xp;
    residuals[1] = intrinsics[0] * distortion * yp;
    return true;
  }
};

struct OuterProjectionWithStaticFunctor {
  OuterProjectionWithStaticFunctor()
      : inner_(new AutoDiffCostFunction<InnerProjectionFunctor, 2, 3, 3>(
            new InnerProjectionFunctor())) {}

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    T p[3];
    AngleAxisRotatePoint(rotation, point, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];
    return inner_(intrinsics, p, residuals);
  }

  CostFunctionToFunctor<2, 3, 3> inner_;
};

struct OuterProjectionWithDynamicFunctor {
  OuterProjectionWithDynamicFunctor()
      : inner_(new AutoDiffCostFunction<InnerProjectionFunctor, 2, 3, 3>(
            new InnerProjectionFunctor())) {}

  template <typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  const T* const intrinsics,
                  const T* const point,
                  T* residuals) const {
    T p[3];
    AngleAxisRotatePoint(rotation, point, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];
    const T* params[2] = {intrinsics, p};
    return inner_(params, residuals);
  }

  DynamicCostFunctionToFunctor inner_;
};

static void BM_CostFunctionToFunctor(benchmark::State& state) {
  AutoDiffCostFunction<OuterProjectionWithStaticFunctor, 2, 3, 3, 3, 3>
      cost_function(new OuterProjectionWithStaticFunctor());
  double rot[3] = {0.1, -0.2, 0.05};
  double trans[3] = {0.5, -0.1, 2.0};
  double intr[3] = {500.0, -0.01, 0.001};
  double pt[3] = {0.3, -0.4, 5.0};
  const double* params[4] = {rot, trans, intr, pt};
  double residuals[2];
  double j0[6], j1[6], j2[6], j3[6];
  double* jacobians[4] = {j0, j1, j2, j3};
  for (auto _ : state) {
    benchmark::DoNotOptimize(rot);
    benchmark::DoNotOptimize(trans);
    benchmark::DoNotOptimize(intr);
    benchmark::DoNotOptimize(pt);
    cost_function.Evaluate(params, residuals, jacobians);
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(j0);
    benchmark::DoNotOptimize(j1);
    benchmark::DoNotOptimize(j2);
    benchmark::DoNotOptimize(j3);
  }
}
BENCHMARK(BM_CostFunctionToFunctor);

static void BM_DynamicCostFunctionToFunctor(benchmark::State& state) {
  AutoDiffCostFunction<OuterProjectionWithDynamicFunctor, 2, 3, 3, 3, 3>
      cost_function(new OuterProjectionWithDynamicFunctor());
  double rot[3] = {0.1, -0.2, 0.05};
  double trans[3] = {0.5, -0.1, 2.0};
  double intr[3] = {500.0, -0.01, 0.001};
  double pt[3] = {0.3, -0.4, 5.0};
  const double* params[4] = {rot, trans, intr, pt};
  double residuals[2];
  double j0[6], j1[6], j2[6], j3[6];
  double* jacobians[4] = {j0, j1, j2, j3};
  for (auto _ : state) {
    benchmark::DoNotOptimize(rot);
    benchmark::DoNotOptimize(trans);
    benchmark::DoNotOptimize(intr);
    benchmark::DoNotOptimize(pt);
    cost_function.Evaluate(params, residuals, jacobians);
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(j0);
    benchmark::DoNotOptimize(j1);
    benchmark::DoNotOptimize(j2);
    benchmark::DoNotOptimize(j3);
  }
}
BENCHMARK(BM_DynamicCostFunctionToFunctor);

template <NumericDiffMethodType kMethod, Dynamic kIsDynamic>
static void BM_SnavelyReprojectionNumericDiff(benchmark::State& state) {
  constexpr int kBlock0 = 9;
  constexpr int kBlock1 = 3;
  constexpr int kNumResiduals = 2;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double const* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[kNumResiduals * kBlock0];
  double jacobian2[kNumResiduals * kBlock1];
  double residuals[kNumResiduals];
  double* jacobians[] = {jacobian1, jacobian2};

  const double observed_x = -2.0;
  const double observed_y = 1.0;

  using FunctorType = ceres::SnavelyReprojectionError;

  std::unique_ptr<ceres::CostFunction> cost_function;
  if constexpr (kIsDynamic) {
    using DynamicFunctor = ToDynamic<FunctorType, 2>;
    auto dynamic_function = std::make_unique<
        ceres::DynamicNumericDiffCostFunction<DynamicFunctor, kMethod>>(
        std::make_unique<const DynamicFunctor>(observed_x, observed_y));
    dynamic_function->AddParameterBlock(kBlock0);
    dynamic_function->AddParameterBlock(kBlock1);
    dynamic_function->SetNumResiduals(kNumResiduals);
    cost_function = std::move(dynamic_function);
  } else {
    cost_function =
        std::make_unique<ceres::NumericDiffCostFunction<FunctorType,
                                                        kMethod,
                                                        kNumResiduals,
                                                        kBlock0,
                                                        kBlock1>>(
            new FunctorType(observed_x, observed_y));
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(parameter_block1);
    benchmark::DoNotOptimize(parameter_block2);
    cost_function->Evaluate(parameters, residuals, jacobians);
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(jacobian1);
    benchmark::DoNotOptimize(jacobian2);
  }
}
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, CENTRAL, kNotDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, CENTRAL, kDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, FORWARD, kNotDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, FORWARD, kDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, RIDDERS, kNotDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff, RIDDERS, kDynamic);

template <NumericDiffMethodType kMethod, Dynamic kIsDynamic>
static void BM_SnavelyReprojectionNumericDiff_ConstantCamera(
    benchmark::State& state) {
  constexpr int kBlock0 = 9;
  constexpr int kBlock1 = 3;
  constexpr int kNumResiduals = 2;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double const* parameters[] = {parameter_block1, parameter_block2};

  double jacobian2[kNumResiduals * kBlock1];
  double residuals[kNumResiduals];
  double* jacobians[] = {nullptr, jacobian2};

  const double observed_x = -2.0;
  const double observed_y = 1.0;

  using FunctorType = ceres::SnavelyReprojectionError;

  std::unique_ptr<ceres::CostFunction> cost_function;
  if constexpr (kIsDynamic) {
    using DynamicFunctor = ToDynamic<FunctorType, 2>;
    auto dynamic_function = std::make_unique<
        ceres::DynamicNumericDiffCostFunction<DynamicFunctor, kMethod>>(
        std::make_unique<const DynamicFunctor>(observed_x, observed_y));
    dynamic_function->AddParameterBlock(kBlock0);
    dynamic_function->AddParameterBlock(kBlock1);
    dynamic_function->SetNumResiduals(kNumResiduals);
    cost_function = std::move(dynamic_function);
  } else {
    cost_function =
        std::make_unique<ceres::NumericDiffCostFunction<FunctorType,
                                                        kMethod,
                                                        kNumResiduals,
                                                        kBlock0,
                                                        kBlock1>>(
            new FunctorType(observed_x, observed_y));
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(parameter_block1);
    benchmark::DoNotOptimize(parameter_block2);
    cost_function->Evaluate(parameters, residuals, jacobians);
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(jacobian2);
  }
}
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff_ConstantCamera,
                   CENTRAL,
                   kNotDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff_ConstantCamera,
                   CENTRAL,
                   kDynamic);

template <NumericDiffMethodType kMethod, Dynamic kIsDynamic>
static void BM_SnavelyReprojectionNumericDiff_Cold(benchmark::State& state) {
  constexpr int kBlock0 = 9;
  constexpr int kBlock1 = 3;
  constexpr int kNumResiduals = 2;
  constexpr size_t kWorkingSetBytes = 64 * 1024 * 1024;

  using FunctorType = ceres::SnavelyReprojectionError;
  struct alignas(64) ProblemInstance {
    std::unique_ptr<ceres::CostFunction> cost_function;
    double b0[kBlock0];
    double b1[kBlock1];
    double residuals[kNumResiduals];
    double j0[kNumResiduals * kBlock0];
    double j1[kNumResiduals * kBlock1];
  };

  const size_t num_instances =
      std::max<size_t>(1024, kWorkingSetBytes / sizeof(ProblemInstance));
  std::vector<ProblemInstance> instances(num_instances);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.5, 2.0);

  for (size_t i = 0; i < num_instances; ++i) {
    for (int k = 0; k < kBlock0; ++k) instances[i].b0[k] = dist(rng);
    for (int k = 0; k < kBlock1; ++k) instances[i].b1[k] = dist(rng);
    const double ox = dist(rng);
    const double oy = dist(rng);
    if constexpr (kIsDynamic) {
      using DynamicFunctor = ToDynamic<FunctorType, 2>;
      auto fn = std::make_unique<
          ceres::DynamicNumericDiffCostFunction<DynamicFunctor, kMethod>>(
          std::make_unique<const DynamicFunctor>(ox, oy));
      fn->AddParameterBlock(kBlock0);
      fn->AddParameterBlock(kBlock1);
      fn->SetNumResiduals(kNumResiduals);
      instances[i].cost_function = std::move(fn);
    } else {
      instances[i].cost_function =
          std::make_unique<ceres::NumericDiffCostFunction<FunctorType,
                                                          kMethod,
                                                          kNumResiduals,
                                                          kBlock0,
                                                          kBlock1>>(
              new FunctorType(ox, oy));
    }
  }

  std::vector<size_t> order(num_instances);
  std::iota(order.begin(), order.end(), 0);
  std::shuffle(order.begin(), order.end(), rng);

  size_t idx = 0;
  for (auto _ : state) {
    auto& inst = instances[order[idx]];
    idx = (idx + 1 == num_instances) ? 0 : idx + 1;
    double const* parameters[] = {inst.b0, inst.b1};
    double* jacobians[] = {inst.j0, inst.j1};
    inst.cost_function->Evaluate(parameters, inst.residuals, jacobians);
    benchmark::DoNotOptimize(inst.residuals);
    benchmark::DoNotOptimize(inst.j0);
    benchmark::DoNotOptimize(inst.j1);
  }
}
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff_Cold,
                   CENTRAL,
                   kNotDynamic);
BENCHMARK_TEMPLATE(BM_SnavelyReprojectionNumericDiff_Cold, CENTRAL, kDynamic);

static void BM_SnavelyReprojectionAutoDiff_DynamicResiduals(
    benchmark::State& state) {
  constexpr int kBlock0 = 9;
  constexpr int kBlock1 = 3;
  constexpr int kNumResiduals = 2;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double const* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[kNumResiduals * kBlock0];
  double jacobian2[kNumResiduals * kBlock1];
  double residuals[kNumResiduals];
  double* jacobians[] = {jacobian1, jacobian2};

  using FunctorType = ceres::SnavelyReprojectionError;
  AutoDiffCostFunction<FunctorType, DYNAMIC, kBlock0, kBlock1> cost_function(
      std::make_unique<FunctorType>(-2.0, 1.0), kNumResiduals);

  for (auto _ : state) {
    benchmark::DoNotOptimize(parameter_block1);
    benchmark::DoNotOptimize(parameter_block2);
    cost_function.Evaluate(parameters, residuals, jacobians);
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(jacobian1);
    benchmark::DoNotOptimize(jacobian2);
  }
}
BENCHMARK(BM_SnavelyReprojectionAutoDiff_DynamicResiduals);

template <NumericDiffMethodType kMethod, int kNumParameters, Dynamic kIsDynamic>
static void BM_NumericDiffFirstOrderFunction(benchmark::State& state) {
  using Functor = RosenbrockFirstOrderFunctor<kNumParameters>;
  std::unique_ptr<FirstOrderFunction> f;
  if constexpr (kIsDynamic) {
    f = std::make_unique<
        NumericDiffFirstOrderFunction<Functor, kMethod, DYNAMIC>>(
        std::make_unique<Functor>(), kNumParameters);
  } else {
    f = std::make_unique<
        NumericDiffFirstOrderFunction<Functor, kMethod, kNumParameters>>(
        std::make_unique<Functor>());
  }

  std::array<double, kNumParameters> x;
  for (int i = 0; i < kNumParameters; ++i) {
    x[i] = 0.5 + 0.01 * i;
  }
  double cost = 0.0;
  std::array<double, kNumParameters> gradient;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    f->Evaluate(x.data(), &cost, gradient.data());
    benchmark::DoNotOptimize(cost);
    benchmark::DoNotOptimize(gradient);
  }
}
BENCHMARK_TEMPLATE(BM_NumericDiffFirstOrderFunction, CENTRAL, 10, kNotDynamic);
BENCHMARK_TEMPLATE(BM_NumericDiffFirstOrderFunction, CENTRAL, 10, kDynamic);
BENCHMARK_TEMPLATE(BM_NumericDiffFirstOrderFunction, CENTRAL, 24, kNotDynamic);
BENCHMARK_TEMPLATE(BM_NumericDiffFirstOrderFunction, CENTRAL, 24, kDynamic);

}  // namespace ceres

BENCHMARK_MAIN();
