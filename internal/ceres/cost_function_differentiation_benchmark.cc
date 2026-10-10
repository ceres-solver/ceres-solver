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
// Authors: darius.rueckert@fau.de (Darius Rueckert)
//          nikolaus@nikolaus-demmel.de (Nikolaus Demmel)
//          sameeragarwal@google.com (Sameer Agarwal)

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/constant_cost_function.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/photometric_error.h"
#include "ceres/autodiff_benchmarks/relative_pose_error.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace ceres {
namespace {

enum DiffType {
  kAutoDiff,
  kDynamicAutoDiff,
  kAutoDiffDynamicResiduals,
  kNumericForward,
  kDynamicNumericForward,
  kNumericCentral,
  kDynamicNumericCentral,
  kNumericRidders,
  kDynamicNumericRidders,
};

// Controls which outputs CostFunction::Evaluate is called to compute:
//   kResidualsOnly:             evaluate residuals only (jacobians = nullptr)
//   kResidualsAndJacobians:     evaluate residuals and all parameter Jacobians
//   kResidualsAndPointJacobian: evaluate residuals and only the 3D point
//                               Jacobian (holding camera parameters constant)
enum EvaluationType {
  kResidualsOnly,
  kResidualsAndJacobians,
  kResidualsAndPointJacobian,
};

// Transforms a static functor into a dynamic one.
template <typename CostFunctionType, int kNumParameterBlocks>
class ToDynamic {
 public:
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<CostFunctionType, Args&&...>>>
  explicit ToDynamic(Args&&... args)
      : cost_function_(std::forward<Args>(args)...) {}

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

// Creates a CostFunction wrapping `CostFunctor` with `kNumResiduals` residuals
// and parameter blocks of sizes `Ns...` using the differentiation wrapper
// specified by `kDiffType`.
template <DiffType kDiffType>
struct CostFunctionFactory {
  template <typename CostFunctor,
            int kNumResiduals,
            int... Ns,
            typename... Args>
  static std::unique_ptr<CostFunction> Create(Args&&... args) {
    constexpr int kNumParameterBlocks = sizeof...(Ns);
    using DynamicFunctor = ToDynamic<CostFunctor, kNumParameterBlocks>;

    if constexpr (kDiffType == kAutoDiff) {
      return std::make_unique<
          AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kAutoDiffDynamicResiduals) {
      return std::make_unique<
          AutoDiffCostFunction<CostFunctor, DYNAMIC, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...),
          kNumResiduals);
    } else if constexpr (kDiffType == kDynamicAutoDiff) {
      auto dynamic_function =
          std::make_unique<DynamicAutoDiffCostFunction<DynamicFunctor>>(
              std::make_unique<DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericForward) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, FORWARD, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericForward) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, FORWARD>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericCentral) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, CENTRAL, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericCentral) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, CENTRAL>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    } else if constexpr (kDiffType == kNumericRidders) {
      return std::make_unique<
          NumericDiffCostFunction<CostFunctor, RIDDERS, kNumResiduals, Ns...>>(
          std::make_unique<CostFunctor>(std::forward<Args>(args)...));
    } else if constexpr (kDiffType == kDynamicNumericRidders) {
      auto dynamic_function = std::make_unique<
          DynamicNumericDiffCostFunction<DynamicFunctor, RIDDERS>>(
          std::make_unique<const DynamicFunctor>(std::forward<Args>(args)...));
      (dynamic_function->AddParameterBlock(Ns), ...);
      dynamic_function->SetNumResiduals(kNumResiduals);
      return dynamic_function;
    }
  }
};

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

 private:
  const double x_;
  const double y_;
};

// ============================================================================
// CostFunction Differentiation Benchmarks
// ============================================================================

template <int kParameterBlockSize, DiffType kDiffType>
void BM_Constant(benchmark::State& state) {
  constexpr int kNumResiduals = 1;
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  double* parameters[] = {parameters_values.data()};

  std::array<double, kNumResiduals> residuals{};
  std::array<double, kNumResiduals * kParameterBlockSize> jacobian_values{};
  double* jacobians[] = {jacobian_values.data()};

  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<ConstantCostFunction<kParameterBlockSize>,
                                  1,
                                  kParameterBlockSize>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals.data(), jacobians));
  }
}

#define REGISTER_CONSTANT_BENCHMARKS(N)          \
  BENCHMARK_TEMPLATE(BM_Constant, N, kAutoDiff); \
  BENCHMARK_TEMPLATE(BM_Constant, N, kDynamicAutoDiff)

REGISTER_CONSTANT_BENCHMARKS(1);
REGISTER_CONSTANT_BENCHMARKS(10);
REGISTER_CONSTANT_BENCHMARKS(20);
REGISTER_CONSTANT_BENCHMARKS(30);
REGISTER_CONSTANT_BENCHMARKS(40);
REGISTER_CONSTANT_BENCHMARKS(50);
REGISTER_CONSTANT_BENCHMARKS(60);

#undef REGISTER_CONSTANT_BENCHMARKS

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Linear1(benchmark::State& state) {
  double parameter_block1[] = {1.};
  double* parameters[] = {parameter_block1};

  double jacobian1[1];
  double residuals[1];
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Linear1CostFunction,
                                                      1,
                                                      1>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Linear10(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  double* parameters[] = {parameter_block1};

  double jacobian1[10 * 10];
  double residuals[10];
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Linear10CostFunction,
                                                      10,
                                                      10>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Rat43(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4.};
  double* parameters[] = {parameter_block1};

  double jacobian1[] = {0.0, 0.0, 0.0, 0.0};
  double residuals = 0.0;
  double* jacobians[] = {jacobian1};
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<Rat43CostFunctor, 1, 4>(
          x, y);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, &residuals, jacobians_ptr));
  }
}

#define REGISTER_RAT43_BENCHMARKS(Diff)                        \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsOnly);          \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsAndJacobians)

REGISTER_RAT43_BENCHMARKS(kAutoDiff);
REGISTER_RAT43_BENCHMARKS(kDynamicAutoDiff);
REGISTER_RAT43_BENCHMARKS(kNumericForward);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericForward);
REGISTER_RAT43_BENCHMARKS(kNumericCentral);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericCentral);
REGISTER_RAT43_BENCHMARKS(kNumericRidders);
REGISTER_RAT43_BENCHMARKS(kDynamicNumericRidders);

#undef REGISTER_RAT43_BENCHMARKS

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_SnavelyReprojection(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double parameter_block2[] = {1., 2., 3.};
  double* parameters[] = {parameter_block1, parameter_block2};

  double jacobian1[2 * 9];
  double jacobian2[2 * 3];
  double residuals[2];
  double* jacobians[] = {
      (kEvalType == kResidualsAndJacobians) ? jacobian1 : nullptr,
      (kEvalType != kResidualsOnly) ? jacobian2 : nullptr,
  };
  double** jacobians_ptr =
      (kEvalType == kResidualsOnly) ? nullptr : jacobians;

  const double x = 0.2;
  const double y = 0.3;
  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<SnavelyReprojectionError, 2, 9, 3>(x, y);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

#define REGISTER_SNAVELY_ALL_MODES(Diff)                                      \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsOnly);           \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsAndJacobians);   \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsAndPointJacobian)

REGISTER_SNAVELY_ALL_MODES(kAutoDiff);
REGISTER_SNAVELY_ALL_MODES(kDynamicAutoDiff);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kAutoDiffDynamicResiduals,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kAutoDiffDynamicResiduals,
                   kResidualsAndJacobians);
REGISTER_SNAVELY_ALL_MODES(kNumericForward);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericForward);
REGISTER_SNAVELY_ALL_MODES(kNumericCentral);
REGISTER_SNAVELY_ALL_MODES(kDynamicNumericCentral);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection, kNumericRidders, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kNumericRidders,
                   kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kDynamicNumericRidders,
                   kResidualsOnly);
BENCHMARK_TEMPLATE(BM_SnavelyReprojection,
                   kDynamicNumericRidders,
                   kResidualsAndJacobians);

#undef REGISTER_SNAVELY_ALL_MODES

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Photometric(benchmark::State& state) {
  constexpr int PATCH_SIZE = 8;

  using FunctorType = PhotometricError<PATCH_SIZE>;
  using ImageType = Eigen::Matrix<uint8_t, 128, 128, Eigen::RowMajor>;

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
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::mt19937::result_type seed = 42;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  std::uniform_int_distribution<unsigned int> uniform0255(0, 255);

  FunctorType::Patch<double> intensities_host =
      FunctorType::Patch<double>::NullaryExpr(
          [&]() { return uniform0255(gen); });

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

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<FunctorType,
                                                      FunctorType::PATCH_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POINT_SIZE>(
          intensities_host, bearings_host, image_target, intrinsics);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_RelativePose(benchmark::State& state) {
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
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  Eigen::Quaterniond q_i_j = Eigen::Quaterniond(1, 2, 3, 4).normalized();
  Eigen::Vector3d t_i_j(1, 2, 3);

  std::unique_ptr<CostFunction> cost_function =
      CostFunctionFactory<kDiffType>::template Create<FunctorType, 6, 7, 7>(
          q_i_j, t_i_j);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_RelativePose, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose,
                   kDynamicNumericCentral,
                   kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
void BM_Brdf(benchmark::State& state) {
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
  double** jacobians_ptr =
      (kEvalType == kResidualsAndJacobians) ? jacobians : nullptr;

  std::unique_ptr<CostFunction> cost_function = CostFunctionFactory<
      kDiffType>::template Create<FunctorType, 3, 10, 3, 3, 3, 3, 3, 3>();

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters, residuals, jacobians_ptr));
  }
}

BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsAndJacobians);

// ============================================================================
// CostFunctionToFunctor / DynamicCostFunctionToFunctor Benchmarks
// ============================================================================

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

template <DiffType kDiffType>
struct OuterProjectionFunctor {
  OuterProjectionFunctor()
      : inner_(std::make_unique<
               AutoDiffCostFunction<InnerProjectionFunctor, 2, 3, 3>>(
            std::make_unique<InnerProjectionFunctor>())) {}

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
    if constexpr (kDiffType == kAutoDiff) {
      return inner_(intrinsics, p, residuals);
    } else {
      const T* params[2] = {intrinsics, p};
      return inner_(params, residuals);
    }
  }

  std::conditional_t<kDiffType == kAutoDiff,
                     CostFunctionToFunctor<2, 3, 3>,
                     DynamicCostFunctionToFunctor>
      inner_;
};

template <DiffType kDiffType>
void BM_CostFunctionToFunctor(benchmark::State& state) {
  std::unique_ptr<CostFunction> cost_function = std::make_unique<
      AutoDiffCostFunction<OuterProjectionFunctor<kDiffType>, 2, 3, 3, 3, 3>>(
      std::make_unique<OuterProjectionFunctor<kDiffType>>());
  double rot[3] = {0.1, -0.2, 0.05};
  double trans[3] = {0.5, -0.1, 2.0};
  double intr[3] = {500.0, -0.01, 0.001};
  double pt[3] = {0.3, -0.4, 5.0};
  const double* params[4] = {rot, trans, intr, pt};
  double residuals[2];
  double jacobian[4 * 2 * 3];
  double* jacobians[4] = {
      jacobian + 0, jacobian + 6, jacobian + 12, jacobian + 18};

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(params, residuals, jacobians));
  }
}

BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kDynamicAutoDiff);

}  // namespace
}  // namespace ceres

BENCHMARK_MAIN();
