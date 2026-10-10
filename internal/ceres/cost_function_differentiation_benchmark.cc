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
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"
#include "benchmark/benchmark.h"
#include "ceres/benchmark_cost_functions.h"
#include "ceres/ceres.h"

namespace ceres {

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
//                               Jacobian (at index CostFunctor::kPointBlockIndex)
enum EvaluationType {
  kResidualsOnly,
  kResidualsAndJacobians,
  kResidualsAndPointJacobian,
};

// Adapts a fixed-arity cost functor `CostFunctionType(p0, p1, ..., residuals)`
// to the dynamic-arity signature `operator()(const T* const* parameters, T*
// residuals)` expected by DynamicAutoDiffCostFunction and
// DynamicNumericDiffCostFunction.
template <typename CostFunctionType, int kNumParameterBlocks>
class ToDynamic {
 public:
  template <typename... Args,
            typename = std::enable_if_t<
                std::is_constructible_v<CostFunctionType, Args&&...>>>
  explicit ToDynamic(Args&&... args)
      : cost_function_(std::forward<Args>(args)...) {}

  template <typename T>
  EIGEN_STRONG_INLINE bool operator()(const T* const* parameters,
                                      T* residuals) const {
    return Apply(
        parameters, residuals, std::make_index_sequence<kNumParameterBlocks>());
  }

 private:
  template <typename T, size_t... Indices>
  EIGEN_STRONG_INLINE bool Apply(const T* const* parameters,
                                 T* residuals,
                                 std::index_sequence<Indices...>) const {
    return cost_function_(parameters[Indices]..., residuals);
  }

  CostFunctionType cost_function_;
};

// Creates a CostFunction wrapping `CostFunctor` with `kNumResiduals` residuals
// and parameter blocks of sizes `Ns...` using the differentiation wrapper
// specified by `kDiffType`.
template <DiffType kDiffType,
          typename CostFunctor,
          int kNumResiduals,
          int... Ns,
          typename... Args>
std::unique_ptr<CostFunction> CreateCostFunction(Args&&... args) {
  constexpr int kNumParameterBlocks = sizeof...(Ns);
  using DynamicFunctor = ToDynamic<CostFunctor, kNumParameterBlocks>;

  auto configure_dynamic = [](auto cost_function) {
    (cost_function->AddParameterBlock(Ns), ...);
    cost_function->SetNumResiduals(kNumResiduals);
    return cost_function;
  };

  if constexpr (kDiffType == kAutoDiff) {
    return std::make_unique<
        AutoDiffCostFunction<CostFunctor, kNumResiduals, Ns...>>(
        std::make_unique<CostFunctor>(std::forward<Args>(args)...));
  } else if constexpr (kDiffType == kAutoDiffDynamicResiduals) {
    return std::make_unique<AutoDiffCostFunction<CostFunctor, DYNAMIC, Ns...>>(
        std::make_unique<CostFunctor>(std::forward<Args>(args)...),
        kNumResiduals);
  } else if constexpr (kDiffType == kDynamicAutoDiff) {
    return configure_dynamic(
        std::make_unique<DynamicAutoDiffCostFunction<DynamicFunctor>>(
            std::make_unique<DynamicFunctor>(std::forward<Args>(args)...)));
  } else if constexpr (kDiffType == kNumericForward) {
    return std::make_unique<
        NumericDiffCostFunction<CostFunctor, FORWARD, kNumResiduals, Ns...>>(
        std::make_unique<CostFunctor>(std::forward<Args>(args)...));
  } else if constexpr (kDiffType == kDynamicNumericForward) {
    return configure_dynamic(
        std::make_unique<
            DynamicNumericDiffCostFunction<DynamicFunctor, FORWARD>>(
            std::make_unique<const DynamicFunctor>(
                std::forward<Args>(args)...)));
  } else if constexpr (kDiffType == kNumericCentral) {
    return std::make_unique<
        NumericDiffCostFunction<CostFunctor, CENTRAL, kNumResiduals, Ns...>>(
        std::make_unique<CostFunctor>(std::forward<Args>(args)...));
  } else if constexpr (kDiffType == kDynamicNumericCentral) {
    return configure_dynamic(
        std::make_unique<
            DynamicNumericDiffCostFunction<DynamicFunctor, CENTRAL>>(
            std::make_unique<const DynamicFunctor>(
                std::forward<Args>(args)...)));
  } else if constexpr (kDiffType == kNumericRidders) {
    return std::make_unique<
        NumericDiffCostFunction<CostFunctor, RIDDERS, kNumResiduals, Ns...>>(
        std::make_unique<CostFunctor>(std::forward<Args>(args)...));
  } else if constexpr (kDiffType == kDynamicNumericRidders) {
    return configure_dynamic(
        std::make_unique<
            DynamicNumericDiffCostFunction<DynamicFunctor, RIDDERS>>(
            std::make_unique<const DynamicFunctor>(
                std::forward<Args>(args)...)));
  }
}

// Constructs a CostFunction via CreateCostFunction<kDiffType, CostFunctor,
// kNumResiduals, Ns...>(args...), allocates stack buffers for residuals and
// Jacobians, verifies that Evaluate succeeds on the input parameters, and
// benchmarks CostFunction::Evaluate.
template <DiffType kDiffType,
          EvaluationType kEvalType,
          typename CostFunctor,
          int kNumResiduals,
          int... Ns,
          typename... Args>
void RunCostFunctionBenchmark(
    benchmark::State& state,
    const std::array<const double*, sizeof...(Ns)>& parameters,
    Args&&... args) {
  const std::unique_ptr<CostFunction> cost_function =
      CreateCostFunction<kDiffType, CostFunctor, kNumResiduals, Ns...>(
          std::forward<Args>(args)...);

  constexpr int kNumParameterBlocks = sizeof...(Ns);
  constexpr int kTotalParameters = (Ns + ...);
  double residuals[kNumResiduals] = {};
  double jacobian_storage[kNumResiduals * kTotalParameters] = {};
  double* jacobians[kNumParameterBlocks] = {};

  constexpr int kBlockSizes[kNumParameterBlocks] = {Ns...};
  double* cursor = jacobian_storage;
  for (int i = 0; i < kNumParameterBlocks; ++i) {
    if constexpr (kEvalType == kResidualsAndJacobians) {
      jacobians[i] = cursor;
    } else if constexpr (kEvalType == kResidualsAndPointJacobian) {
      static_assert(CostFunctor::kPointBlockIndex >= 0 &&
                    CostFunctor::kPointBlockIndex < kNumParameterBlocks);
      if (i == CostFunctor::kPointBlockIndex) {
        jacobians[i] = cursor;
      }
    }
    cursor += kNumResiduals * kBlockSizes[i];
  }
  double** jacobians_ptr = (kEvalType == kResidualsOnly) ? nullptr : jacobians;

  CHECK(cost_function->Evaluate(parameters.data(), residuals, jacobians_ptr));

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        cost_function->Evaluate(parameters.data(), residuals, jacobians_ptr));
    benchmark::DoNotOptimize(residuals);
    benchmark::DoNotOptimize(jacobian_storage);
  }
}

#define REGISTER_AUTODIFF_BENCHMARKS(Fn)                           \
  BENCHMARK_TEMPLATE(Fn, kAutoDiff, kResidualsOnly);               \
  BENCHMARK_TEMPLATE(Fn, kAutoDiff, kResidualsAndJacobians);       \
  BENCHMARK_TEMPLATE(Fn, kDynamicAutoDiff, kResidualsOnly);        \
  BENCHMARK_TEMPLATE(Fn, kDynamicAutoDiff, kResidualsAndJacobians)

template <int kParameterBlockSize, DiffType kDiffType>
static void BM_Constant(benchmark::State& state) {
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  RunCostFunctionBenchmark<kDiffType,
                           kResidualsAndJacobians,
                           ConstantCostFunction<kParameterBlockSize>,
                           1,
                           kParameterBlockSize>(state,
                                                {parameters_values.data()});
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
static void BM_Linear1(benchmark::State& state) {
  const double parameter_block1[] = {1.};
  RunCostFunctionBenchmark<kDiffType, kEvalType, Linear1CostFunction, 1, 1>(
      state, {parameter_block1});
}

REGISTER_AUTODIFF_BENCHMARKS(BM_Linear1);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Linear10(benchmark::State& state) {
  const double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  RunCostFunctionBenchmark<kDiffType, kEvalType, Linear10CostFunction, 10, 10>(
      state, {parameter_block1});
}

REGISTER_AUTODIFF_BENCHMARKS(BM_Linear10);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Rat43(benchmark::State& state) {
  const double parameter_block1[] = {1., 2., 3., 4.};
  RunCostFunctionBenchmark<kDiffType, kEvalType, Rat43CostFunctor, 1, 4>(
      state, {parameter_block1}, 0.2, 0.3);
}

#define REGISTER_RAT43_BENCHMARKS(Diff)               \
  BENCHMARK_TEMPLATE(BM_Rat43, Diff, kResidualsOnly); \
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
static void BM_SnavelyReprojection(benchmark::State& state) {
  const double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  const double parameter_block2[] = {1., 2., 3.};
  RunCostFunctionBenchmark<kDiffType,
                           kEvalType,
                           SnavelyReprojectionError,
                           2,
                           9,
                           3>(
      state, {parameter_block1, parameter_block2}, 0.2, 0.3);
}

#define REGISTER_SNAVELY_ALL_MODES(Diff)                                    \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsOnly);         \
  BENCHMARK_TEMPLATE(BM_SnavelyReprojection, Diff, kResidualsAndJacobians); \
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
static void BM_Photometric(benchmark::State& state) {
  constexpr int PATCH_SIZE = 8;
  using FunctorType = PhotometricError<PATCH_SIZE>;
  using ImageType = Eigen::Matrix<uint8_t, 128, 128, Eigen::RowMajor>;

  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  const double parameter_block3[] = {1.};
  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  std::mt19937 gen(42);
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

  RunCostFunctionBenchmark<kDiffType,
                           kEvalType,
                           FunctorType,
                           FunctorType::PATCH_SIZE,
                           FunctorType::POSE_SIZE,
                           FunctorType::POSE_SIZE,
                           FunctorType::POINT_SIZE>(
      state,
      {parameter_block1, parameter_block2, parameter_block3},
      intensities_host,
      bearings_host,
      image_target,
      intrinsics);
}

REGISTER_AUTODIFF_BENCHMARKS(BM_Photometric);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_RelativePose(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  const Eigen::Quaterniond q_i_j = Eigen::Quaterniond(1, 2, 3, 4).normalized();
  const Eigen::Vector3d t_i_j(1, 2, 3);

  RunCostFunctionBenchmark<kDiffType, kEvalType, RelativePoseError, 6, 7, 7>(
      state, {parameter_block1, parameter_block2}, q_i_j, t_i_j);
}

REGISTER_AUTODIFF_BENCHMARKS(BM_RelativePose);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose, kNumericCentral, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_RelativePose, kDynamicNumericCentral, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_RelativePose,
                   kDynamicNumericCentral,
                   kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Brdf(benchmark::State& state) {
  const double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  const Eigen::Vector3d c(0.1, 0.2, 0.3);
  const Eigen::Vector3d n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  const Eigen::Vector3d v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  const Eigen::Vector3d l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  const Eigen::Vector3d y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  RunCostFunctionBenchmark<kDiffType,
                           kEvalType,
                           Brdf,
                           3,
                           10,
                           3,
                           3,
                           3,
                           3,
                           3,
                           3>(
      state,
      {material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()});
}

REGISTER_AUTODIFF_BENCHMARKS(BM_Brdf);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_CostFunctionToFunctor(benchmark::State& state) {
  using Functor = OuterProjectionFunctor<kDiffType == kDynamicAutoDiff>;
  const double rot[3] = {0.1, -0.2, 0.05};
  const double trans[3] = {0.5, -0.1, 2.0};
  const double intr[3] = {500.0, -0.01, 0.001};
  const double pt[3] = {0.3, -0.4, 5.0};
  RunCostFunctionBenchmark<kAutoDiff, kEvalType, Functor, 2, 3, 3, 3, 3>(
      state, {rot, trans, intr, pt});
}

REGISTER_AUTODIFF_BENCHMARKS(BM_CostFunctionToFunctor);

#undef REGISTER_AUTODIFF_BENCHMARKS

}  // namespace ceres

BENCHMARK_MAIN();
