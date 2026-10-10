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

#include "benchmark/benchmark.h"
#include "ceres/autodiff_benchmarks/brdf_cost_function.h"
#include "ceres/autodiff_benchmarks/constant_cost_function.h"
#include "ceres/autodiff_benchmarks/cost_function_benchmark_utils.h"
#include "ceres/autodiff_benchmarks/cost_function_to_functor_benchmark.h"
#include "ceres/autodiff_benchmarks/linear_cost_functions.h"
#include "ceres/autodiff_benchmarks/photometric_error.h"
#include "ceres/autodiff_benchmarks/rat43_cost_function.h"
#include "ceres/autodiff_benchmarks/relative_pose_error.h"
#include "ceres/autodiff_benchmarks/snavely_reprojection_error.h"

namespace ceres {

template <int kParameterBlockSize, DiffType kDiffType>
static void BM_Constant(benchmark::State& state) {
  std::array<double, kParameterBlockSize> parameters_values;
  std::iota(parameters_values.begin(), parameters_values.end(), 0);
  auto cf = CostFunctionFactory<kDiffType>::template Create<
      ConstantCostFunction<kParameterBlockSize>,
      1,
      kParameterBlockSize>();
  RunCostFunctionBenchmark<kResidualsAndJacobians, -1, 1, kParameterBlockSize>(
      state, *cf, {parameters_values.data()});
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
  auto cf = CostFunctionFactory<
      kDiffType>::template Create<Linear1CostFunction, 1, 1>();
  RunCostFunctionBenchmark<kEvalType, -1, 1, 1>(state, *cf, {parameter_block1});
}

BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear1, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Linear10(benchmark::State& state) {
  const double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  auto cf = CostFunctionFactory<
      kDiffType>::template Create<Linear10CostFunction, 10, 10>();
  RunCostFunctionBenchmark<kEvalType, -1, 10, 10>(
      state, *cf, {parameter_block1});
}

BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Linear10, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_Rat43(benchmark::State& state) {
  const double parameter_block1[] = {1., 2., 3., 4.};
  auto cf =
      CostFunctionFactory<kDiffType>::template Create<Rat43CostFunctor, 1, 4>(
          0.2, 0.3);
  RunCostFunctionBenchmark<kEvalType, -1, 1, 4>(state, *cf, {parameter_block1});
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
  auto cf = CostFunctionFactory<
      kDiffType>::template Create<SnavelyReprojectionError, 2, 9, 3>(0.2, 0.3);
  RunCostFunctionBenchmark<kEvalType, 1, 2, 9, 3>(
      state, *cf, {parameter_block1, parameter_block2});
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

  auto cf =
      CostFunctionFactory<kDiffType>::template Create<FunctorType,
                                                      FunctorType::PATCH_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POSE_SIZE,
                                                      FunctorType::POINT_SIZE>(
          intensities_host, bearings_host, image_target, intrinsics);

  RunCostFunctionBenchmark<kEvalType,
                           -1,
                           FunctorType::PATCH_SIZE,
                           FunctorType::POSE_SIZE,
                           FunctorType::POSE_SIZE,
                           FunctorType::POINT_SIZE>(
      state, *cf, {parameter_block1, parameter_block2, parameter_block3});
}

BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Photometric, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_RelativePose(benchmark::State& state) {
  double parameter_block1[] = {1., 2., 3., 4., 5., 6., 7.};
  double parameter_block2[] = {1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1};
  Eigen::Map<Eigen::Quaterniond>(parameter_block1).normalize();
  Eigen::Map<Eigen::Quaterniond>(parameter_block2).normalize();

  Eigen::Quaterniond q_i_j = Eigen::Quaterniond(1, 2, 3, 4).normalized();
  Eigen::Vector3d t_i_j(1, 2, 3);

  auto cf = CostFunctionFactory<
      kDiffType>::template Create<RelativePoseError, 6, 7, 7>(q_i_j, t_i_j);
  RunCostFunctionBenchmark<kEvalType, -1, 6, 7, 7>(
      state, *cf, {parameter_block1, parameter_block2});
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
static void BM_Brdf(benchmark::State& state) {
  const double material[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
  const Eigen::Vector3d c(0.1, 0.2, 0.3);
  const Eigen::Vector3d n = Eigen::Vector3d(-0.1, 0.5, 0.2).normalized();
  const Eigen::Vector3d v = Eigen::Vector3d(0.5, -0.2, 0.9).normalized();
  const Eigen::Vector3d l = Eigen::Vector3d(-0.3, 0.4, -0.3).normalized();
  const Eigen::Vector3d x = Eigen::Vector3d(0.5, 0.7, -0.1).normalized();
  const Eigen::Vector3d y = Eigen::Vector3d(0.2, -0.2, -0.2).normalized();

  auto cf = CostFunctionFactory<
      kDiffType>::template Create<Brdf, 3, 10, 3, 3, 3, 3, 3, 3>();
  RunCostFunctionBenchmark<kEvalType, -1, 3, 10, 3, 3, 3, 3, 3, 3>(
      state,
      *cf,
      {material, c.data(), n.data(), v.data(), l.data(), x.data(), y.data()});
}

BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_Brdf, kDynamicAutoDiff, kResidualsAndJacobians);

template <DiffType kDiffType, EvaluationType kEvalType>
static void BM_CostFunctionToFunctor(benchmark::State& state) {
  auto cf = std::make_unique<
      AutoDiffCostFunction<OuterProjectionFunctor<kDiffType>, 2, 3, 3, 3, 3>>(
      std::make_unique<OuterProjectionFunctor<kDiffType>>());
  const double rot[3] = {0.1, -0.2, 0.05};
  const double trans[3] = {0.5, -0.1, 2.0};
  const double intr[3] = {500.0, -0.01, 0.001};
  const double pt[3] = {0.3, -0.4, 5.0};
  RunCostFunctionBenchmark<kEvalType, -1, 2, 3, 3, 3, 3>(
      state, *cf, {rot, trans, intr, pt});
}

BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kAutoDiff, kResidualsAndJacobians);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor, kDynamicAutoDiff, kResidualsOnly);
BENCHMARK_TEMPLATE(BM_CostFunctionToFunctor,
                   kDynamicAutoDiff,
                   kResidualsAndJacobians);

}  // namespace ceres

BENCHMARK_MAIN();
