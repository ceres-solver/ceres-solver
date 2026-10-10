// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2026 Google Inc. All rights reserved.
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
//          sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_FACTORY_H_
#define CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_FACTORY_H_

#include <memory>
#include <type_traits>
#include <utility>

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

}  // namespace ceres

#endif  // CERES_INTERNAL_AUTODIFF_BENCHMARKS_COST_FUNCTION_FACTORY_H_
