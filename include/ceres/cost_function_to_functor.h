// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2013 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: sameeragarwal@google.com (Sameer Agarwal)
//
// CostFunctionToFunctor is an adapter class that allows users to use
// CostFunction objects in templated functors which are to be used for
// automatic differentiation.  This allows the user to seamlessly mix
// analytic, numeric and automatic differentiation.
//
// For example, let us assume that
//
//  class IntrinsicProjection : public SizedCostFunction<2, 5, 3> {
//    public:
//      IntrinsicProjection(const double* observations);
//      virtual bool Evaluate(double const* const* parameters,
//                            double* residuals,
//                            double** jacobians) const;
//  };
//
// is a cost function that implements the projection of a point in its
// local coordinate system onto its image plane and subtracts it from
// the observed point projection. It can compute its residual and
// either via analytic or numerical differentiation can compute its
// jacobians.
//
// Now we would like to compose the action of this CostFunction with
// the action of camera extrinsics, i.e., rotation and
// translation. Say we have a templated function
//
//   template<typename T>
//   void RotateAndTranslatePoint(const T* rotation,
//                                const T* translation,
//                                const T* point,
//                                T* result);
//
// Then we can now do the following,
//
// struct CameraProjection {
//   CameraProjection(double* observation) {
//     intrinsic_projection_.reset(
//         new CostFunctionToFunctor<2, 5, 3>(new IntrinsicProjection(observation_)));
//   }
//   template <typename T>
//   bool operator(const T* rotation,
//                 const T* translation,
//                 const T* intrinsics,
//                 const T* point,
//                 const T* residual) const {
//     T transformed_point[3];
//     RotateAndTranslatePoint(rotation, translation, point, transformed_point);
//     return (*intrinsic_projection_)(intrinsics, transformed_point, residual);
//   }
//
//  private:
//   scoped_ptr<CostFunctionToFunctor<2,5,3> > intrinsic_projection_;
// };

#include <numeric>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/fixed_array.h"
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/jet.h"

namespace ceres {

template <int kNumResiduals,
          int N0,
          int N1 = 0,
          int N2 = 0,
          int N3 = 0,
          int N4 = 0,
          int N5 = 0,
          int N6 = 0,
          int N7 = 0,
          int N8 = 0,
          int N9 = 0>
class CostFunctionToFunctor {
 public:
  CostFunctionToFunctor(CostFunction* cost_function)
  : cost_function_(cost_function) {
    CHECK_NOTNULL(cost_function);

    CHECK(kNumResiduals > 0);
    CHECK_EQ(cost_function->num_residuals(), kNumResiduals);

    // This block breaks the 80 column rule to keep it somewhat readable.
    CHECK((!N1 && !N2 && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && !N2 && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && !N3 && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && !N4 && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && !N5 && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && !N6 && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && !N7 && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && !N8 && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && (N8 > 0) && !N9) ||
          ((N1 > 0) && (N2 > 0) && (N3 > 0) && (N4 > 0) && (N5 > 0) && (N6 > 0) && (N7 > 0) && (N8 > 0) && (N9 > 0)))
        << "Zero block cannot precede a non-zero block. Block sizes are "
        << "(ignore trailing 0s): " << N0 << ", " << N1 << ", " << N2 << ", "
        << N3 << ", " << N4 << ", " << N5 << ", " << N6 << ", " << N7 << ", "
        << N8 << ", " << N9;

    const vector<int16>& parameter_block_sizes = cost_function->parameter_block_sizes();
    const int num_parameter_blocks =
        (N0 > 0) + (N1 > 0) + (N2 > 0) + (N3 > 0) + (N4 > 0) +
        (N5 > 0) + (N6 > 0) + (N7 > 0) + (N8 > 0) + (N9 > 0);
    CHECK_EQ(parameter_block_sizes.size(), num_parameter_blocks);

    CHECK(N0 == 0 || N0 == parameter_block_sizes[0]);
    if (parameter_block_sizes.size() > 1) CHECK(N1 == 0 || N1 == parameter_block_sizes[1]);
    if (parameter_block_sizes.size() > 2) CHECK(N2 == 0 || N2 == parameter_block_sizes[2]);
    if (parameter_block_sizes.size() > 3) CHECK(N3 == 0 || N3 == parameter_block_sizes[3]);
    if (parameter_block_sizes.size() > 4) CHECK(N4 == 0 || N4 == parameter_block_sizes[4]);
    if (parameter_block_sizes.size() > 5) CHECK(N5 == 0 || N5 == parameter_block_sizes[5]);
    if (parameter_block_sizes.size() > 6) CHECK(N6 == 0 || N6 == parameter_block_sizes[6]);
    if (parameter_block_sizes.size() > 7) CHECK(N7 == 0 || N7 == parameter_block_sizes[7]);
    if (parameter_block_sizes.size() > 8) CHECK(N8 == 0 || N8 == parameter_block_sizes[8]);
    if (parameter_block_sizes.size() > 9) CHECK(N9 == 0 || N9 == parameter_block_sizes[9]);

    CHECK_EQ(accumulate(parameter_block_sizes.begin(), parameter_block_sizes.end(), 0),
             N0 + N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9);
  }


  bool operator()(const double* x1, double* residuals) const {
    return cost_function_->Evaluate(&x1, residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(2);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(3);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(4);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(5);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  const double* x6,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(6);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    parameter_blocks[5] = x6;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  const double* x6,
                  const double* x7,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(7);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    parameter_blocks[5] = x6;
    parameter_blocks[6] = x7;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  const double* x6,
                  const double* x7,
                  const double* x8,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(8);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    parameter_blocks[5] = x6;
    parameter_blocks[6] = x7;
    parameter_blocks[7] = x8;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  const double* x6,
                  const double* x7,
                  const double* x8,
                  const double* x9,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(9);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    parameter_blocks[5] = x6;
    parameter_blocks[6] = x7;
    parameter_blocks[7] = x8;
    parameter_blocks[8] = x9;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  bool operator()(const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4,
                  const double* x5,
                  const double* x6,
                  const double* x7,
                  const double* x8,
                  const double* x9,
                  const double* x10,
                  double* residuals) const {
    internal::FixedArray<const double*> parameter_blocks(10);
    parameter_blocks[0] = x1;
    parameter_blocks[1] = x2;
    parameter_blocks[2] = x3;
    parameter_blocks[3] = x4;
    parameter_blocks[4] = x5;
    parameter_blocks[5] = x6;
    parameter_blocks[6] = x7;
    parameter_blocks[7] = x8;
    parameter_blocks[8] = x9;
    parameter_blocks[9] = x10;
    return cost_function_->Evaluate(parameter_blocks.get(), residuals, NULL);
  }

  template <typename JetT>
  bool operator()(const JetT* x1, JetT* residuals) const {
    return EvaluateWithJets(&x1, residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(2);
    jets[0] = x1;
    jets[1] = x2;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(3);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(4);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(5);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  const JetT* x6,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(6);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    jets[5] = x6;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  const JetT* x6,
                  const JetT* x7,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(7);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    jets[5] = x6;
    jets[6] = x7;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  const JetT* x6,
                  const JetT* x7,
                  const JetT* x8,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(8);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    jets[5] = x6;
    jets[6] = x7;
    jets[7] = x8;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  const JetT* x6,
                  const JetT* x7,
                  const JetT* x8,
                  const JetT* x9,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(9);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    jets[5] = x6;
    jets[6] = x7;
    jets[7] = x8;
    jets[8] = x9;
    return EvaluateWithJets(jets.get(), residuals);
  }

  template <typename JetT>
  bool operator()(const JetT* x1,
                  const JetT* x2,
                  const JetT* x3,
                  const JetT* x4,
                  const JetT* x5,
                  const JetT* x6,
                  const JetT* x7,
                  const JetT* x8,
                  const JetT* x9,
                  const JetT* x10,
                  JetT* residuals) const {
    internal::FixedArray<const JetT*> jets(10);
    jets[0] = x1;
    jets[1] = x2;
    jets[2] = x3;
    jets[3] = x4;
    jets[4] = x5;
    jets[5] = x6;
    jets[6] = x7;
    jets[7] = x8;
    jets[8] = x9;
    jets[9] = x10;
    return EvaluateWithJets(jets.get(), residuals);
  }

 private:
  template <typename JetT>
  bool EvaluateWithJets(const JetT** inputs, JetT* output) const {
    const int kNumParameters =  N0 + N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9;
    const vector<int16>& parameter_block_sizes = cost_function_->parameter_block_sizes();
    const int num_parameter_blocks = parameter_block_sizes.size();
    const int num_residuals = cost_function_->num_residuals();

    internal::FixedArray<double> parameters(kNumParameters);
    internal::FixedArray<double*> parameter_blocks(num_parameter_blocks);
    internal::FixedArray<double> jacobians(num_residuals * kNumParameters);
    internal::FixedArray<double*> jacobian_blocks(num_parameter_blocks);
    internal::FixedArray<double> residuals(num_residuals);

    double* parameter_ptr = parameters.get();
    double* jacobian_ptr = jacobians.get();
    for (int i = 0; i < num_parameter_blocks; ++i) {
      parameter_blocks[i] = parameter_ptr;
      jacobian_blocks[i] = jacobian_ptr;
      for (int j = 0; j < parameter_block_sizes[i]; ++j) {
        *parameter_ptr++ = inputs[i][j].a;
      }
      jacobian_ptr += num_residuals * parameter_block_sizes[i];
    }

    if (!cost_function_->Evaluate(parameter_blocks.get(),
                                  residuals.get(),
                                  jacobian_blocks.get())) {
      return false;
    }

    for (int i = 0; i < num_residuals; ++i) {
      output[i].a = residuals[i];
      output[i].v.setZero();

      for (int j = 0; j < num_parameter_blocks; ++j) {
        const int16 block_size = parameter_block_sizes[j];
        for (int k = 0; k < parameter_block_sizes[j]; ++k) {
          output[i].v += jacobian_blocks[j][i * block_size + k] * inputs[j][k].v;
        }
      }
    }

    return true;
  }

 private:
  internal::scoped_ptr<CostFunction> cost_function_;
};

}  // namespace ceres
