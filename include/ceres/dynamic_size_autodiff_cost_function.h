// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
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
// Author: thadh@google.com (Thad Hughes)
//

#ifndef CERES_PUBLIC_DYNAMIC_SIZE_AUTODIFF_COST_FUNCTION_H_
#define CERES_PUBLIC_DYNAMIC_SIZE_AUTODIFF_COST_FUNCTION_H_

#include <glog/logging.h>
#include "ceres/internal/autodiff.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/cost_function.h"
#include "ceres/types.h"

namespace ceres {

template <typename CostFunctor, typename Jet_>
class DynamicSizeAutodiffCostFunction : public ceres::CostFunction {
 public:
  DynamicSizeAutodiffCostFunction(CostFunctor* functor)
    : functor_(functor),
      num_residuals_(-1) {}
  virtual ~DynamicSizeAutodiffCostFunction() {}

  void AddParameterBlock(const double* block, int size) {
    CHECK_EQ(-1, num_residuals_);
    parameter_blocks_.push_back(block);
    mutable_parameter_block_sizes()->push_back(size);
  }

  const double* const* parameters() const {
    return parameter_blocks_.data();
  }

  void CountResiduals() {
    CHECK_EQ(num_residuals_, -1);
    CHECK_GT(parameter_blocks_.size(), 0);
    num_residuals_ = 0;
    ResidualCounter residual_counter(&num_residuals_);
    CHECK((*functor_)(parameters(), &residual_counter))
        << "CostFunction failed when counting residuals.  "
        << "This may mean initial parameter values are not valid.";
    CHECK_GT(num_residuals_, 0);
    set_num_residuals(num_residuals_);

    jacobian_evaluator_.reset(
        new JacobianEvaluator(parameter_block_sizes(), num_residuals_));
  }

  int NumResiduals() const {
    return num_residuals_;
  }

  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    CHECK_GT(num_residuals_, 0);
    if (jacobians == NULL) {
      ResidualEvaluator residual_evaluator(residuals, num_residuals_);
      if (!(*functor_)(parameters, &residual_evaluator)) {
        return false;
      }
      CHECK_EQ(num_residuals_, residual_evaluator.residual_count_);
      return true;
    }

    VLOG(1) << "Evaluating Jacobian"
        << ", parameter_block_sizes().size()="
        << parameter_block_sizes().size()
        << ", num_residuals=" << num_residuals_;
    jacobian_evaluator_->Setup(parameters, residuals, jacobians);
    while(jacobian_evaluator_->HasMoreDerivativeBlocks()) {
      // Turn on this derivative block.
      jacobian_evaluator_->SetDerivativeBlockActive(true);

      // Compute the derivative block.
      if (!(*functor_)(jacobian_evaluator_->jet_params_.data(),
                       jacobian_evaluator_.get())) {
        return false;
      }
      CHECK_EQ(num_residuals_, jacobian_evaluator_->residual_count_);

      // Turn off this derivative block.
      jacobian_evaluator_->SetDerivativeBlockActive(false);

      // We just calculated the residuals once, now we can check them.
      jacobian_evaluator_->set_residuals_ = false;

      // Move to the next derivative block.
      jacobian_evaluator_->MoveToNextDerivativeBlock();
    };

    return true;
  }

 private:
  typedef Jet_ JetT;
  typedef typename JetT::Scalar T;
  enum { JetSize = JetT::DIMENSION };

  internal::scoped_ptr<CostFunctor> functor_;
  int num_residuals_;
  vector<const double*> parameter_blocks_;

  class ResidualCounter {
   private:
    int* num_residuals_;
   public:
    ResidualCounter(int* num_residuals) : num_residuals_(num_residuals) {}
    void EmitResidual(const T& residual) {
      // Ignore the residual, just count them.
      (*num_residuals_)++;
    }
  };

  class ResidualEvaluator {
   private:
    const int num_residuals_;
    double* residual_dest_;
   public:
    int residual_count_;
    ResidualEvaluator(double* residual_dest, int num_residuals)
        : num_residuals_(num_residuals),
          residual_dest_(residual_dest),
          residual_count_(0) {}
    void EmitResidual(const double& residual) {
      CHECK_LT(residual_count_, num_residuals_);
      residual_dest_[residual_count_++] = residual;
    }
  };

  class JacobianEvaluator {
   public:
    const int num_residuals_;
    double* residual_dest_;
    double** jacobian_dest_;
    int residual_count_;
    bool set_residuals_;
    vector<vector<JetT> > jet_param_vectors_;
    vector<JetT*> jet_params_;
    int start_param_block_;
    int start_param_;

    JacobianEvaluator(
        const vector<int16>& parameter_block_sizes, int num_residuals) :
      num_residuals_(num_residuals),
      residual_dest_(NULL),
      jacobian_dest_(NULL),
      residual_count_(0),
      set_residuals_(true),
      start_param_block_(0),
      start_param_(0) {
      // Allocate space for the Jet parameters we'll use.
      // This could be thread-local.
      jet_param_vectors_.resize(
          parameter_block_sizes.size(), vector<JetT>(0.0));
      for (int i = 0; i < parameter_block_sizes.size(); ++i) {
        jet_param_vectors_[i].resize(parameter_block_sizes[i]);
        jet_params_.push_back(jet_param_vectors_[i].data());
      }
    }

    void Setup(double const* const* parameters,
               double* residuals,
               double** jacobians) {
      // Copy the parameters into our Jets.
      // This would really only have to be done once per optimizer iteration.
      for (int b = 0; b < jet_param_vectors_.size(); ++b) {
        vector<JetT>& block = jet_param_vectors_[b];
        for (int p = 0; p < block.size(); ++p) {
          block[p].a = parameters[b][p];
        }
      }
      set_residuals_ = true;
      residual_dest_ = residuals;
      jacobian_dest_ = jacobians;
      start_param_block_ = 0;
      start_param_ = 0;
    }

    bool HasMoreDerivativeBlocks() const {
      return start_param_block_ < jet_param_vectors_.size() &&
          start_param_ < jet_param_vectors_[start_param_block_].size();
    }

    bool MoveToNextParam(int* param_block, int* param) {
      (*param)++;
      if (*param < jet_param_vectors_[*param_block].size()) {
        return true;
      }
      *param = 0;
      (*param_block)++;
      return *param_block < jet_param_vectors_.size();
    }

    void SetDerivativeBlockActive(const double active) {
      int param_block = start_param_block_;
      int param = start_param_;
      for (int i = 0; i < JetSize; ++i) {
        CHECK(HasMoreDerivativeBlocks());
        CHECK_LT(param_block, jet_param_vectors_.size());
        CHECK_LT(param, jet_param_vectors_[param_block].size());
        jet_param_vectors_[param_block][param].v[i] = active;
        if (!MoveToNextParam(&param_block, &param)) {
          return;
        }
      }
    }

    void MoveToNextDerivativeBlock() {
      residual_count_ = 0;
      for (int i = 0; i < JetSize; ++i) {
        CHECK(HasMoreDerivativeBlocks());
        if (!MoveToNextParam(&start_param_block_, &start_param_)) {
          return;
        }
      }
    }

    void EmitResidual(const JetT& residual) {
      const int residual_index = residual_count_;
      CHECK_LT(residual_index, num_residuals_);
      residual_count_++;

      // Emit the residual.
      if (set_residuals_) {
        residual_dest_[residual_index] = residual.a;
      } else {
        CHECK_EQ(residual_dest_[residual_index], residual.a) << residual_index;
      }

      // Emit this block of the Jacobian from the Jet we got.
      int param_block = start_param_block_;
      int param = start_param_;
      for (int i = 0; i < JetSize; ++i) {
        CHECK(HasMoreDerivativeBlocks());
        const int jacobian_index =
            residual_index*jet_param_vectors_[param_block].size() + param;
        jacobian_dest_[param_block][jacobian_index] = residual.v[i];
        if (!MoveToNextParam(&param_block, &param)) {
          return;
        }
      }

    }
  };  // class JacobianEvaluator

  // Our single instance of JacobianEvaluator, allocated by CountResiduals().
  internal::scoped_ptr<JacobianEvaluator> jacobian_evaluator_;

};

}  // namespace ceres

#endif  // CERES_PUBLIC_DYNAMIC_SIZE_AUTODIFF_COST_FUNCTION_H_
