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
// Helpers for making CostFunctions as needed by Ceres, with Jacobians
// computed via automatic differentiation.
//
// Unlike the automatic differentiation for CostFunctions implemented in
// autodiff_coft_function.h, this helper enables the automatic
// differentiation of functions where both the number of parameters and
// and the number of residuals are specified at run-time (as opposed to the
// compile-time constants required by AutoDiffCostFunction).
//
// To implement this dynamic-sizing, both this class and the underlying
// CostFunctor that this class will differentiate have a different
// interface than the statically sized AutoDiffCostFunction, as described
// below.
//
// Similar to AutoDiffCostFunction:
//
// To get an auto differentiated cost function, you must define a class with a
// templated operator() (a functor) that computes the cost function in terms of
// the template parameter T. The autodiff framework substitutes appropriate
// "jet" objects for T in order to compute the derivative when necessary, but
// this is hidden, and you should write the function as if T were a scalar type
// (e.g. a double-precision floating point number).
//
// Unlike AutoDiffCostFunction:
//
// Our CostFunctor's operator() takes two arguments:
//     const* const* parameters:
//         identical to the arguments supplied to AutoDiffCostFunction,
//         this supplies the parameters for which to evaluate this CostFunctor.
//     *residual_emitter:
//         instead of a block of memory into which errors should be written,
//         this is an object into which this CostFunctor should emit its
//         errors.  It has a single method:
//         residual_emitter->EmitResidual(T* residual)
//         This CostFunctor should invoke this method once for each residual
//         (in the same order each time), and the residuals and their
//         derivatives will be copied into the appropriate place in Ceres'
//         residuals and jacobians arrays.
//
// Also unlike AutoDiffCostFunction, DynamicSizeAutodiffCostFunction must
// be explicitly told about about the parameters used by the CostFunctor, so
// that the CostFunctor can be evaluated once prior to optimization, in order
// to count the residuals.  This is so that Ceres can allocate the memory
// required for the residuals and Jacobians prior to invoking
// DynamicSizeAutodiffCostFunction's operator().
//
// Implementation/Performance notes:
//
// DynamicSizeAutodiffCostFunction makes several passes over the supplied
// CostFunctor, each time computing the derivatives for JetT::DIMENSION
// residuals.  Any sized JetT can be used with DynamicSizeAutodiffCostFunction,
// but different sizes have different performance effects.  For example,
// using Jet<double, 1> will result in evaluating the the CostFunctor once
// for each parameter, whereas using Jet<double, 32> results in
// ceil(num_parameters / 32) evaluations of the CostFunctor (however, with
// a larger Jet, each operation is more expensive).
//
// This class is less efficient that using the statically-sized
// AutoDiffCostFunction--the reason for its existence is flexibility.


#ifndef CERES_PUBLIC_DYNAMIC_SIZE_AUTODIFF_COST_FUNCTION_H_
#define CERES_PUBLIC_DYNAMIC_SIZE_AUTODIFF_COST_FUNCTION_H_

#include <glog/logging.h>
#include "ceres/internal/autodiff.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/cost_function.h"
#include "ceres/types.h"

namespace ceres {

template <typename CostFunctor, typename JetT>
class DynamicSizeAutodiffCostFunction : public ceres::CostFunction {
 public:
  DynamicSizeAutodiffCostFunction(CostFunctor* functor)
    : functor_(functor),
      num_residuals_(-1) {}
  virtual ~DynamicSizeAutodiffCostFunction() {}

  void AddParameterBlock(const double* block, int size) {
    CHECK_EQ(-1, num_residuals_) << "You can't add a parameter block after "
        << "you've counted the number of residuals.";
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

    jacobian_evaluator_->Setup(parameters, residuals, jacobians);
    while(jacobian_evaluator_->HasMoreDerivativeBlocks()) {
      // Turn on this derivative block.
      jacobian_evaluator_->SetDerivativeBlockActive(1.0);

      // Compute the derivative block, passing jacobian_evaluator_ to the
      // user's functor_ as a callback.  The user's functor_ will invoke
      // jacobian_evaluator_'s EmitResidual method, which fills in the
      // appropriate entries inside jacobians.
      if (!(*functor_)(jacobian_evaluator_->jet_params_.data(),
                       jacobian_evaluator_.get())) {
        return false;
      }
      CHECK_EQ(num_residuals_, jacobian_evaluator_->residual_count_);

      // Turn off this derivative block.
      jacobian_evaluator_->SetDerivativeBlockActive(0.0);

      // We just calculated the residuals once, now we can check them.
      jacobian_evaluator_->set_residuals_ = false;

      // Move to the next derivative block.
      jacobian_evaluator_->MoveToNextDerivativeBlock();
    };

    return true;
  }

 private:
  typedef typename JetT::Scalar T;
  enum { JetSize = JetT::DIMENSION };

  internal::scoped_ptr<CostFunctor> functor_;
  int num_residuals_;
  vector<const double*> parameter_blocks_;

  class ResidualCounter {
   public:
    ResidualCounter(int* num_residuals) : num_residuals_(num_residuals) {}
    void EmitResidual(const T& residual) {
      // Ignore the emitted residual, just count them.
      (*num_residuals_)++;
    }

   private:
    int* num_residuals_;
  };

  class ResidualEvaluator {
   public:
    int residual_count_;
    ResidualEvaluator(double* residual_dest, int num_residuals)
        : residual_count_(0),
          num_residuals_(num_residuals),
          residual_dest_(residual_dest) {}
    void EmitResidual(const double& residual) {
      CHECK_LT(residual_count_, num_residuals_);
      residual_dest_[residual_count_++] = residual;
    }

   private:
    const int num_residuals_;
    double* const residual_dest_;
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

    JacobianEvaluator(const vector<int16>& parameter_block_sizes,
                      int num_residuals) :
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
