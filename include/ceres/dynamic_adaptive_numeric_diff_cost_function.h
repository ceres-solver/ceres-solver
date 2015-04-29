// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
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
// Author: mierle@gmail.com (Keir Mierle)
//         sameeragarwal@google.com (Sameer Agarwal)
//         thadh@gmail.com (Thad Hughes)
//         tbennun@gmail.com (Tal Ben-Nun)
//
// This numeric difference implementation differs from the one found in
// dynamic_numeric_diff_cost_function.h by using adaptive differentiation
// on the parameters to obtain the Jacobian matrix. The adaptive algorithm
// is based on Ridders' method for adaptive differentiation.
//
// References:
// C.J.F. Ridders, Accurate computation of F'(x) and F'(x) F"(x), Advances 
// in Engineering Software (1978), Volume 4, Issue 2, April 1982, Pages 75-76,
// ISSN 0141-1195, http://dx.doi.org/10.1016/S0141-1195(82)80057-0.
//
// The functor API differs slightly from the API for fixed size
// numeric diff; the expected interface for the cost functors is:
//
//   struct MyCostFunctor {
//     template<typename T>
//     bool operator()(double const* const* parameters, double* residuals) const {
//       // Use parameters[i] to access the i'th parameter block.
//     }
//   }
//
// Since the sizing of the parameters is done at runtime, you must
// also specify the sizes after creating the
// DynamicAdaptiveNumericDiffCostFunction. For example:
//
//   DynamicAdaptiveNumericDiffCostFunction<MyCostFunctor, CENTRAL> cost_function(
//       new MyCostFunctor());
//   cost_function.AddParameterBlock(5);
//   cost_function.AddParameterBlock(10);
//   cost_function.SetNumResiduals(21);

#ifndef CERES_PUBLIC_DYNAMIC_ADAPTIVE_NUMERIC_DIFF_COST_FUNCTION_H_
#define CERES_PUBLIC_DYNAMIC_ADAPTIVE_NUMERIC_DIFF_COST_FUNCTION_H_

#include <cmath>
#include <numeric>
#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/numeric_diff.h"
#include "glog/logging.h"

namespace ceres {

	/**
	 * @brief Dynamically-sized adaptive numeric differentiation.
	 * @param method Numeric differentiation scheme (e.g. central, forward, 
	 *               backward)
	 * @param max_extrapolation Maximal number of extrapolation steps to 
	 *                          perform until convergence.
	 * @param relative_error If true, derivative convergence error uses 
	 *                       d(a,b) = (1 - a/b). Otherwise, d(a,b) = ||a-b||
	 *                       is used.
	 */
	template <typename CostFunctor, NumericDiffMethod method = CENTRAL, 
			  int max_extrapolation = 10, bool relative_error = true>
	class DynamicAdaptiveNumericDiffCostFunction : public CostFunction {
	public:
		/**
		 * @brief Initializes a dynamically-sized adaptive numeric diff cost
		 *        function.
		 * @param functor The cost functor to use.
		 * @param ownership Determines whether this class takes ownership
		 *                  of functor deallocation.
		 * @param relative_step_size Initial step size to use (relative to
		 *                           parameter values)
		 * @param epsilon Derivative convergence criterion.
		 * @param step_shrink_factor Shrink factor for step size with each 
		 *                           extrapolation.
		 */
		explicit DynamicAdaptiveNumericDiffCostFunction(
			const CostFunctor* functor,
			Ownership ownership = TAKE_OWNERSHIP,
			double relative_step_size = 1e-2,
			double epsilon = 1e-6, double step_shrink_factor = 1.4)
			: functor_(functor),
			ownership_(ownership),
			relative_step_size_(relative_step_size),
			epsilon_(epsilon),
		    step_shrink_factor_(step_shrink_factor) {
		}

		virtual ~DynamicAdaptiveNumericDiffCostFunction() {
			if (ownership_ != TAKE_OWNERSHIP) {
				functor_.release();
			}
		}

		void AddParameterBlock(int size) {
			mutable_parameter_block_sizes()->push_back(size);
		}

		void SetNumResiduals(int num_residuals) {
			set_num_residuals(num_residuals);
		}

		virtual bool Evaluate(double const* const* parameters,
			double* residuals,
			double** jacobians) const {
				CHECK_GT(num_residuals(), 0)
					<< "You must call DynamicAdaptiveNumericDiffCostFunction::SetNumResiduals() "
					<< "before DynamicAdaptiveNumericDiffCostFunction::Evaluate().";

				const vector<int32>& block_sizes = parameter_block_sizes();
				CHECK(!block_sizes.empty())
					<< "You must call DynamicAdaptiveNumericDiffCostFunction::AddParameterBlock() "
					<< "before DynamicAdaptiveNumericDiffCostFunction::Evaluate().";

				const bool status = EvaluateCostFunctor(parameters, residuals);
				if (jacobians == NULL || !status) {
					return status;
				}

				// Create local space for a copy of the parameters which will get mutated.
				int parameters_size = accumulate(block_sizes.begin(), block_sizes.end(), 0);
				vector<double> parameters_copy(parameters_size);
				vector<double*> parameters_references_copy(block_sizes.size());
				parameters_references_copy[0] = &parameters_copy[0];
				for (int block = 1; block < block_sizes.size(); ++block) {
					parameters_references_copy[block] = parameters_references_copy[block - 1]
					+ block_sizes[block - 1];
				}

				// Copy the parameters into the local temp space.
				for (int block = 0; block < block_sizes.size(); ++block) {
					memcpy(parameters_references_copy[block],
						parameters[block],
						block_sizes[block] * sizeof(*parameters[block]));
				}

				for (int block = 0; block < block_sizes.size(); ++block) {
					if (jacobians[block] != NULL &&
						!EvaluateJacobianForParameterBlock(block_sizes[block],
						block,
						relative_step_size_,
						residuals,
						&parameters_references_copy[0],
						jacobians)) {
							return false;
					}
				}
				return true;
		}

	private:
		bool EvaluateJacobianForParameterBlock(const int parameter_block_size,
											   const int parameter_block,
											   const double relative_step_size,
											   double const* residuals_at_eval_point,
											   double** parameters,
											   double** jacobians) const {
			using Eigen::Map;
			using Eigen::Matrix;
			using Eigen::Dynamic;
			using Eigen::RowMajor;

			typedef Matrix<double, Dynamic, 1> ResidualVector;
			typedef Matrix<double, Dynamic, 1> ParameterVector;
			typedef Matrix<double, Dynamic, Dynamic, RowMajor> JacobianMatrix;

			int num_residuals = this->num_residuals();

			Map<JacobianMatrix> parameter_jacobian(jacobians[parameter_block],
				num_residuals,
				parameter_block_size);

			// Mutate one element at a time and then restore.
			Map<ParameterVector> x_plus_delta(parameters[parameter_block],
				parameter_block_size);
			ParameterVector x(x_plus_delta);
			ParameterVector step_size = x.array().abs() * relative_step_size;

			// To handle cases where a parameter is exactly zero, instead use
			// the mean step_size for the other dimensions.
			double fallback_step_size = step_size.sum() / step_size.rows();
			if (fallback_step_size == 0.0) {
				// If all the parameters are zero, there's no good answer. Use the given
				// relative step_size as absolute step_size and hope for the best.
				fallback_step_size = relative_step_size;
			}

			// For each parameter in the parameter block, use adaptive finite
			// differences to compute the derivative for that parameter.
			for (int j = 0; j < parameter_block_size; ++j) {
				if (step_size(j) == 0.0) {
					// The parameter is exactly zero, so compromise and use the
					// mean step_size from the other parameters. This can break in
					// many cases, but it's hard to pick a good number without
					// problem specific knowledge.
					step_size(j) = fallback_step_size;
				}

				// Initialize parameters for adaptive differentiation
				double current_step_size = step_size(j);

				// Constant vector of ones
				ResidualVector ones(num_residuals);
				ones.setConstant(1.0);

				// Temporary vector to store finite difference residuals
				// at (x+delta)
				ResidualVector temp_residuals(num_residuals);

				// Vector that stores results
				ResidualVector residuals(num_residuals);
				
				// Double-buffering temporary differential candidate vectors
				// from previous step size
				ResidualVector stepsize_candidates_a[max_extrapolation];
				ResidualVector stepsize_candidates_b[max_extrapolation];
				ResidualVector *cur_candidates = stepsize_candidates_a;
				ResidualVector *prev_candidates = stepsize_candidates_b;
				
				double norm_err = std::numeric_limits<double>::infinity();

				// Loop over decreasing step sizes until:
				//	1. Error is smaller than a given value (epsilon), or
				//	2. Maximal order of extrapolation reached, or
				//  3. Extrapolation becomes numerically unstable.
				for(int i = 0; i < max_extrapolation; ++i) {
					cur_candidates[0].resize(num_residuals);

					// Compute the numerical derivative at this step size
					if(!EvaluateNumericDiff(j, current_step_size,
											num_residuals,
											parameter_block_size,
											parameters, x.data(),
											x_plus_delta.data(), 
											residuals_at_eval_point,
											temp_residuals.data(),
											cur_candidates[0].data())) {
						// Something went wrong; bail.
						return false;
					}

					// Shrink differentiation step size
					current_step_size /= step_shrink_factor_;
					
					double factor = step_shrink_factor_ * step_shrink_factor_;
					for(int k = 1; k <= i; ++k) {
						cur_candidates[k].resize(num_residuals);

						ResidualVector& candidate = cur_candidates[k];

						// Extrapolate various orders of finite difference 
						// using Neville's algorithm
						candidate = 
							(factor * cur_candidates[k - 1] - 
							 prev_candidates[k - 1]) / (factor - 1.0);

						factor *= step_shrink_factor_ * step_shrink_factor_;

						// Compute the difference between the previous
						// value and the current
						double temp_error;
						if(relative_error) {
							// Compare using relative error: (1 - a/b)
							temp_error = std::max(
							  (ones - candidate.cwiseQuotient(
								cur_candidates[k - 1])).norm(),
							  (ones - candidate.cwiseQuotient(
								prev_candidates[k - 1])).norm());
						} else {						
							// Compare values using absolute error: (a - b)
							temp_error = std::max(
								(candidate - cur_candidates[k - 1]).norm(),
								(candidate - prev_candidates[k - 1]).norm());
						}

						// If the error has decreased, update results
						if(temp_error <= norm_err) {
							norm_err = temp_error;
							residuals = candidate;

							// If the error is small enough, stop
							if(norm_err < epsilon_)
								break;
						}
					}

					// If the error is small enough, stop
					if(norm_err < epsilon_)
						break;

					// Testing for numerical stability of the results
					if(i > 0) {
						if(relative_error) {
							if((ones - cur_candidates[i].cwiseQuotient( 
								prev_candidates[i - 1])).norm() >= 
							   2 * norm_err)
									break;
						} else {
							if((cur_candidates[i] - 
								prev_candidates[i - 1]).norm() >= 
							   2 * norm_err)
									break;
						}
					}

					// Swap the two vector pointers
					std::swap(cur_candidates, prev_candidates);
				}
				
				// Store the obtained value
				parameter_jacobian.col(j).matrix() = residuals;
			}
			return true;
		}

		bool EvaluateNumericDiff(int ind, double current_step_size,
							int num_residuals,
							int parameter_block_size,
							double** parameters,
							double const* x_ptr,
							double* x_plus_delta_ptr,
							double const* residuals_at_eval_point,
							double* temp_residuals_ptr, 
							double* residuals_ptr) const {
			using Eigen::Map;
			using Eigen::Matrix;
			using Eigen::Dynamic;
			
			typedef Matrix<double, Dynamic, 1> ResidualVector;
			typedef Matrix<double, Dynamic, 1> ParameterVector;
			
			Map<const ParameterVector> x(x_ptr, parameter_block_size);
			Map<ParameterVector> x_plus_delta(x_plus_delta_ptr,
											  parameter_block_size);

			Map<ResidualVector> residuals(residuals_ptr, num_residuals);
			Map<ResidualVector> temp_residuals(temp_residuals_ptr,
				                               num_residuals);

			// Mutate one element at a time and then restore.
			x_plus_delta(ind) = x(ind) + current_step_size;


			if (!EvaluateCostFunctor(parameters, residuals_ptr)) {
				// Something went wrong; bail.
				return false;
			}

			// Compute this column of the jacobian in 3 steps:
			// 1. Store residuals for the forward part.
			// 2. Subtract residuals for the backward (or 0) part.
			// 3. Divide out the run.
			double one_over_h = 1 / current_step_size;
			if (method == CENTRAL) {
				// Compute the function on the other side of x(j).
				x_plus_delta(ind) = x(ind) - current_step_size;

				if (!EvaluateCostFunctor(parameters, temp_residuals_ptr)) {
					// Something went wrong; bail.
					return false;
				}

				residuals -= temp_residuals;
				one_over_h /= 2;
			} else {
				// Forward difference only; reuse existing residuals evaluation.
				residuals -=
					Map<const ResidualVector>(residuals_at_eval_point,
											  num_residuals);
			}
			x_plus_delta(ind) = x(ind);  // Restore x_plus_delta.

			// Divide out the run to get slope.
			residuals *= one_over_h;

			return true;
		}

		bool EvaluateCostFunctor(double const* const* parameters,
			double* residuals) const {
				return EvaluateCostFunctorImpl(functor_.get(),
					parameters,
					residuals,
					functor_.get());
		}

		// Helper templates to allow evaluation of a functor or a
		// CostFunction.
		bool EvaluateCostFunctorImpl(const CostFunctor* functor,
			double const* const* parameters,
			double* residuals,
			const void* /* NOT USED */) const {
				return (*functor)(parameters, residuals);
		}

		bool EvaluateCostFunctorImpl(const CostFunctor* functor,
			double const* const* parameters,
			double* residuals,
			const CostFunction* /* NOT USED */) const {
				return functor->Evaluate(parameters, residuals, NULL);
		}

		internal::scoped_ptr<const CostFunctor> functor_;
		Ownership ownership_;
		const double relative_step_size_;
		const double epsilon_;
		const double step_shrink_factor_;
	};

}  // namespace ceres

#endif  // CERES_PUBLIC_DYNAMIC_ADAPTIVE_NUMERIC_DIFF_COST_FUNCTION_H_
