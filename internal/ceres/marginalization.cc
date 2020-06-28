// Author: evanlevine138e@gmail.com (Evan Levine)

#include "ceres/marginalization.h"

#include <vector>

#include "ceres/marginalization_impl.h"
#include "ceres/problem.h"

namespace ceres {

using std::set;
using std::vector;

Marginalization::Marginalization() {
  impl_.reset(new internal::MarginalizationImpl());
}

Marginalization::~Marginalization() {}

bool Marginalization::MarginalizeOutVariables(
    const set<double*>& parameter_blocks_to_marginalize, Problem* problem) {
  return impl_->MarginalizeOutVariables(parameter_blocks_to_marginalize,
                                        problem);
}

bool Marginalization::Compute(
    const set<double*>& parameter_blocks_to_marginalize, Problem* problem,
    vector<double*>* markov_blanket_parameter_blocks,
    MarginalFactorCostFunction** cost_function) {
  return impl_->Compute(parameter_blocks_to_marginalize, problem,
                        markov_blanket_parameter_blocks, cost_function);
}

}  // namespace ceres
