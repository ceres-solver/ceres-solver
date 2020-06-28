// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_INTERNAL_MARGINALIZATION_H_
#define CERES_INTERNAL_MARGINALIZATION_H_

#include <set>
#include <vector>

#include "ceres/parameter_block.h"
#include "ceres/problem.h"

namespace ceres {

class MarginalFactorCostFunction;

namespace internal {

class MarginalizationImpl {
 public:
  explicit MarginalizationImpl();
  ~MarginalizationImpl();

  bool Compute(const std::set<double*>& parameter_blocks_to_marginalize,
               Problem* problem,
               std::vector<double*>* markov_blanket_parameter_blocks,
               MarginalFactorCostFunction** cost_function);

  bool MarginalizeOutVariables(
      const std::set<double*>& parameter_blocks_to_marginalize,
      Problem* problem);
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_MARGINALIZATION_H_
