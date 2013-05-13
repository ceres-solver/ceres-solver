#include "ceres/covariance.h"

#include <utility>
#include <vector>
#include "ceres/problem.h"
#include "ceres/covariance_impl.h"

namespace ceres {

Covariance::Covariance(const Covariance::Options& options) {
  // Depending on options, construct blah blah
}

bool Covariance::Compute(const vector<pair<double*, double*> >& covariance_blocks,
                         Problem* problem) {
  return impl_->Compute(covariance_blocks, problem);
}

bool Covariance::GetCovarianceBlock(double* parameter_block1,
                                    double* parameter_block2,
                                    double* covariance_block) {
  return impl_->GetCovarianceBlock(parameter_block1,
                                   parameter_block2,
                                   covariance_block);
}

}  // namespace ceres
