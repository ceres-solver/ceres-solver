#ifndef CERES_INTERNAL_SPARSE_COVARIANCE_IMPL_H_
#define CERES_INTERNAL_SPARSE_COVARIANCE_IMPL_H_

#include <utility>
#include <vector>

namespace ceres {

class Problem;

namespace internal {

class CovarianceImpl {
 public:
  virtual ~CovarianceImpl() {}
  virtual bool Compute(const vector<pair<double*, double*> >& covariance_blocks,
                       Problem* problem) = 0;
  virtual bool GetCovarianceBlock(double* parameter_block1,
                                  double* parameter_block2,
                                  double* covariance_block) = 0;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SPARSE_COVARIANCE_IMPL_H_
