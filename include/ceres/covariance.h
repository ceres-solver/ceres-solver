#ifndef CERES_PUBLIC_COVARIANCE_H_
#define CERES_PUBLIC_COVARIANCE_H_

#include <vector>
#include "ceres/internal/port.h"
#include "ceres/internal/scoped_ptr.h"

namespace ceres {

class Problem;

namespace internal { class CovarianceImpl; } // namespace internal

class Covariance {
 public:
  struct Options {};

  Covariance(const Options& options);
  bool Compute(const vector<pair<double*, double*> >& covariance_blocks,
               Problem* problem);

  bool GetCovarianceBlock(double* parameter_block1,
                          double* parameter_block2,
                          double* covariance_block);

 private:
  internal::scoped_ptr<internal::CovarianceImpl> impl_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_COVARIANCE_H_
