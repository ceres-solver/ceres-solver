
#ifndef CERES_INTERNAL_SPARSE_COVARIANCE_H_
#define CERES_INTERNAL_SPARSE_COVARIANCE_H_

#include <utility>
#include <map>
#include <vector>

#include "ceres/covariance.h"
#include "ceres/covariance_impl.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/suitesparse.h"
#include "ceres/problem.h"

namespace ceres {

namespace internal {

class CompressedRowSparseMatrix;

class SparseCovariance : CovarianceImpl {
 public:
  explicit SparseCovariance(const Covariance::Options& options);
  virtual ~SparseCovariance();
  virtual bool Compute(const vector<pair<double*, double*> >& covariance_blocks,
                       Problem* problem);

  virtual bool GetCovarianceBlock(double* parameter_block1,
                                  double* parameter_block2,
                                  double* covariance_block);

  bool ComputeCovarianceSparsity(
      const vector<pair<double*, double*> >& covariance_blocks,
      Problem* problem);

  bool ComputeCovarianceValues();

  const CompressedRowSparseMatrix* sparse_covariance() const {
    return  sparse_covariance_.get();
  }

 private:
  //const Covariance::Options& options_;
  Problem* problem_;
  Problem::EvaluateOptions evaluate_options_;
  map<double*, int> parameter_block_to_row_index_;
  scoped_ptr<CompressedRowSparseMatrix> sparse_covariance_;
  SuiteSparse ss_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SPARSE_COVARIANCE_H_
