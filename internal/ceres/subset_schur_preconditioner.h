#ifndef CERES_INTERNAL_SUBSET_SCHUR_PRECONDITIONER_H_
#define CERES_INTERNAL_SUBSET_SCHUR_PRECONDITIONER_H_

#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ceres/linear_solver.h"
#include "ceres/preconditioner.h"
#include "ceres/sparse_cholesky.h"

namespace ceres {
namespace internal {

class BlockRandomAccessSparseMatrix;
class BlockSparseMatrix;
struct CompressedRowBlockStructure;
class SchurEliminatorBase;

class SubsetSchurPreconditioner : public BlockSparseMatrixPreconditioner {
 public:
  // Initialize the symbolic structure of the preconditioner. bs is
  // the block structure of the linear system to be solved. It is used
  // to determine the sparsity structure of the preconditioner matrix.
  //
  // It has the same structural requirement as other Schur complement
  // based solvers. Please see schur_eliminator.h for more details.
  SubsetSchurPreconditioner(const CompressedRowBlockStructure& bs,
                            const Preconditioner::Options& options);
  virtual ~SubsetSchurPreconditioner() = default;

  // Precsonditioner interface
  void RightMultiply(const double* x, double* y) const final;
  int num_rows() const final;

 private:
  bool UpdateImpl(const BlockSparseMatrix& A, const double* D) final;
  void InitStorage();
  void InitEliminator();
  LinearSolverTerminationType Factorize();

  Preconditioner::Options options_;
  int num_blocks_;
  std::unique_ptr<CompressedRowBlockStructure> subset_block_structure_;
  std::unique_ptr<SchurEliminatorBase> eliminator_;
  // Preconditioner matrix.
  std::unique_ptr<BlockRandomAccessSparseMatrix> m_;
  std::unique_ptr<SparseCholesky> sparse_cholesky_;
  std::vector<int> blocks_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_SUBSET_SCHUR_PRECONDITIONER_H_
