#include "ceres/compressed_row_sparse_matrix.h"

namespace ceres {
namespace internal {

CompressedRowSparseMatrix* IncompleteLQFactorization(
    const CompressedRowSparseMatrix& matrix,
    const int l_level_of_fill,
    const double l_drop_tolerance,
    const int q_level_of_fill,
    const double q_drop_tolerance);

}  // namespace internal
}  // namespace ceres
