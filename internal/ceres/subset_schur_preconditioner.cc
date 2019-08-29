#include "ceres/subset_schur_preconditioner.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "ceres/block_random_access_sparse_matrix.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/canonical_views_clustering.h"
#include "ceres/graph.h"
#include "ceres/graph_algorithms.h"
#include "ceres/linear_solver.h"
#include "ceres/schur_eliminator.h"
#include "ceres/single_linkage_clustering.h"
#include "ceres/visibility.h"
#include "glog/logging.h"

namespace ceres {
namespace internal {

using std::make_pair;
using std::pair;
using std::set;
using std::swap;
using std::vector;

// TODO(sameeragarwal): Currently these are magic weights for the
// preconditioner construction. Move these higher up into the Options
// struct and provide some guidelines for choosing them.
//
// This will require some more work on the clustering algorithm and
// possibly some more refactoring of the code.
static const double kCanonicalViewsSizePenaltyWeight = 3.0;
static const double kCanonicalViewsSimilarityPenaltyWeight = 0.0;
static const double kSingleLinkageMinSimilarity = 0.9;

SubsetSchurPreconditioner::SubsetSchurPreconditioner(
    const CompressedRowBlockStructure& bs,
    const Preconditioner::Options& options)
    : options_(options) {
  CHECK_GT(options_.elimination_groups.size(), 1);
  CHECK_GT(options_.elimination_groups[0], 0);
  CHECK(options_.type == SUBSET)
      << "Unknown preconditioner type: " << options_.type;
  num_blocks_ = bs.cols.size() - options_.elimination_groups[0];

  // TODO(sameeragarwal): What is the best thing to do here?
  CHECK_GT(num_blocks_, 0) << "Jacobian should have at least 1 f_block for "
                           << "subset schur preconditioning.";
  CHECK(options_.context != NULL);

  subset_block_structure_.reset(new CompressedRowBlockStructure);
  subset_block_structure_->cols = bs.cols;
  subset_block_structure_->rows.reserve(options.subset_preconditioner_rows.size());
  for (auto r : options.subset_preconditioner_rows) {
    subset_block_structure_->rows.emplace_back(bs.rows[r]);
  }

  InitStorage();
  InitEliminator();

  LinearSolver::Options sparse_cholesky_options;
  sparse_cholesky_options.sparse_linear_algebra_library_type =
      options_.sparse_linear_algebra_library_type;
  // The preconditioner's sparsity is not available in the
  // preprocessor, so the columns of the Jacobian have not been
  // reordered to minimize fill in when computing its sparse Cholesky
  // factorization. So we must tell the SparseCholesky object to
  // perform approximate minimum-degree reordering, which is done by
  // setting use_postordering to true.
  sparse_cholesky_options.use_postordering = true;
  sparse_cholesky_ = SparseCholesky::Create(sparse_cholesky_options);
}

// Determine the non-zero blocks in the Schur Complement matrix, and
// initialize a BlockRandomAccessSparseMatrix object.
void SubsetSchurPreconditioner::InitStorage() {
  const int num_eliminate_blocks = options_.elimination_groups[0];
  const int num_col_blocks = subset_block_structure_->cols.size();
  const int num_row_blocks = subset_block_structure_->rows.size();

  blocks_.resize(num_col_blocks - num_eliminate_blocks, 0);
  for (int i = num_eliminate_blocks; i < num_col_blocks; ++i) {
    blocks_[i - num_eliminate_blocks] = subset_block_structure_->cols[i].size;
  }

  set<pair<int, int>> block_pairs;
  for (int i = 0; i < blocks_.size(); ++i) {
    block_pairs.insert(make_pair(i, i));
  }

  int r = 0;
  while (r < num_row_blocks) {
    int e_block_id = subset_block_structure_->rows[r].cells.front().block_id;
    if (e_block_id >= num_eliminate_blocks) {
      break;
    }
    vector<int> f_blocks;

    // Add to the chunk until the first block in the row is
    // different than the one in the first row for the chunk.
    for (; r < num_row_blocks; ++r) {
      const CompressedRow& row = subset_block_structure_->rows[r];
      if (row.cells.front().block_id != e_block_id) {
        break;
      }

      // Iterate over the blocks in the row, ignoring the first
      // block since it is the one to be eliminated.
      for (int c = 1; c < row.cells.size(); ++c) {
        const Cell& cell = row.cells[c];
        f_blocks.push_back(cell.block_id - num_eliminate_blocks);
      }
    }

    sort(f_blocks.begin(), f_blocks.end());
    f_blocks.erase(unique(f_blocks.begin(), f_blocks.end()), f_blocks.end());
    for (int i = 0; i < f_blocks.size(); ++i) {
      for (int j = i + 1; j < f_blocks.size(); ++j) {
        block_pairs.insert(make_pair(f_blocks[i], f_blocks[j]));
      }
    }
  }

  // Remaining rows do not contribute to the chunks and directly go
  // into the schur complement via an outer product.
  for (; r < num_row_blocks; ++r) {
    const CompressedRow& row = subset_block_structure_->rows[r];
    CHECK_GE(row.cells.front().block_id, num_eliminate_blocks);
    for (int i = 0; i < row.cells.size(); ++i) {
      int r_block1_id = row.cells[i].block_id - num_eliminate_blocks;
      for (int j = 0; j < row.cells.size(); ++j) {
        int r_block2_id = row.cells[j].block_id - num_eliminate_blocks;
        if (r_block1_id <= r_block2_id) {
          block_pairs.insert(make_pair(r_block1_id, r_block2_id));
        }
      }
    }
  }

  m_.reset(new BlockRandomAccessSparseMatrix(blocks_, block_pairs));
}

// Initialize the SchurEliminator.
void SubsetSchurPreconditioner::InitEliminator() {
  LinearSolver::Options eliminator_options;
  eliminator_options.elimination_groups = options_.elimination_groups;
  eliminator_options.num_threads = options_.num_threads;
  eliminator_options.e_block_size = options_.e_block_size;
  eliminator_options.f_block_size = options_.f_block_size;
  eliminator_options.row_block_size = options_.row_block_size;
  eliminator_options.context = options_.context;
  eliminator_.reset(SchurEliminatorBase::Create(eliminator_options));
  const bool kFullRankETE = true;
  eliminator_->Init(eliminator_options.elimination_groups[0],
                    kFullRankETE,
                    subset_block_structure_.get());
}

// Update the values of the preconditioner matrix and factorize it.
bool SubsetSchurPreconditioner::UpdateImpl(const BlockSparseMatrix& A,
                                           const double* D) {
  const time_t start_time = time(NULL);
  const int num_rows = m_->num_rows();
  CHECK_GT(num_rows, 0);
  eliminator_->Eliminate(
      BlockSparseMatrixData(subset_block_structure_.get(), A.values()),
      nullptr,
      D,
      m_.get(),
      nullptr);

  const time_t elimination_time = time(NULL);
  LinearSolverTerminationType status = Factorize();
  const time_t factorize_time = time(NULL);
  VLOG(2) << "Update time: " << factorize_time - start_time
          << " elimination time: " << elimination_time - start_time
          << " factorize time: " << factorize_time - elimination_time;

  return (status == LINEAR_SOLVER_SUCCESS);
}

// Compute the sparse Cholesky factorization of the preconditioner
// matrix.
LinearSolverTerminationType SubsetSchurPreconditioner::Factorize() {
  // Extract the TripletSparseMatrix that is used for actually storing
  // S and convert it into a CompressedRowSparseMatrix.
  const TripletSparseMatrix* tsm =
      down_cast<BlockRandomAccessSparseMatrix*>(m_.get())->mutable_matrix();
  std::unique_ptr<CompressedRowSparseMatrix> lhs;
  const CompressedRowSparseMatrix::StorageType storage_type =
      sparse_cholesky_->StorageType();
  if (storage_type == CompressedRowSparseMatrix::UPPER_TRIANGULAR) {
    lhs.reset(CompressedRowSparseMatrix::FromTripletSparseMatrix(*tsm));
    lhs->set_storage_type(CompressedRowSparseMatrix::UPPER_TRIANGULAR);
  } else {
    lhs.reset(
        CompressedRowSparseMatrix::FromTripletSparseMatrixTransposed(*tsm));
    lhs->set_storage_type(CompressedRowSparseMatrix::LOWER_TRIANGULAR);
  }

  *lhs->mutable_col_blocks() = blocks_;
  *lhs->mutable_row_blocks() = blocks_;

  std::string message;
  return sparse_cholesky_->Factorize(lhs.get(), &message);
}

void SubsetSchurPreconditioner::RightMultiply(const double* x,double* y) const {
  CHECK(x != nullptr);
  CHECK(y != nullptr);
  CHECK(sparse_cholesky_ != nullptr);
  std::string message;
  sparse_cholesky_->Solve(x, y, &message);
}

int SubsetSchurPreconditioner::num_rows() const { return m_->num_rows(); }

}  // namespace internal
}  // namespace ceres
