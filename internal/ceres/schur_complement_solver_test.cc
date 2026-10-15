// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "ceres/schur_complement_solver.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_format.h"
#include "ceres/block_sparse_matrix.h"
#include "ceres/block_structure.h"
#include "ceres/casts.h"
#include "ceres/context_impl.h"
#include "ceres/dense_sparse_matrix.h"
#include "ceres/detect_structure.h"
#include "ceres/internal/config.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/test_util.h"
#include "ceres/triplet_sparse_matrix.h"
#include "ceres/types.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres::internal {
namespace {

// The configuration of a Schur complement based linear solver.
struct SolverConfig {
  LinearSolverType type;
  DenseLinearAlgebraLibraryType dense_linear_algebra_library_type;
  SparseLinearAlgebraLibraryType sparse_linear_algebra_library_type;
  OrderingType ordering_type;
};

void PrintTo(const SolverConfig& config, std::ostream* os) {
  if (config.type == DENSE_SCHUR) {
    *os << absl::StrFormat("%s_%s",
                           LinearSolverTypeToString(config.type),
                           DenseLinearAlgebraLibraryTypeToString(
                               config.dense_linear_algebra_library_type));
  } else {
    *os << absl::StrFormat("%s_%s_",
                           LinearSolverTypeToString(config.type),
                           SparseLinearAlgebraLibraryTypeToString(
                               config.sparse_linear_algebra_library_type))
        << config.ordering_type;
  }
}

// Returns the configurations of the Schur complement solvers available in the
// current build.
std::vector<SolverConfig> SolverConfigs() {
  std::vector<SolverConfig> configs{
      {DENSE_SCHUR, EIGEN, NO_SPARSE, OrderingType::NATURAL},
#ifndef CERES_NO_LAPACK
      {DENSE_SCHUR, LAPACK, NO_SPARSE, OrderingType::NATURAL},
#endif  // CERES_NO_LAPACK
  };

  const auto add_sparse_configs =
      [&configs](SparseLinearAlgebraLibraryType library,
                 std::initializer_list<OrderingType> ordering_types) {
        for (const OrderingType ordering_type : ordering_types) {
          configs.push_back({SPARSE_SCHUR, EIGEN, library, ordering_type});
        }
      };
#ifndef CERES_NO_SUITESPARSE
  add_sparse_configs(SUITE_SPARSE,
                     {
                         OrderingType::NATURAL,
                         OrderingType::AMD,
#ifndef CERES_NO_CHOLMOD_PARTITION
                         OrderingType::NESDIS,
#endif  // CERES_NO_CHOLMOD_PARTITION
                     });
#endif  // CERES_NO_SUITESPARSE
#ifndef CERES_NO_ACCELERATE_SPARSE
  add_sparse_configs(ACCELERATE_SPARSE,
                     {OrderingType::AMD, OrderingType::NESDIS});
#endif  // CERES_NO_ACCELERATE_SPARSE
#ifdef CERES_USE_EIGEN_SPARSE
  add_sparse_configs(EIGEN_SPARSE,
                     {
                         OrderingType::NATURAL,
                         OrderingType::AMD,
#ifndef CERES_NO_EIGEN_METIS
                         OrderingType::NESDIS,
#endif  // CERES_NO_EIGEN_METIS
                     });
#endif  // CERES_USE_EIGEN_SPARSE
  return configs;
}

// Creates a random problem with the block structure required by the Schur
// complement solvers. Each row block contains at most one cell in the first
// num_e_blocks column blocks, which precedes its remaining cells, and the
// row blocks containing such a cell precede the remaining row blocks. Every
// column block is covered by more rows than columns, so the matrix has full
// column rank almost surely.
std::unique_ptr<LinearLeastSquaresProblem> CreateRandomSchurProblem(
    int num_e_blocks, int num_f_blocks, std::uint32_t seed) {
  constexpr int kMinBlockSize = 1;
  constexpr int kMaxBlockSize = 3;
  constexpr double kFBlockProbability = 0.5;
  std::mt19937 prng(seed);
  std::uniform_int_distribution<int> block_size(kMinBlockSize, kMaxBlockSize);
  std::bernoulli_distribution has_f_cell(kFBlockProbability);
  std::uniform_real_distribution<double> value(-1.0, 1.0);

  auto* bs = new CompressedRowBlockStructure;
  int num_cols = 0;
  for (int i = 0; i < num_e_blocks + num_f_blocks; ++i) {
    bs->cols.emplace_back(block_size(prng), num_cols);
    num_cols += bs->cols.back().size;
  }

  int num_rows = 0;
  int num_nonzeros = 0;
  // Appends row blocks containing the given column block until it is covered
  // by more rows than columns. Every row block also contains a random subset
  // of the F blocks. The cells are ordered by their column block.
  const auto add_row_blocks = [&](int col_block_id) {
    const int col_block_size = bs->cols[col_block_id].size;
    for (int covered_rows = 0; covered_rows <= col_block_size;) {
      CompressedRow row;
      row.block = Block(block_size(prng), num_rows);
      const auto add_cell = [&row, &num_nonzeros, bs](int cell_block_id) {
        row.cells.emplace_back(cell_block_id, num_nonzeros);
        num_nonzeros += row.block.size * bs->cols[cell_block_id].size;
      };
      if (col_block_id < num_e_blocks) {
        add_cell(col_block_id);
      }
      for (int f = num_e_blocks; f < num_e_blocks + num_f_blocks; ++f) {
        if (f == col_block_id || has_f_cell(prng)) {
          add_cell(f);
        }
      }
      num_rows += row.block.size;
      covered_rows += row.block.size;
      bs->rows.push_back(std::move(row));
    }
  };
  for (int col_block_id = 0; col_block_id < num_e_blocks + num_f_blocks;
       ++col_block_id) {
    add_row_blocks(col_block_id);
  }

  auto A = std::make_unique<BlockSparseMatrix>(bs);
  std::generate_n(A->mutable_values(), A->num_nonzeros(), [&value, &prng] {
    return value(prng);
  });

  auto problem = std::make_unique<LinearLeastSquaresProblem>();
  problem->A = std::move(A);
  problem->b = std::make_unique<double[]>(num_rows);
  std::generate_n(
      problem->b.get(), num_rows, [&value, &prng] { return value(prng); });
  problem->D = std::make_unique<double[]>(num_cols);
  std::generate_n(problem->D.get(), num_cols, [&value, &prng] {
    return 1.0 + std::abs(value(prng));
  });
  problem->num_eliminate_blocks = num_e_blocks;
  return problem;
}

// A linear least squares problem and whether to regularize it.
struct ProblemConfig {
  std::string name;
  std::function<std::unique_ptr<LinearLeastSquaresProblem>()> create;
  bool regularization;
};

void PrintTo(const ProblemConfig& config, std::ostream* os) {
  *os << absl::StrFormat(
      "%s_%s",
      config.name,
      config.regularization ? "Regularized" : "Unregularized");
}

std::vector<ProblemConfig> ProblemConfigs() {
  std::vector<ProblemConfig> configs;
  const auto add_problem =
      [&configs](
          const std::string& name,
          const std::function<std::unique_ptr<LinearLeastSquaresProblem>()>&
              create,
          bool full_rank) {
        if (full_rank) {
          configs.push_back({name, create, /*regularization=*/false});
        }
        configs.push_back({name, create, /*regularization=*/true});
      };

  // Problems 4 and 6 are rank deficient without the regularization.
  for (const auto& [id, full_rank] : {std::pair{2, true},
                                      std::pair{3, true},
                                      std::pair{4, false},
                                      std::pair{5, true},
                                      std::pair{6, false}}) {
    add_problem(
        absl::StrFormat("Problem%d", id),
        [id = id] { return CreateLinearLeastSquaresProblemFromId(id); },
        full_rank);
  }

  constexpr int kNumRandomProblems = 3;
  for (std::uint32_t seed = 0; seed < kNumRandomProblems; ++seed) {
    const int num_e_blocks = 2 + 3 * seed;
    const int num_f_blocks = 1 + 2 * seed;
    add_problem(
        absl::StrFormat("Random%dx%d", num_e_blocks, num_f_blocks),
        [num_e_blocks, num_f_blocks, seed] {
          return CreateRandomSchurProblem(num_e_blocks, num_f_blocks, seed);
        },
        /*full_rank=*/true);
  }
  return configs;
}

using Param = std::tuple<SolverConfig, ProblemConfig>;

std::string ParamInfoToString(const ::testing::TestParamInfo<Param>& info) {
  return absl::StrFormat("%s_%s",
                         ::testing::PrintToString(std::get<0>(info.param)),
                         ::testing::PrintToString(std::get<1>(info.param)));
}

class SchurComplementSolverTest : public ::testing::TestWithParam<Param> {};

// Compares the solution of the Schur complement solver to the one of the dense
// QR solver.
TEST_P(SchurComplementSolverTest, MatchesDenseQR) {
  const auto& [solver_config, problem_config] = GetParam();
  std::unique_ptr<LinearLeastSquaresProblem> problem = problem_config.create();
  ASSERT_NE(problem, nullptr);
  auto* A = down_cast<BlockSparseMatrix*>(problem->A.get());
  const int num_cols = A->num_cols();

  LinearSolver::PerSolveOptions per_solve_options;
  if (problem_config.regularization) {
    per_solve_options.D = problem->D.get();
  }

  ContextImpl context;
  LinearSolver::Options qr_options;
  qr_options.type = DENSE_QR;
  qr_options.context = &context;
  std::unique_ptr<LinearSolver> qr(LinearSolver::Create(qr_options));
  TripletSparseMatrix triplet_A(
      A->num_rows(), A->num_cols(), A->num_nonzeros());
  A->ToTripletSparseMatrix(&triplet_A);
  DenseSparseMatrix dense_A(triplet_A);
  Vector expected(num_cols);
  ASSERT_EQ(
      qr->Solve(&dense_A, problem->b.get(), per_solve_options, expected.data())
          .termination_type,
      LinearSolverTerminationType::SUCCESS);

  LinearSolver::Options options;
  options.type = solver_config.type;
  options.dense_linear_algebra_library_type =
      solver_config.dense_linear_algebra_library_type;
  options.sparse_linear_algebra_library_type =
      solver_config.sparse_linear_algebra_library_type;
  options.ordering_type = solver_config.ordering_type;
  options.context = &context;
  options.elimination_groups.push_back(problem->num_eliminate_blocks);
  options.elimination_groups.push_back(A->block_structure()->cols.size() -
                                       problem->num_eliminate_blocks);
  DetectStructure(*A->block_structure(),
                  problem->num_eliminate_blocks,
                  &options.row_block_size,
                  &options.e_block_size,
                  &options.f_block_size);
  std::unique_ptr<LinearSolver> solver(LinearSolver::Create(options));

  Vector actual(num_cols);
  const LinearSolver::Summary summary =
      solver->Solve(A, problem->b.get(), per_solve_options, actual.data());
  ASSERT_EQ(summary.termination_type, LinearSolverTerminationType::SUCCESS)
      << summary.message;

  // The tolerance applies to the norm of the difference divided by the number
  // of columns.
  constexpr double kTolerance = 1e-10;
  EXPECT_THAT(actual, MatrixNear(expected, kTolerance * num_cols));
}

INSTANTIATE_TEST_SUITE_P(
    SchurComplementSolver,
    SchurComplementSolverTest,
    ::testing::Combine(::testing::ValuesIn(SolverConfigs()),
                       ::testing::ValuesIn(ProblemConfigs())),
    ParamInfoToString);

}  // namespace
}  // namespace ceres::internal
