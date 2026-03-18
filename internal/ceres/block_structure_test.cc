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

#include "ceres/block_structure.h"

#include <vector>

#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace ceres::internal {

TEST(BlockStructure, CellLessThan) {
  Cell a(1, 10);
  Cell b(1, 20);
  Cell c(2, 5);

  EXPECT_TRUE(CellLessThan(a, b));
  EXPECT_FALSE(CellLessThan(b, a));
  EXPECT_TRUE(CellLessThan(a, c));
  EXPECT_FALSE(CellLessThan(c, a));
}

TEST(BlockStructure, Tail) {
  std::vector<Block> blocks;
  blocks.emplace_back(2, 0);
  blocks.emplace_back(3, 2);
  blocks.emplace_back(4, 5);

  {
    std::vector<Block> tail = Tail(absl::MakeSpan(blocks), 0);
    EXPECT_EQ(tail.size(), 0);
  }

  {
    std::vector<Block> tail = Tail(absl::MakeSpan(blocks), 1);
    EXPECT_EQ(tail.size(), 1);
    EXPECT_EQ(tail[0].size, 4);
    EXPECT_EQ(tail[0].position, 0);
  }

  {
    std::vector<Block> tail = Tail(absl::MakeSpan(blocks), 2);
    EXPECT_EQ(tail.size(), 2);
    EXPECT_EQ(tail[0].size, 3);
    EXPECT_EQ(tail[0].position, 0);
    EXPECT_EQ(tail[1].size, 4);
    EXPECT_EQ(tail[1].position, 3);
  }

  {
    std::vector<Block> tail = Tail(absl::MakeSpan(blocks), 3);
    EXPECT_EQ(tail.size(), 3);
    EXPECT_EQ(tail[0].size, 2);
    EXPECT_EQ(tail[0].position, 0);
    EXPECT_EQ(tail[1].size, 3);
    EXPECT_EQ(tail[1].position, 2);
    EXPECT_EQ(tail[2].size, 4);
    EXPECT_EQ(tail[2].position, 5);
  }
}

TEST(BlockStructure, SumSquaredSizes) {
  std::vector<Block> blocks;
  blocks.emplace_back(2, 0);
  blocks.emplace_back(3, 2);
  blocks.emplace_back(4, 5);

  EXPECT_EQ(SumSquaredSizes(absl::MakeSpan(blocks)), 2 * 2 + 3 * 3 + 4 * 4);
}

TEST(BlockStructure, NumScalarEntries) {
  {
    std::vector<Block> blocks;
    EXPECT_EQ(NumScalarEntries(absl::MakeSpan(blocks)), 0);
  }

  {
    std::vector<Block> blocks;
    blocks.emplace_back(2, 0);
    blocks.emplace_back(3, 2);
    blocks.emplace_back(4, 5);
    EXPECT_EQ(NumScalarEntries(absl::MakeSpan(blocks)), 9);
  }
}

}  // namespace ceres::internal
