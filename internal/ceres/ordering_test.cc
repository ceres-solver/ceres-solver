// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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

#include "ceres/ordering.h"

#include <cstddef>
#include <vector>
#include "gtest/gtest.h"
#include "ceres/collections_port.h"

namespace ceres {
namespace internal {

TEST(Ordering, EmptyOrderingBehavesCorrectly) {
  Ordering ordering;
  EXPECT_EQ(ordering.NumGroups(), 0);
  EXPECT_EQ(ordering.NumParameterBlocks(), 0);
  EXPECT_EQ(ordering.GroupSize(1), 0);
  double x;
  EXPECT_DEATH_IF_SUPPORTED(ordering.GroupIdForParameterBlock(&x),
                            "Tried finding");
  EXPECT_DEATH_IF_SUPPORTED(ordering.RemoveParameterBlock(&x),
                            "Tried finding");
}

TEST(Ordering, EverythingInOneGroup) {
  Ordering ordering;
  double x[3];
  ordering.AddParameterBlockToGroup(x, 1);
  ordering.AddParameterBlockToGroup(x + 1, 1);
  ordering.AddParameterBlockToGroup(x + 2, 1);
  ordering.AddParameterBlockToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumParameterBlocks(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 1);

  ordering.RemoveParameterBlock(x);
  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumParameterBlocks(), 2);
  EXPECT_EQ(ordering.GroupSize(1), 2);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_DEATH_IF_SUPPORTED(ordering.GroupIdForParameterBlock(x),
                            "Tried finding");
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 1);
}

TEST(Ordering, StartInOneGroupAndThenSplit) {
  Ordering ordering;
  double x[3];
  ordering.AddParameterBlockToGroup(x, 1);
  ordering.AddParameterBlockToGroup(x + 1, 1);
  ordering.AddParameterBlockToGroup(x + 2, 1);
  ordering.AddParameterBlockToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumParameterBlocks(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 1);

  ordering.AddParameterBlockToGroup(x, 5);
  EXPECT_EQ(ordering.NumGroups(), 2);
  EXPECT_EQ(ordering.NumParameterBlocks(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 2);
  EXPECT_EQ(ordering.GroupSize(5), 1);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_EQ(ordering.GroupIdForParameterBlock(x), 5);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 1);
}

TEST(Ordering, AddAndRemoveEveryThingFromOneGroup) {
  Ordering ordering;
  double x[3];
  ordering.AddParameterBlockToGroup(x, 1);
  ordering.AddParameterBlockToGroup(x + 1, 1);
  ordering.AddParameterBlockToGroup(x + 2, 1);
  ordering.AddParameterBlockToGroup(x, 1);

  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumParameterBlocks(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 1);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 1);

  ordering.AddParameterBlockToGroup(x, 5);
  ordering.AddParameterBlockToGroup(x + 1, 5);
  ordering.AddParameterBlockToGroup(x + 2, 5);
  EXPECT_EQ(ordering.NumGroups(), 1);
  EXPECT_EQ(ordering.NumParameterBlocks(), 3);
  EXPECT_EQ(ordering.GroupSize(1), 0);
  EXPECT_EQ(ordering.GroupSize(5), 3);
  EXPECT_EQ(ordering.GroupSize(0), 0);

  EXPECT_EQ(ordering.GroupIdForParameterBlock(x), 5);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 1), 5);
  EXPECT_EQ(ordering.GroupIdForParameterBlock(x + 2), 5);
}

}  // namespace internal
}  // namespace ceres
