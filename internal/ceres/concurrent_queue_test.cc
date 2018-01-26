// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2018 Google Inc. All rights reserved.
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
// Author: vitus@google.com (Michael Vitus)

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifdef CERES_USE_CXX11_THREADS

#include "ceres/concurrent_queue.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

TEST(ConcurrentQueue, PushPop) {
  ConcurrentQueue<int> queue;

  int num_to_add = 10;
  for (int i = 0; i < num_to_add; ++i) {
    queue.Push(i);
  }

  for (int i = 0; i < num_to_add; ++i) {
    int value;
    ASSERT_TRUE(queue.Pop(&value));
    EXPECT_EQ(i, value);
  }
}

TEST(ConcurrentQueue, Stop) {
  ConcurrentQueue<int> queue;

  queue.Push(123);
  int value;
  ASSERT_TRUE(queue.Pop(&value));
  EXPECT_EQ(123, value);

  queue.Push(456);
  queue.Abort();

  EXPECT_FALSE(queue.Pop(&value));

  // Try to push another element into the queue after abort has been called.
  queue.Push(456);
  EXPECT_FALSE(queue.Pop(&value));

  // Call abort again which should not change the state of the queue.
  queue.Abort();
  EXPECT_FALSE(queue.Pop(&value));
}

}  // namespace internal
}  // namespace ceres

#endif // CERES_USE_CXX11_THREADS
