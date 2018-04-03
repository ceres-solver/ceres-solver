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

#include "ceres/parallel_for.h"

#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "ceres/context_impl.h"
#include "glog/logging.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

using testing::ElementsAreArray;
using testing::UnorderedElementsAreArray;

// Tests the parallel for loop computes the correct result for various number of
// threads.
TEST(ParallelFor, NumThreads) {
  ContextImpl context;
  context.EnsureMinimumThreads(/*num_threads=*/2);

  const int size = 16;
  std::vector<int> expected_results(size, 0);
  for (int i = 0; i < size; ++i) {
    expected_results[i] = std::sqrt(i);
  }

  for (int num_threads = 1; num_threads <= 8; ++num_threads) {
    std::vector<int> values(size, 0);
    ParallelFor(&context, 0, size, num_threads,
                [&values](int i) { values[i] = std::sqrt(i); });
    EXPECT_THAT(values, ElementsAreArray(expected_results));
  }
}

// Tests the parallel for loop with the thread ID interface computes the correct
// result for various number of threads.
TEST(ParallelForWithThreadId, NumThreads) {
  ContextImpl context;
  context.EnsureMinimumThreads(/*num_threads=*/2);

  const int size = 16;
  std::vector<int> expected_results(size, 0);
  for (int i = 0; i < size; ++i) {
    expected_results[i] = std::sqrt(i);
  }

  for (int num_threads = 1; num_threads <= 8; ++num_threads) {
    std::vector<int> values(size, 0);
    ParallelFor(&context, 0, size, num_threads,
                [&values](int thread_id, int i) { values[i] = std::sqrt(i); });
    EXPECT_THAT(values, ElementsAreArray(expected_results));
  }
}

// Tests nested for loops do not result in a deadlock.
TEST(ParallelFor, NestedParallelForDeadlock) {
  ContextImpl context;
  context.EnsureMinimumThreads(/*num_threads=*/2);

  // Increment each element in the 2D matrix.
  std::vector<std::vector<int>> x(3, {1, 2, 3});
  ParallelFor(&context, 0, 3, 2, [&x, &context](int i) {
    std::vector<int>& y = x.at(i);
    ParallelFor(&context, 0, 3, 2, [&y](int j) { ++y.at(j); });
  });

  const std::vector<int> results = {2, 3, 4};
  for (const std::vector<int>& value : x) {
    EXPECT_THAT(value, ElementsAreArray(results));
  }
}

// Tests nested for loops do not result in a deadlock for the parallel for with
// thread ID interface.
TEST(ParallelForWithThreadId, NestedParallelForDeadlock) {
  ContextImpl context;
  context.EnsureMinimumThreads(/*num_threads=*/2);

  // Increment each element in the 2D matrix.
  std::vector<std::vector<int>> x(3, {1, 2, 3});
  ParallelFor(&context, 0, 3, 2, [&x, &context](int thread_id, int i) {
    std::vector<int>& y = x.at(i);
    ParallelFor(&context, 0, 3, 2, [&y](int thread_id, int j) { ++y.at(j); });
  });

  const std::vector<int> results = {2, 3, 4};
  for (const std::vector<int>& value : x) {
    EXPECT_THAT(value, ElementsAreArray(results));
  }
}

// This test is only valid when multithreading support is enabled.
#ifndef CERES_NO_THREADS
TEST(ParallelForWithThreadId, UniqueThreadIds) {
  // Ensure the hardware supports more than 1 thread to ensure the test will
  // pass.
  const int num_hardware_threads = std::thread::hardware_concurrency();
  if (num_hardware_threads <= 1) {
    LOG(ERROR)
        << "Test not supported, the hardware does not support threading.";
    return;
  }

  ContextImpl context;
  context.EnsureMinimumThreads(/*num_threads=*/2);
  // Increment each element in the 2D matrix.
  std::vector<int> x(2, -1);
  std::mutex mutex;
  std::condition_variable condition;
  int count = 0;
  ParallelFor(&context, 0, 2, 2,
              [&x, &mutex, &condition, &count](int thread_id, int i) {
                std::unique_lock<std::mutex> lock(mutex);
                x[i] = thread_id;
                ++count;
                condition.notify_all();
                condition.wait(lock, [&]() { return count == 2; });
              });

  EXPECT_THAT(x, UnorderedElementsAreArray({0,1}));
}
#endif  // CERES_NO_THREADS

}  // namespace internal
}  // namespace ceres
