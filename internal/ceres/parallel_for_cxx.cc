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
#include "ceres/internal/config.h"

#ifdef CERES_USE_CXX_THREADS

#include <cmath>
#include <limits>
#include <memory>
#include <thread>
#include <unsupported/Eigen/CXX11/ThreadPool>

#include "ceres/parallel_for.h"
#include "glog/logging.h"

namespace ceres::internal {

namespace {
int NumShardsUsedByFixedBlockSizeScheduling(const int total,
                                            const int block_size,
                                            const int num_threads) {
  if (block_size <= 0 || total <= 1 || total <= block_size ||
      num_threads == 1) {
    return 1;
  }
  return (total + block_size - 1) / block_size;
}
}  // namespace

// See ParallelFor (below) for more details.
void ParallelFor(ContextImpl* context, int start, int end,
                 const std::function<void(int start, int end)>& function) {
  CHECK(context != nullptr);
  if (end <= start) {
    return;
  }

  // Fast path for when there is no thread pool or called from a child thread to
  // prevent a deadlock for nested parallel fors.
  if (context->eigen_thread_pool_ == nullptr ||
      context->eigen_thread_pool_->CurrentThreadId() != -1) {
    function(start, end);
    return;
  }

  ParallelFor(context, start, end,
              [&function](int /*thread_id*/, int start, int end) {
                function(start, end);
              });
}

// The thread ID is guaranteed to be in
// [0, num_threads]
void ParallelFor(
    ContextImpl* context, int start, int end,
    const std::function<void(int thread_id, int start, int end)>& function) {
  CHECK(context != nullptr);
  if (end <= start) {
    return;
  }

  // Fast path for when there is no thread pool or called from a child thread to
  // prevent a deadlock for nested parallel fors.
  if (context->eigen_thread_pool_ == nullptr ||
      context->eigen_thread_pool_->CurrentThreadId() != -1) {
    function(/*thread_id=*/0, start, end);
    return;
  }

  CHECK(context->eigen_thread_pool_ != nullptr);

  // Choose a block size that corresponds to 4 * num threads.
  const int block_size =
      (end - start) / (4 * context->eigen_thread_pool_->NumThreads()) + 1;
  const int num_shards_used = NumShardsUsedByFixedBlockSizeScheduling(
      end - start, block_size, context->eigen_thread_pool_->NumThreads());

  if (num_shards_used == 1) {
    function(/*thread_id=*/0, start, end);
    return;
  }

  // Adapted from Eigen's parallelFor implementation.
  Eigen::Barrier barrier(num_shards_used);
  std::function<void(int, int)> handle_range =
      [=, &handle_range, &barrier, &function](int first, int last) {
        while (last - first > block_size) {
          // Find something near the midpoint which is a multiple of block size.
          const int mid = first + ((last - first) / 2 + block_size - 1) /
                                      block_size * block_size;
          context->eigen_thread_pool_->Schedule(
              [=, &handle_range]() { handle_range(mid, last); });
          last = mid;
        }

        // We may use the current thread to do some work synchronously.
        // When calling CurrentThreadId() from outside of the thread
        // pool, we get -1, so we can shift every id up by 1.
        const int thread_id =
            context->eigen_thread_pool_->CurrentThreadId() + 1;

        // Single block or less, execute directly.
        function(thread_id, first, last);
        barrier.Notify();  // The shard is done.
      };

  if (num_shards_used <= context->eigen_thread_pool_->NumThreads()) {
    // Avoid a thread hop by running the root of the tree and one block on the
    // main thread.
    handle_range(start, end);
  } else {
    // Execute the root in the thread pool to avoid running work on more than
    // numThreads() threads.
    context->eigen_thread_pool_->Schedule(
        [=, &handle_range]() { handle_range(start, end); });
  }
  barrier.Wait();
}

}  // namespace ceres::internal

#endif  // CERES_USE_CXX_THREADS
