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

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "ceres/concurrent_queue.h"
#include "ceres/parallel_for.h"
#include "ceres/scoped_thread_token.h"
#include "ceres/thread_token_provider.h"
#include "glog/logging.h"

namespace ceres::internal {
namespace {
// This class creates a thread safe barrier which will block until a
// pre-specified number of threads call Finished.  This allows us to block the
// main thread until all the parallel threads are finished processing all the
// work.
class BlockUntilFinished {
 public:
  explicit BlockUntilFinished(int num_total_jobs)
      : num_total_jobs_finished_(0), num_total_jobs_(num_total_jobs) {}

  // Increment the number of jobs that have been processed by the number of
  // jobs processed by caller and signal the blocking thread if all jobs
  // have finished.
  void Finished(int num_jobs_finished) {
    if (num_jobs_finished == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    num_total_jobs_finished_ += num_jobs_finished;
    CHECK_LE(num_total_jobs_finished_, num_total_jobs_);
    if (num_total_jobs_finished_ == num_total_jobs_) {
      condition_.notify_one();
    }
  }

  // Block until receiving confirmation of all jobs being finished.
  void Block() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(
        lock, [&]() { return num_total_jobs_finished_ == num_total_jobs_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  int num_total_jobs_finished_;
  int num_total_jobs_;
};

// Shared state between the parallel tasks. Each thread will use this
// information to get the next block of work to be performed.
struct ThreadPoolState {
  // The entire range [start, end) is split into num_work_blocks contiguous
  // disjoint intervals (blocks), which are as equal as possible given
  // total index count and requested number of  blocks.
  //
  // Those num_work_blocks blocks are then processed by num_workers
  // workers
  //
  // Total number of integer indices in interval [start, end) is
  // end - start, and when splitting them into num_work_blocks blocks
  // we can either
  //  - Split into equal blocks when (end - start) is divisible by
  //    num_work_blocks
  //  - Split into blocks with size difference at most 1:
  //     - Size of the smallest block(s) is (end - start) / num_work_blocks
  //     - (end - start) % num_work_blocks will need to be 1 index larger
  //
  // Note that this splitting is optimal in the sense of maximal difference
  // between block sizes, since splitting into equal blocks is possible
  // if and only if number of indices is divisible by number of blocks.
  ThreadPoolState(int start, int end, int num_work_blocks, int num_workers)
      : start(start),
        end(end),
        num_work_blocks(num_work_blocks),
        base_block_size((end - start) / num_work_blocks),
        num_base_p1_sized_blocks((end - start) % num_work_blocks),
        block_id(0),
        thread_token_provider(num_workers),
        block_until_finished(num_work_blocks) {}

  // The start and end index of the for loop.
  const int start;
  const int end;
  // The number of blocks that need to be processed.
  const int num_work_blocks;
  // Size of the smallest block
  const int base_block_size;
  // Number of blocks of size base_block_size + 1
  const int num_base_p1_sized_blocks;

  // The next block of work to be assigned to a worker.  The parallel for loop
  // range is split into num_work_blocks blocks of work, with a single block of
  // work being of size
  //  - base_block_size + 1 for the first num_base_p1_sized_blocks blocks
  //  - base_block_size for the rest blocks
  //  blocks of indices are contiguous and disjoint
  std::atomic<int> block_id;

  // Provides a unique thread ID among all active threads working on the same
  // group of tasks.  Thread-safe.
  ThreadTokenProvider thread_token_provider;

  // Used to signal when all the work has been completed.  Thread safe.
  BlockUntilFinished block_until_finished;
};

}  // namespace

int MaxNumThreadsAvailable() { return ThreadPool::MaxNumThreadsAvailable(); }

// Maximal number of work items scheduled for a single thread
//  - Lower number of work items results in larger runtimes on unequal tasks
//  - Higher number of work items results in larger losses for synchronization
const int kWorkBlocksPerThread = 4;

// See ParallelFor (below) for more details.
void ParallelFor(ContextImpl* context,
                 int start,
                 int end,
                 int num_threads,
                 const std::function<void(int, int)>& function) {
  CHECK_GT(num_threads, 0);
  if (start >= end) {
    return;
  }

  // Fast path for when it is single threaded.
  if (num_threads == 1) {
    function(start, end);
    return;
  }

  ParallelFor(context,
              start,
              end,
              num_threads,
              [&function](int /*thread_id*/, int start, int end) {
                function(start, end);
              });
}

// This implementation uses a fixed size max worker pool with a shared task
// queue. The problem of executing the function for the interval of [start, end)
// is broken up into at most num_threads * kWorkBlocksPerThread blocks
// and added to the thread pool. To avoid deadlocks, the calling thread is
// allowed to steal work from the worker pool.
// This is implemented via a shared state between the tasks. In order for
// the calling thread or thread pool to get a block of work, it will query the
// shared state for the next block of work to be done. If there is nothing left,
// it will return. We will exit the ParallelFor call when all of the work has
// been done, not when all of the tasks have been popped off the task queue.
//
// A unique thread ID among all active tasks will be acquired once for each
// block of work.  This avoids the significant performance penalty for acquiring
// it on every iteration of the for loop. The thread ID is guaranteed to be in
// [0, num_threads).
//
// A performance analysis has shown this implementation is on par with OpenMP
// and TBB.
void ParallelFor(
    ContextImpl* context,
    int start,
    int end,
    int num_threads,
    const std::function<void(int thread_id, int start, int end)>& function) {
  CHECK_GT(num_threads, 0);
  // Fast path for when it is single threaded.
  if (num_threads == 1) {
    // Even though we only have one thread, use the thread token provider to
    // guarantee the exact same behavior when running with multiple threads.
    ThreadTokenProvider thread_token_provider(num_threads);
    const ScopedThreadToken scoped_thread_token(&thread_token_provider);
    const int thread_id = scoped_thread_token.token();
    function(thread_id, start, end);
    return;
  }

  CHECK(context != nullptr);
  if (end <= start) {
    return;
  }

  // Interval [start, end) is being split into
  // num_threads * kWorkBlocksPerThread contiguous disjoint blocks.
  //
  // In order to avoid creating empty blocks of work, we need to limit
  // number of work blocks by a total number of indices.
  const int num_work_blocks =
      std::min((end - start), num_threads * kWorkBlocksPerThread);

  // We use a std::shared_ptr because the main thread can finish all
  // the work before the tasks have been popped off the queue.  So the
  // shared state needs to exist for the duration of all the tasks.
  std::shared_ptr<ThreadPoolState> shared_state(
      new ThreadPoolState(start, end, num_work_blocks, num_threads));

  // A function which tries to perform several chunks of work.
  auto task_function = [shared_state, &function]() {
    int num_jobs_finished = 0;
    const ScopedThreadToken scoped_thread_token(
        &shared_state->thread_token_provider);
    const int thread_id = scoped_thread_token.token();
    const int start = shared_state->start;
    const int end = shared_state->end;
    const int base_block_size = shared_state->base_block_size;
    const int num_base_p1_sized_blocks = shared_state->num_base_p1_sized_blocks;

    while (true) {
      // Get the next available chunk of work to be performed. If there is no
      // work, return.
      int block_id = shared_state->block_id++;
      if (block_id >= shared_state->num_work_blocks) {
        break;
      }
      ++num_jobs_finished;

      // For-loop interval [start, end) was split into num_work_blocks,
      // with num_base_p1_sized_blocks of size base_block_size + 1 and remaining
      // num_work_blocks - num_base_p1_sized_blocks of size base_block_size
      //
      // Then, start index of the block #block_id is given by a total
      // length of preceeding blocks:
      //  * Total length of preceeding blocks of size base_block_size + 1:
      //     min(block_id, num_base_p1_sized_blocks) * (base_block_size + 1)
      //
      //  * Total length of preceeding blocks of size base_block_size:
      //     (block_id - min(block_id, num_base_p1_sized_blocks)) *
      //     base_block_size
      //
      // Simplifying sum of those quantities yields a following
      // expression for start index of the block #block_id
      const int curr_start = start + block_id * base_block_size +
                             std::min(block_id, num_base_p1_sized_blocks);
      // First num_base_p1_sized_blocks have size base_block_size + 1
      //
      // Note that it is guaranteed that all blocks are within
      // [start, end) interval
      const int curr_end = curr_start + base_block_size +
                           (block_id < num_base_p1_sized_blocks ? 1 : 0);
      // Perform each task in current block
      function(thread_id, curr_start, curr_end);
    }
    shared_state->block_until_finished.Finished(num_jobs_finished);
  };

  // Add all the tasks to the thread pool.
  for (int i = 0; i < num_threads; ++i) {
    // Note we are taking the task_function as value so the shared_state
    // shared pointer is copied and the ref count is increased. This is to
    // prevent it from being deleted when the main thread finishes all the
    // work and exits before the threads finish.
    context->thread_pool.AddTask([task_function]() { task_function(); });
  }

  // Try to do any available work on the main thread. This may steal work from
  // the thread pool, but when there is no work left the thread pool tasks
  // will be no-ops.
  task_function();

  // Wait until all tasks have finished.
  shared_state->block_until_finished.Block();
}

}  // namespace ceres::internal

#endif  // CERES_USE_CXX_THREADS
