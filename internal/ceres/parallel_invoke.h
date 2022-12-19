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
// Authors: vitus@google.com (Michael Vitus),
//          dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#ifndef CERES_INTERNAL_PARALLEL_INVOKE_H_
#define CERES_INTERNAL_PARALLEL_INVOKE_H_

#include <memory>
#include <tuple>
#include <type_traits>

#include "ceres/parallel_for_synchronization.h"

namespace ceres::internal {

// Get arguments of callable as a tuple
template <typename F, typename... Args>
std::tuple<std::decay_t<Args>...> args_of(void (F::*)(Args...) const);

template <typename F>
using args_of_t = decltype(args_of(&F::operator()));

template <typename F>
struct InvokeOnSegmentTraits {
  using Args = args_of_t<std::remove_reference_t<F>>;
  static constexpr int NumArgs = std::tuple_size_v<Args>;
  using FirstArg = typename std::tuple_element<0, Args>::type;
  using LastArg = typename std::tuple_element<NumArgs - 1, Args>::type;

  static constexpr bool FirstArgIsInt = std::is_same_v<FirstArg, int>;
  static constexpr bool LastArgIsInt = std::is_same_v<LastArg, int>;

  // Functions having at least 2 arguments with the first of them being int
  // will be passed thread_id as the first argument.
  static constexpr bool PassThreadId = NumArgs > 1 && FirstArgIsInt;
  // Loop over indexes corresponding to segment is added for functions with 1 or
  // 2 int arguments
  static constexpr bool AddLoop = LastArgIsInt && FirstArgIsInt && NumArgs <= 2;
};

// InvokeWithThreadId handles passing thread_id to the function
template <typename F,
          typename... Args,
          std::enable_if_t<InvokeOnSegmentTraits<F>::PassThreadId, bool> = true>
void InvokeWithThreadId(int thread_id, F&& function, Args&&... args) {
  function(thread_id, std::forward<Args>(args)...);
}

template <
    typename F,
    typename... Args,
    std::enable_if_t<!InvokeOnSegmentTraits<F>::PassThreadId, bool> = true>
void InvokeWithThreadId(int /* thread_id */, F&& function, Args&&... args) {
  function(std::forward<Args>(args)...);
}

// InvokeOnSegment either runs a loop over segment indices or passes it to the
// function
template <typename F,
          std::enable_if_t<InvokeOnSegmentTraits<F>::AddLoop, bool> = true>
void InvokeOnSegment(int thread_id, std::tuple<int, int> range, F&& function) {
  auto [start, end] = range;
  for (int i = start; i != end; ++i) {
    InvokeWithThreadId(thread_id, function, i);
  }
}

template <typename F,
          std::enable_if_t<!InvokeOnSegmentTraits<F>::AddLoop, bool> = true>
void InvokeOnSegment(int thread_id, std::tuple<int, int> range, F&& function) {
  InvokeWithThreadId(thread_id, function, range);
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
template <typename F>
void ParallelInvoke(
    ContextImpl* context, int start, int end, int num_threads, F&& function) {
  CHECK(context != nullptr);

  // Maximal number of work items scheduled for a single thread
  //  - Lower number of work items results in larger runtimes on unequal tasks
  //  - Higher number of work items results in larger losses for synchronization
  constexpr int kWorkBlocksPerThread = 4;

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
  auto shared_state =
      std::make_shared<ThreadPoolState>(start, end, num_work_blocks);

  // A function which tries to perform several chunks of work.
  auto task = [shared_state, num_threads, &function]() {
    int num_jobs_finished = 0;
    const int thread_id = shared_state->thread_id.fetch_add(1);
    // In order to avoid dead-locks in nested parallel for loops, task() will be
    // invoked num_threads + 1 times:
    //  - num_threads times via enqueueing task into thread pool
    //  - one more time in the main thread
    //  Tasks enqueued to thread pool might take some time before execution, and
    //  the last task being executed will be terminated here in order to avoid
    //  having more than num_threads active threads
    if (thread_id >= num_threads) return;

    const int start = shared_state->start;
    const int base_block_size = shared_state->base_block_size;
    const int num_base_p1_sized_blocks = shared_state->num_base_p1_sized_blocks;
    const int num_work_blocks = shared_state->num_work_blocks;

    while (true) {
      // Get the next available chunk of work to be performed. If there is no
      // work, return.
      int block_id = shared_state->block_id.fetch_add(1);
      if (block_id >= num_work_blocks) {
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
      const auto range = std::make_tuple(curr_start, curr_end);
      InvokeOnSegment(thread_id, range, function);
    }
    shared_state->block_until_finished.Finished(num_jobs_finished);
  };

  // Add all the tasks to the thread pool.
  for (int i = 0; i < num_threads; ++i) {
    // Note we are taking the task as value so the copy of shared_state shared
    // pointer (captured by value at declaration of task lambda-function) is
    // copied and the ref count is increased. This is to prevent it from being
    // deleted when the main thread finishes all the work and exits before the
    // threads finish.
    context->thread_pool.AddTask([task]() { task(); });
  }

  // Try to do any available work on the main thread. This may steal work from
  // the thread pool, but when there is no work left the thread pool tasks
  // will be no-ops.
  task();

  // Wait until all tasks have finished.
  shared_state->block_until_finished.Block();
}

}  // namespace ceres::internal

#endif
