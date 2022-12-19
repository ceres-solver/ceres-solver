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
// Authors: vitus@google.com (Michael Vitus),
//          dmitriy.korchemkin@gmail.com (Dmitriy Korchemkin)

#ifndef CERES_INTERNAL_PARALLEL_FOR_H_
#define CERES_INTERNAL_PARALLEL_FOR_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "ceres/context_impl.h"
#include "ceres/internal/config.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "glog/logging.h"

namespace ceres::internal {

// Use a dummy mutex if num_threads = 1.
inline decltype(auto) MakeConditionalLock(const int num_threads,
                                          std::mutex& m) {
  return (num_threads == 1) ? std::unique_lock<std::mutex>{}
                            : std::unique_lock<std::mutex>{m};
}

// Returns the maximum number of threads supported by the threading backend
// Ceres was compiled with.
CERES_NO_EXPORT
int MaxNumThreadsAvailable();

// Parallel for implementations share a common set of routines in order
// to enforce inlining of loop bodies, ensuring that single-threaded
// performance is equivalent to a simple for loop
namespace parallel_for_details {
// Get arguments of callable as a tuple
template <typename F, typename... Args>
std::tuple<std::decay_t<Args>...> args_of(void (F::*)(Args...) const);

template <typename F>
using args_of_t = decltype(args_of(&F::operator()));

// Parallelizable functions might require passing thread_id as the first
// argument. This class supplies thread_id argument to functions that
// support it and ignores it otherwise.
template <typename F, typename Args>
struct InvokeImpl;

// For parallel for iterations of type [](int i) -> void
template <typename F>
struct InvokeImpl<F, std::tuple<int>> {
  static void InvokeOnSegment(int thread_id,
                              std::tuple<int, int> range,
                              const F& function) {
    (void)thread_id;
    auto [start, end] = range;
    for (int i = start; i < end; ++i) {
      function(i);
    }
  }
};

// For parallel for iterations of type [](int thread_id, int i) -> void
template <typename F>
struct InvokeImpl<F, std::tuple<int, int>> {
  static void InvokeOnSegment(int thread_id,
                              std::tuple<int, int> range,
                              const F& function) {
    auto [start, end] = range;
    for (int i = start; i < end; ++i) {
      function(thread_id, i);
    }
  }
};

// For parallel for iterations of type [](tuple<int, int> range) -> void
template <typename F>
struct InvokeImpl<F, std::tuple<std::tuple<int, int>>> {
  static void InvokeOnSegment(int thread_id,
                              std::tuple<int, int> range,
                              const F& function) {
    (void)thread_id;
    function(range);
  }
};

// For parallel for iterations of type [](int thread_id, tuple<int, int> range)
// -> void
template <typename F>
struct InvokeImpl<F, std::tuple<int, std::tuple<int, int>>> {
  static void InvokeOnSegment(int thread_id,
                              std::tuple<int, int> range,
                              const F& function) {
    function(thread_id, range);
  }
};

// Invoke function on indices from contiguous range according to function
// signature. The following signatures are supported:
//  - Functions processing single index per call:
//    - [](int index) -> void
//    - [](int thread_id, int index) -> void
//  - Functions processing contiguous range [start, end) of indices per call:
//    - [](std::tuple<int, int> range) -> void
//    - [](int thread_id, std::tuple<int, int> range) -> void
// Function arguments might have reference type and const qualifier
template <typename F>
void InvokeOnSegment(int thread_id,
                     std::tuple<int, int> range,
                     const F& function) {
  InvokeImpl<F, args_of_t<F>>::InvokeOnSegment(thread_id, range, function);
}

// Check if it is possible to split range [start; end) into at most
// max_num_partitions  contiguous partitions of cost not greater than
// max_partition_cost. Inclusive integer cumulative costs are provided by
// cumulative_cost_data objects, with cumulative_cost_offset being a total cost
// of all indices (starting from zero) preceding start element. Cumulative costs
// are returned by cumulative_cost_fun called with a reference to
// cumulative_cost_data element with index from range[start; end), and should be
// non-decreasing. Partition of the range is returned via partition argument
template <typename CumulativeCostData, typename CumulativeCostFun>
bool MaxPartitionCostIsFeasible(int start,
                                int end,
                                int max_num_partitions,
                                int max_partition_cost,
                                int cumulative_cost_offset,
                                const CumulativeCostData* cumulative_cost_data,
                                const CumulativeCostFun& cumulative_cost_fun,
                                std::vector<int>* partition) {
  partition->clear();
  partition->push_back(start);
  int partition_start = start;
  int cost_offset = cumulative_cost_offset;

  while (partition_start < end) {
    // Already have max_num_partitions
    if (partition->size() > max_num_partitions) {
      return false;
    }
    const int target = max_partition_cost + cost_offset;
    const int partition_end =
        std::partition_point(
            cumulative_cost_data + partition_start,
            cumulative_cost_data + end,
            [&cumulative_cost_fun, target](const CumulativeCostData& item) {
              return cumulative_cost_fun(item) <= target;
            }) -
        cumulative_cost_data;
    // Unable to make a partition from a single element
    if (partition_end == partition_start) {
      return false;
    }

    const int cost_last =
        cumulative_cost_fun(cumulative_cost_data[partition_end - 1]);
    partition->push_back(partition_end);
    partition_start = partition_end;
    cost_offset = cost_last;
  }
  return true;
}

// Split integer interval [start, end) into at most max_num_partitions
// contiguous intervals, minimizing maximal total cost of a single interval.
// Inclusive integer cumulative costs for each (zero-based) index are provided
// by cumulative_cost_data objects, and are returned by cumulative_cost_fun call
// with a reference to one of the objects from range [start, end)
template <typename CumulativeCostData, typename CumulativeCostFun>
std::vector<int> ComputePartition(
    int start,
    int end,
    int max_num_partitions,
    const CumulativeCostData* cumulative_cost_data,
    const CumulativeCostFun& cumulative_cost_fun) {
  // Given maximal partition cost, it is possible to verify if it is admissible
  // and obtain corresponding partition using MaxPartitionCostIsFeasible
  // function. In order to find the lowest admissible value, a binary search
  // over all potentially optimal cost values is being performed
  const int cumulative_cost_last =
      cumulative_cost_fun(cumulative_cost_data[end - 1]);
  const int cumulative_cost_offset =
      start ? cumulative_cost_fun(cumulative_cost_data[start - 1]) : 0;
  const int total_cost = cumulative_cost_last - cumulative_cost_offset;

  // Minimal maximal partition cost is not smaller than the average
  // We will use non-inclusive lower bound
  int partition_cost_lower_bound = total_cost / max_num_partitions - 1;
  // Minimal maximal partition cost is not larger than the total cost
  // Upper bound is inclusive
  int partition_cost_upper_bound = total_cost;

  std::vector<int> partition, partition_upper_bound;
  // Binary search over partition cost, returning the lowest admissible cost
  while (partition_cost_upper_bound - partition_cost_lower_bound > 1) {
    partition.reserve(max_num_partitions + 1);
    const int partition_cost =
        partition_cost_lower_bound +
        (partition_cost_upper_bound - partition_cost_lower_bound) / 2;
    bool admissible = MaxPartitionCostIsFeasible(start,
                                                 end,
                                                 max_num_partitions,
                                                 partition_cost,
                                                 cumulative_cost_offset,
                                                 cumulative_cost_data,
                                                 cumulative_cost_fun,
                                                 &partition);
    if (admissible) {
      partition_cost_upper_bound = partition_cost;
      std::swap(partition, partition_upper_bound);
    } else {
      partition_cost_lower_bound = partition_cost;
    }
  }

  // After binary search over partition cost, interval
  // (partition_cost_lower_bound, partition_cost_upper_bound] contains the only
  // admissible partition cost value - partition_cost_upper_bound
  //
  // Partition for this cost value might have been already computed
  if (partition_upper_bound.empty() == false) {
    return partition_upper_bound;
  }
  // Partition for upper bound is not computed if and only if upper bound was
  // never updated This is a simple case of a single interval containing all
  // values, which we were not able to break into pieces
  partition = {start, end};
  return partition;
}
}  // namespace parallel_for_details

// Forward declaration of parallel invocation function that is to be
// implemented by each threading backend
template <typename F>
void ParallelInvoke(ContextImpl* context,
                    int start,
                    int end,
                    int num_threads,
                    const F& function);

// Execute the function for every element in the range [start, end) with at most
// num_threads. It will execute all the work on the calling thread if
// num_threads or (end - start) is equal to 1.
//
// Depending on function signature, it can be supplied with thread_id; functions
// operating on a single loop index and on a contiguous range of loop indices
// are supported.
template <typename F>
void ParallelFor(ContextImpl* context,
                 int start,
                 int end,
                 int num_threads,
                 const F& function) {
  using namespace parallel_for_details;
  CHECK_GT(num_threads, 0);
  if (start >= end) {
    return;
  }

  if (num_threads == 1 || end - start == 1) {
    InvokeOnSegment<F>(0, std::make_tuple(start, end), function);
    return;
  }

  CHECK(context != nullptr);
  ParallelInvoke<F>(context, start, end, num_threads, function);
}

// Execute function for every element in the range [start, end) with at most
// num_threads, using the user provided partitioning. taking into account
// user-provided integer cumulative costs of iterations.
template <typename F>
void ParallelFor(ContextImpl* context,
                 int start,
                 int end,
                 int num_threads,
                 const F& function,
                 const std::vector<int>& partitions) {
  using namespace parallel_for_details;
  CHECK_GT(num_threads, 0);
  if (start >= end) {
    return;
  }
  CHECK_EQ(partitions.front(), start);
  CHECK_EQ(partitions.back(), end);
  if (num_threads == 1 || end - start <= num_threads) {
    ParallelFor(context, start, end, num_threads, function);
    return;
  }
  CHECK_GT(partitions.size(), 1);
  const int num_partitions = partitions.size() - 1;
  ParallelFor(context,
              0,
              num_partitions,
              num_threads,
              [&function, &partitions](int thread_id, int partition_id) {
                const int partition_start = partitions[partition_id];
                const int partition_end = partitions[partition_id + 1];
                const auto range =
                    std::make_tuple(partition_start, partition_end);

                InvokeOnSegment<F>(thread_id, range, function);
              });
}

// Execute function for every element in the range [start, end) with at most
// num_threads, taking into account user-provided integer cumulative costs of
// iterations. Cumulative costs of iteration for indices in range [0, end) are
// stored in objects from cumulative_cost_data. User-provided
// cumulative_cost_fun returns non-decreasing integer values corresponding to
// inclusive cumulative cost of loop iterations, provided with a reference to
// user-defined object. Only indices from [start, end) will be referenced. This
// routine assumes that cumulative_cost_fun is non-decreasing (in other words,
// all costs are non-negative);
template <typename F, typename CumulativeCostData, typename CumulativeCostFun>
void ParallelFor(ContextImpl* context,
                 int start,
                 int end,
                 int num_threads,
                 const F& function,
                 const CumulativeCostData* cumulative_cost_data,
                 const CumulativeCostFun& cumulative_cost_fun) {
  using namespace parallel_for_details;
  CHECK_GT(num_threads, 0);
  if (start >= end) {
    return;
  }
  if (num_threads == 1 || end - start <= num_threads) {
    ParallelFor(context, start, end, num_threads, function);
    return;
  }
  // Creating several partitions allows us to tolerate imperfections of
  // partitioning and user-supplied iteration costs up to a certain extent
  const int kNumPartitionsPerThread = 4;
  const int kMaxPartitions = num_threads * kNumPartitionsPerThread;
  const std::vector<int> partitions = ComputePartition(
      start, end, kMaxPartitions, cumulative_cost_data, cumulative_cost_fun);
  CHECK_GT(partitions.size(), 1);
  const int num_partitions = partitions.size() - 1;
  ParallelFor(context, start, end, num_threads, function, partitions);
}

// Evaluate vector expression in parallel
// Assuming LhsExpression and RhsExpression are some sort of
// column-vector expression, assignment lhs = rhs
// is eavluated over a set of contiguous blocks in parallel.
// This is expected to work well in the case of vector-based
// expressions (since they typically do not result into
// temporaries).
// This method expects lhs to be size-compatible with rhs
template <typename LhsExpression, typename RhsExpression>
void ParallelAssign(ContextImpl* context,
                    int num_threads,
                    LhsExpression& lhs,
                    const RhsExpression& rhs) {
  static_assert(LhsExpression::ColsAtCompileTime == 1);
  static_assert(RhsExpression::ColsAtCompileTime == 1);
  CHECK_EQ(lhs.rows(), rhs.rows());
  const int num_rows = lhs.rows();
  ParallelFor(context,
              0,
              num_rows,
              num_threads,
              [&lhs, &rhs](const std::tuple<int, int>& range) {
                auto [start, end] = range;
                lhs.segment(start, end - start) =
                    rhs.segment(start, end - start);
              });
}

// Set vector to zero using num_threads
template <typename VectorType>
void ParallelSetZero(ContextImpl* context,
                     int num_threads,
                     VectorType& vector) {
  ParallelSetZero(context, num_threads, vector.data(), vector.rows());
}
void ParallelSetZero(ContextImpl* context,
                     int num_threads,
                     double* values,
                     int num_values);

// This class creates a thread safe barrier which will block until a
// pre-specified number of threads call Finished.  This allows us to block the
// main thread until all the parallel threads are finished processing all the
// work.
class BlockUntilFinished {
 public:
  explicit BlockUntilFinished(int num_total_jobs);

  // Increment the number of jobs that have been processed by the number of
  // jobs processed by caller and signal the blocking thread if all jobs
  // have finished.
  void Finished(int num_jobs_finished);

  // Block until receiving confirmation of all jobs being finished.
  void Block();

 private:
  // Mutex guards access to condition variable, while changes to number of
  // finished jobs are handled via atomic non-zero increments; this allows us to
  // only capture lock once
  std::mutex mutex_;
  std::condition_variable condition_;
  std::atomic<int> num_total_jobs_finished_;
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
  ThreadPoolState(int start, int end, int num_work_blocks, int num_workers);

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
  //  - base_block_size for the rest of the blocks
  //  blocks of indices are contiguous and disjoint
  std::atomic<int> block_id;

  // Provides a unique thread ID among all active threads
  // We do not schedule more than num_threads threads via thread pool
  // and caller thread might steal one ID
  std::atomic<int> thread_id;

  // Used to signal when all the work has been completed.  Thread safe.
  BlockUntilFinished block_until_finished;
};

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
void ParallelInvoke(ContextImpl* context,
                    int start,
                    int end,
                    int num_threads,
                    const F& function) {
  using namespace parallel_for_details;
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
  std::shared_ptr<ThreadPoolState> shared_state(
      new ThreadPoolState(start, end, num_work_blocks, num_threads));

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
      InvokeOnSegment<F>(thread_id, range, function);
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

#endif  // CERES_INTERNAL_PARALLEL_FOR_H_
