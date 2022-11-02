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
#include <functional>
#include <mutex>

#include "ceres/context_impl.h"
#include "ceres/internal/disable_warnings.h"
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
std::tuple<Args...> args_of(void (F::*)(Args...) const);

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
  static void Invoke(int thread_id, int i, const F& function) {
    (void)thread_id;
    function(i);
  }
};

// For parallel for iterations of type [](int thread_id, int i) -> void
template <typename F>
struct InvokeImpl<F, std::tuple<int, int>> {
  static void Invoke(int thread_id, int i, const F& function) {
    function(thread_id, i);
  }
};

// Invoke function passing thread_id only if required
template <typename F>
void Invoke(int thread_id, int i, const F& function) {
  InvokeImpl<F, args_of_t<F>>::Invoke(thread_id, i, function);
}

// Try to split range into contiguous partitions within the threshold for
// maximal cost
template <typename T, typename G>
bool TryPartition(int start,
                  int end,
                  int max_num_partitions,
                  int max_partition_cost,
                  int cumulative_cost_offset,
                  const T* iteration_data,
                  const G& cumulative_cost_getter,
                  std::vector<int>* partition) {
  partition->clear();
  partition->push_back(start);
  int partition_start = start;
  int cost_offset = cumulative_cost_offset;

  const T* const range_end = iteration_data + end;
  while (partition_start < end) {
    // Already have max_num_partitions
    if (partition->size() > max_num_partitions) return false;
    const int target = max_partition_cost + cost_offset;
    const int partition_end =
        std::partition_point(iteration_data + partition_start,
                             iteration_data + end,
                             [&cumulative_cost_getter, target](const T& item) {
                               return cumulative_cost_getter(item) <= target;
                             }) -
        iteration_data;
    // Unable to make a partition from a single element
    if (partition_end == partition_start) return false;

    const int cost_last =
        cumulative_cost_getter(iteration_data[partition_end - 1]);
    const int partition_cost = cost_last - cost_offset;
    partition->push_back(partition_end);
    partition_start = partition_end;
    cost_offset = cost_last;
  }
  return true;
}

// Split workload into contiguous partitions with minimal maximal size
template <typename T, typename G>
std::vector<int> ComputePartition(int start,
                                  int end,
                                  int max_num_partitions,
                                  const T* iteration_data,
                                  const G& cumulative_cost_getter) {
  // If we already have admissible value of maximal cost, we can compute
  // partitions by launching binary search over cumulative cost
  // max_num_partitions times
  //
  // Thus we perform a binary search over maximal cost
  const int cumulative_cost_last =
      cumulative_cost_getter(iteration_data[end - 1]);
  const int cumulative_cost_offset =
      start ? cumulative_cost_getter(iteration_data[start - 1]) : 0;
  const int total_cost = cumulative_cost_last - cumulative_cost_offset;

  // Minimal maximal partition cost is not smaller than the average
  int partition_cost_left = total_cost / max_num_partitions - 1;
  // Minimal maximal partition cost is not largerg than the total cost
  int partition_cost_right = total_cost;

  std::vector<int> partition;
  partition.reserve(max_num_partitions + 1);
  while (partition_cost_right - partition_cost_left > 1) {
    const int midpoint =
        partition_cost_left + (partition_cost_right - partition_cost_left) / 2;
    bool admissible = TryPartition(start,
                                   end,
                                   max_num_partitions,
                                   midpoint,
                                   cumulative_cost_offset,
                                   iteration_data,
                                   cumulative_cost_getter,
                                   &partition);
    if (admissible) {
      partition_cost_right = midpoint;
    } else {
      partition_cost_left = midpoint;
    }
  }

  bool admissible = TryPartition(start,
                                 end,
                                 max_num_partitions,
                                 partition_cost_right,
                                 cumulative_cost_offset,
                                 iteration_data,
                                 cumulative_cost_getter,
                                 &partition);
  CHECK(admissible);
  return partition;
}
}  // namespace parallel_for_details

// Forward declaration of parallel invocation function that is to be
// implemented by each threading backend
template <typename F>
void ParallelInvoke(ContextImpl* context,
                    int i,
                    int num_threads,
                    const F& function);

// Execute the function for every element in the range [start, end) with at most
// num_threads. It will execute all the work on the calling thread if
// num_threads or (end - start) is equal to 1.
//
// Functions with two arguments will be passed thread_id and loop index on each
// invocation, functions with one argument will be invoked with loop index
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
    for (int i = start; i < end; ++i) {
      Invoke<F>(0, i, function);
    }
    return;
  }

  CHECK(context != nullptr);
  ParallelInvoke<F>(context, start, end, num_threads, function);
}

// Execute function for every element in the range [start, end) with at most
// num_threads, taking into account user-provided integer cumulative costs of
// iterations
template <typename F, typename T, typename G>
void ParallelFor(ContextImpl* context,
                 int start,
                 int end,
                 int num_threads,
                 const F& function,
                 const T* iteration_data,
                 const G& cumulative_cost_getter) {
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
      start, end, kMaxPartitions, iteration_data, cumulative_cost_getter);
  CHECK_GT(partitions.size(), 1);
  const int num_partitions = partitions.size() - 1;
  ParallelFor(context,
              0,
              num_partitions,
              num_threads,
              [&function, &partitions](int thread_id, int partition_id) {
                const int partition_start = partitions[partition_id];
                const int partition_end = partitions[partition_id + 1];

                for (int i = partition_start; i < partition_end; ++i) {
                  Invoke<F>(thread_id, i, function);
                }
              });
}

}  // namespace ceres::internal

// Backend-specific implementations of ParallelInvoke
#include "ceres/parallel_for_cxx.h"
#include "ceres/parallel_for_openmp.h"
#ifdef CERES_NO_THREADS
namespace ceres::internal {
template <typename F>
void ParallelInvoke(ContextImpl* context,
                    int start,
                    int end,
                    int num_threads,
                    const F& function) {
  ParallelFor(context, start, end, 1, function);
}
}  // namespace ceres::internal
#endif

#include "ceres/internal/disable_warnings.h"

#endif  // CERES_INTERNAL_PARALLEL_FOR_H_
