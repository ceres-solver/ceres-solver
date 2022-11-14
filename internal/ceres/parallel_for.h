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

  const CumulativeCostData* const range_end = cumulative_cost_data + end;
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
