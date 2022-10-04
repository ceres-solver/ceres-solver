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

// This include must come before any #ifndef check on Ceres compile options.
#ifndef CERES_INTERNAL_PARALLEL_FOR_CXX_H_
#define CERES_INTERNAL_PARALLEL_FOR_CXX_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/config.h"

#ifdef CERES_USE_CXX_THREADS

#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unsupported/Eigen/CXX11/Tensor>

#include "glog/logging.h"

namespace ceres::internal {
// This implementation uses a fixed size max worker pool with a shared task
// queue. We will exit the ParallelFor call when all of the work has
// been done, not when all of the tasks have been popped off the task queue.
//
// The thread ID is guaranteed to be in [0, num_threads].
//
// A performance analysis has shown this implementation is on par with OpenMP
// and TBB.
template <typename F>
void ParallelInvoke(ContextImpl* context,
                    int start,
                    int end,
                    const F& function) {
  using namespace parallel_for_details;
  CHECK(context != nullptr);
  if (end <= start) {
    return;
  }

  // Fast path for when there is no thread pool or called from a child thread to
  // prevent a deadlock for nested parallel fors.
  if (context->eigen_thread_pool_ == nullptr ||
      context->eigen_thread_pool_->CurrentThreadId() != -1) {
     // Use the current thread ID to avoid conflicting with other threads.
     const int thread_id =
            context->eigen_thread_pool_->CurrentThreadId() + 1;
    for (int i = start; i < end; ++i) {
      Invoke<F>(thread_id, i, function);
    }
    return;
  }

  CHECK(context->eigen_thread_pool_ != nullptr);
  Eigen::ThreadPoolDevice device(context->eigen_thread_pool_.get(),
                                 context->eigen_thread_pool_->NumThreads());
  std::function<void(int64_t, int64_t)> task_function =
      [&start, &function, &context](int first, int last) {
        // We may use the current thread to do some work synchronously.
        // When calling CurrentThreadId() from outside of the thread
        // pool, we get -1, so we can shift every id up by 1.
        const int thread_id =
            context->eigen_thread_pool_->CurrentThreadId() + 1;
        for (int i = first; i < last; ++i) {
          Invoke<F>(thread_id, start + i, function);
        }
      };

  // Eigen doesn't have a way to specify a constant block size, rather
  // it chooses the block size based upon the cost. We choose a large
  // cost to force splitting and Eigen commonly chooses 4 * num_threads
  // blocks.
  device.parallelFor(
      end - start, Eigen::TensorOpCost(0, 0, 1 << 30), task_function);
}

}  // namespace ceres::internal
#endif
#endif
