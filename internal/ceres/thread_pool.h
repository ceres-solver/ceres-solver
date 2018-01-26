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

#ifndef CERES_INTERNAL_THREAD_POOL_H_
#define CERES_INTERNAL_THREAD_POOL_H_

#include <mutex>
#include <thread>
#include <vector>

#include "ceres/concurrent_queue.h"

namespace ceres {
namespace internal {

// A thread pool with a fixed number of workers and an unbounded task
// queue.
class ThreadPool {
 public:
  // Default constructor with no active threads.
  ThreadPool();

  // Instantiates a thread pool with the minimum of (number of hardware threads,
  // num_threads) number of threads.
  explicit ThreadPool(int num_threads);

  ~ThreadPool();

  // Resize the thread pool if it is currently less than the requested number of
  // threads. The thread pool will be resized to the minimum of (number of
  // hardware threads, num_threads) number of threads.
  void Resize(int num_threads);

  // Add a task to the queue and wake up a blocked thread.
  void AddTask(const std::function<void()>& func);

  // Returns the current size of the thread pool.
  int Size();

 private:
  // Main loop for the threads which blocks on the task queue until work becomes
  // available.
  void ThreadMainLoop();

  // Signal all the threads to stop.
  void Stop();

  ConcurrentQueue<std::function<void()>> task_queue_;
  std::vector<std::thread> thread_pool_;
  std::mutex thread_pool_mutex_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_THREAD_POOL_H_
