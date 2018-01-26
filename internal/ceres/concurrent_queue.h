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

#ifndef CERES_INTERNAL_CONCURRENT_QUEUE_H_
#define CERES_INTERNAL_CONCURRENT_QUEUE_H_

// This include must come before any #ifndef check on Ceres compile options.
#include "ceres/internal/port.h"

#ifdef CERES_USE_CXX11_THREADS

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "glog/logging.h"

namespace ceres {
namespace internal {

// A thread-safe multi-producer, multi-consumer queue for queueing items that
// are typically handled asynchronously by multiple threads. The ConcurrentQueue
// has two states:
//
//  (1) Abort has not been called.  Pop will block until an item is available to
//  pop from the queue.  Push will add an item to the queue.
//
//  (2) Abort has been called.  Pop will not wait and immediately return false
//  without popping any value off the queue. Push will be a no-op and not add
//  any new values to the queue.
//
// A common use case is using the concurrent queue as an interface for
// scheduling tasks for a set of thread workers:
//
// ConcurrentQueue<Task> task_queue;
//
// [Worker threads]:
//   Task task;
//   while(task_queue.Pop(&task)) {
//     ...
//   }
//
// [Producers]:
//   task_queue.Push(...);
//   ..
//   task_queue.Push(...);
//   ...
//   // Signal worker threads to stop blocking on Pop and terminate.
//   task_queue.Abort();
template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() : abort_(false) {}

  // Unblock all threads waiting to pop a value from the queue. All future Pop
  // requests will also return immediately with no value.
  void Abort() {
    std::unique_lock<std::mutex> lock(mutex_);
    abort_ = true;
    lock.unlock();
    work_pending_condition_.notify_all();
  }

  // Pop an element from the queue. Blocks until one is available or Abort is
  // called.  Returns true if an element was successfully popped from the queue,
  // otherwise returns false.  If Abort has been called, this will immediately
  // return false without popping an element off the queue (even if there are
  // items in the queue).
  bool Pop(T* value) {
    CHECK(value != nullptr);

    std::unique_lock<std::mutex> lock(mutex_);
    while (!abort_ && queue_.empty()) {
      work_pending_condition_.wait(lock);
    }
    if (abort_) {
      return false;
    }

    *value = queue_.front();
    queue_.pop();

    return true;
  }

  // Push an element onto the queue and wake up a blocked thread.  If Abort has
  // been called, the value will not be pushed onto the queue.
  void Push(const T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (abort_) {
      return;
    }
    queue_.push(value);
    work_pending_condition_.notify_one();
  }

 private:
  // The mutex controls read and write access to the queue_ and stop_
  // variables. It is also used to block the calling thread until an element is
  // available to pop from the queue.
  std::mutex mutex_;
  std::condition_variable work_pending_condition_;

  std::queue<T> queue_;
  // If true, signals that callers should not block waiting to pop an element
  // off the queue.
  bool abort_;
};


}  // namespace internal
}  // namespace ceres

#endif // CERES_USE_CXX11_THREADS

#endif  // CERES_INTERNAL_CONCURRENT_QUEUE_H_
