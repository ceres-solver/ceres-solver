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

// A blocking thread safe queue.
template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() : stop_(false) {}

  // Unblock all threads waiting to pop a value from the queue. All future Pop
  // requests will also return immediately with no value.
  void Stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_ = true;
    lock.unlock();
    condition_.notify_all();
  }

  // Pop an element from the queue. Blocks until one is available or Stop is
  // called.
  bool Pop(T* value) {
    CHECK(value != nullptr);

    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_ && queue_.empty()) {
      condition_.wait(lock);
    }
    if (stop_) {
      return false;
    }

    *value = queue_.front();
    queue_.pop();

    return true;
  }

  // Push an element onto the queue and wake up a blocked thread.
  void Push(const T& value) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(value);
    lock.unlock();
    condition_.notify_one();
  }

 private:
  // The mutex controls read and write access to the queue_ and stop_
  // variables. It is also used to block the calling thread until an element is
  // available to pop from the queue.
  std::mutex mutex_;
  std::condition_variable condition_;

  std::queue<T> queue_;
  // If true, signals that callers should not block waiting to pop an element
  // off the queue.
  bool stop_;
};


}  // namespace internal
}  // namespace ceres

#endif // CERES_USE_CXX11_THREADS

#endif  // CERES_INTERNAL_CONCURRENT_QUEUE_H_
