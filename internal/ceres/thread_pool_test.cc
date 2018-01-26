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
#include "ceres/internal/port.h"

#ifdef CERES_USE_CXX11_THREADS

#include "ceres/thread_pool.h"

#include <atomic>
#include <thread>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ceres {
namespace internal {

// Adds a number of tasks to the thread pool and ensures they all run.
TEST(ThreadPool, AddTask) {
  int value = 0;
  const int num_tasks = 100;
  {
    ThreadPool thread_pool(2);

    std::condition_variable condition;
    std::mutex mutex;

    for (int i = 0; i < num_tasks; ++i) {
      thread_pool.AddTask([&]() {
          std::unique_lock<std::mutex> lock(mutex);
          ++value;
          condition.notify_all();
        });
    }

    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&](){return value == num_tasks;});
  }

  EXPECT_EQ(num_tasks, value);
}

TEST(ThreadPool, Resize) {
  // Ensure the hardware supports more than 1 thread to ensure the test will
  // pass.
  const int num_hardware_threads = std::thread::hardware_concurrency();
  if (num_hardware_threads <= 1) {
    LOG(ERROR)
        << "Test not supported, the hardware does not support threading.";
    return;
  }

  ThreadPool thread_pool(1);

  EXPECT_EQ(1, thread_pool.Size());

  thread_pool.Resize(2);

  EXPECT_EQ(2, thread_pool.Size());
}

}  // namespace internal
}  // namespace ceres

#endif // CERES_USE_CXX11_THREADS
