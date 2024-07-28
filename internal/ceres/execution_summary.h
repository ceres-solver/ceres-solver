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
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_EXECUTION_SUMMARY_H_
#define CERES_INTERNAL_EXECUTION_SUMMARY_H_

#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ceres/internal/export.h"

namespace ceres::internal {

struct CallStatistics {
  CallStatistics() = default;
  absl::Duration time = absl::ZeroDuration();
  int calls{0};
};

// Struct used by various objects to report statistics about their
// execution.
class ExecutionSummary {
 public:
  void IncrementTimeBy(const std::string& name, absl::Duration delta) {
    std::lock_guard<std::mutex> l(mutex_);
    CallStatistics& call_stats = statistics_[name];
    call_stats.time += delta;
    ++call_stats.calls;
  }

  const std::map<std::string, CallStatistics>& statistics() const {
    return statistics_;
  }

 private:
  std::mutex mutex_;
  std::map<std::string, CallStatistics> statistics_;
};

class ScopedExecutionTimer {
 public:
  ScopedExecutionTimer(std::string name, ExecutionSummary* summary)
      : start_time_(absl::Now()), name_(std::move(name)), summary_(summary) {}

  ~ScopedExecutionTimer() {
    summary_->IncrementTimeBy(name_, absl::Now() - start_time_);
  }

 private:
  absl::Time start_time_;
  const std::string name_;
  ExecutionSummary* summary_;
};

}  // namespace ceres::internal

#endif  // CERES_INTERNAL_EXECUTION_SUMMARY_H_
