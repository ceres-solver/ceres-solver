// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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
// Author: strandmark@google.com (Petter Strandmark)

#ifndef CERES_INTERNAL_WALL_TIME_H_
#define CERES_INTERNAL_WALL_TIME_H_

#include <map>
#include "glog/logging.h"
#include "ceres/internal/port.h"
#include "ceres/stringprintf.h"

namespace ceres {
namespace internal {

// Returns time, in seconds, from some arbitrary starting point. Has very
// high precision if OpenMP is available, otherwise only second granularity.
double WallTimeInSeconds();

class EventTimer {
 public:
  explicit EventTimer(map<string, double>* aggregate_times,
                      bool print_event_log_at_destruction = true)
      : start_time_(WallTimeInSeconds()),
        last_event_time_(start_time_),
        print_event_log_at_destruction_(print_event_log_at_destruction),
        events_("\n"),
        times_(aggregate_times) {}

  ~EventTimer() {
    if (print_event_log_at_destruction_) {
      VLOG(2) << events_;
    }
  }

  void AddRelativeEvent(const string& event_name) {
    const double current_time = WallTimeInSeconds();
    const double time_delta = current_time - last_event_time_;
    last_event_time_ = current_time;
    Update(event_name, time_delta);
  }

  void AddAbsoluteEvent(const string& event_name) {
    last_event_time_ = WallTimeInSeconds();
    const double time_delta =  last_event_time_ - start_time_;
    Update(event_name, time_delta);
  }

 private:
  void Update(const string& event_name, const double time_delta) {
    (*times_)[event_name] += time_delta;
    if (print_event_log_at_destruction_) {
      StringAppendF(&events_, "%s : %e\n", event_name.c_str(), time_delta);
    }
  }

  const double start_time_;
  double last_event_time_;
  const bool print_event_log_at_destruction_;
  string events_;
  map<string, double>* times_;
};

}  // namespace internal
}  // namespace ceres

#endif  // CERES_INTERNAL_WALL_TIME_H_
