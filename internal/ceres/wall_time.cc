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

#include "ceres/wall_time.h"

#ifdef CERES_USE_OPENMP
#include <omp.h>
#else
#include <ctime>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace ceres {
namespace internal {

double WallTimeInSeconds() {
#ifdef CERES_USE_OPENMP
  return omp_get_wtime();
#else
#ifdef _WIN32
  return static_cast<double>(std::time(NULL));
#else
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
#endif
#endif
}

EventTimer::EventTimer(map<string, double>* aggregate_times,
                       bool print_event_log_at_destruction)
    : start_time_(WallTimeInSeconds()),
      last_event_time_(start_time_),
      print_event_log_at_destruction_(print_event_log_at_destruction),
      events_("\n"),
      times_(aggregate_times) {}

EventTimer::~EventTimer() {
  if (print_event_log_at_destruction_) {
    VLOG(2) << "\n" << events_ << "\n";
  }
}

void EventTimer::AddRelativeEvent(const string& event_name) {
  const double current_time = WallTimeInSeconds();
  const double time_delta = current_time - last_event_time_;
  last_event_time_ = current_time;
  Update(event_name, time_delta);
}

void EventTimer::AddAbsoluteEvent(const string& event_name) {
  last_event_time_ = WallTimeInSeconds();
  const double time_delta =  last_event_time_ - start_time_;
  Update(event_name, time_delta);
}

void EventTimer::Update(const string& event_name, const double time_delta) {
  (*times_)[event_name] += time_delta;
  if (print_event_log_at_destruction_) {
    StringAppendF(&events_, "%30s : %5.2e\n", event_name.c_str(), time_delta);
  }
}

}  // namespace internal
}  // namespace ceres
