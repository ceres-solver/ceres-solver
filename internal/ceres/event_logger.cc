// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2024 Google Inc. All rights reserved.
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
// Author: strandmark@google.com (Petter Strandmark)

#include "ceres/event_logger.h"

#include <string>
#include <string_view>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace ceres::internal {

EventLogger::EventLogger(std::string_view logger_name)
    : start_time_(absl::Now()) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  last_event_time_ = start_time_;
  events_ = absl::StrFormat(
      "\n%s\n                                        Delta   Cumulative\n",
      logger_name);
}

EventLogger::~EventLogger() {
  if (!VLOG_IS_ON(3)) {
    return;
  }
  AddEvent("Total");
  VLOG(3) << "\n" << events_ << "\n";
}

void EventLogger::AddEvent(std::string_view event_name) {
  if (!VLOG_IS_ON(3)) {
    return;
  }

  const absl::Time now = absl::Now();
  absl::StrAppendFormat(&events_,
                        "  %30s : %10.5f   %10.5f\n",
                        event_name,
                        absl::ToDoubleSeconds(now - last_event_time_),
                        absl::ToDoubleSeconds(now - start_time_));
  last_event_time_ = now;
}

}  // namespace ceres::internal
