# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
# http://code.google.com/p/ceres-solver/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Google Inc. nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: keir@google.com (Keir Mierle)

ADD_EXECUTABLE(quadratic quadratic.cc)
TARGET_LINK_LIBRARIES(quadratic ceres)

ADD_EXECUTABLE(quadratic_auto_diff quadratic_auto_diff.cc)
TARGET_LINK_LIBRARIES(quadratic_auto_diff ceres)

ADD_EXECUTABLE(quadratic_numeric_diff quadratic_numeric_diff.cc)
TARGET_LINK_LIBRARIES(quadratic_numeric_diff ceres)

ADD_EXECUTABLE(powell powell.cc)
TARGET_LINK_LIBRARIES(powell ceres)

IF (GFLAGS)
  ADD_EXECUTABLE(circle_fit circle_fit.cc)
  TARGET_LINK_LIBRARIES(circle_fit ceres)

  ADD_EXECUTABLE(bundle_adjuster
                 bundle_adjuster.cc
                 bal_problem.cc)
  TARGET_LINK_LIBRARIES(bundle_adjuster ceres)
ENDIF (GFLAGS)

ADD_EXECUTABLE(simple_bundle_adjuster
               simple_bundle_adjuster.cc)
TARGET_LINK_LIBRARIES(simple_bundle_adjuster ceres)
