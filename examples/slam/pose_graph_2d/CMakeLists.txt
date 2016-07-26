# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2016 Google Inc. All rights reserved.
# http://ceres-solver.org/
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
# Author: vitus@google.com (Michael Vitus)

add_library(pose_graph_2d_lib
  angle_local_parameterization.h
  normalize_angle.h
  pose_graph_2d_error_term.cc
)

add_executable(pose_graph_2d read_g2o.cc pose_graph_2d.cc)
target_link_libraries(pose_graph_2d
  ceres
  pose_graph_2d_lib
)

if (BUILD_TESTING)
  macro (CERES_TEST NAME)
    add_executable(${NAME}_test ${NAME}_test.cc)
    target_link_libraries(${NAME}_test test_util  pose_graph_2d_lib ceres gtest)
    if (BUILD_SHARED_LIBS)
      # Define gtest-specific shared library flags for linking.
      append_target_property(${NAME}_test COMPILE_DEFINITIONS
        GTEST_LINKED_AS_SHARED_LIBRARY)
    endif()
    add_test(NAME ${NAME}_test
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test
             --test_srcdir
             ${CMAKE_SOURCE_DIR}/data)
  endmacro (CERES_TEST)

  ceres_test(angle_local_parameterization)
  ceres_test(normalize_angle)
  ceres_test(pose_graph_2d_error_term)
endif(BUILD_TESTING)

