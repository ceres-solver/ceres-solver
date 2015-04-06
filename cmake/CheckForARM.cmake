# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2015 Google Inc. All rights reserved.
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
# Author: alexs.mac@gmail.com (Alex Stewart)

# Sets ${TARGET_CPU_IS_ARM_VAR} to TRUE iff the target CPU is an ARM, false
# otherwise.
#
# Note that we use CMAKE_SYSTEM_PROCESSOR, not CMAKE_HOST_SYSTEM_PROCESSOR, as
# we care about the target CPU, not the host CPU when cross-compiling.
FUNCTION(CHECK_FOR_ARM TARGET_CPU_IS_ARM_VAR)
  STRING(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" LOWER_CASE_SYSTEM_PROCESSOR)
  IF ("${LOWER_CASE_SYSTEM_PROCESSOR}" MATCHES ".*arm.*")
    SET(${TARGET_CPU_IS_ARM_VAR} TRUE PARENT_SCOPE)
  ELSE()
    SET(${TARGET_CPU_IS_ARM_VAR} FALSE PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()
