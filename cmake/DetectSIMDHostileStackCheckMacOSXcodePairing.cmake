# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2019 Google Inc. All rights reserved.
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

# As detailed in [1] the combination of macOS 10.15.x (Catalina) and Xcode 11
# (up to at least 10.15.1 / Xcode 11.1) enables by default -fstack-check which
# can break the alignment requirements for SIMD instructions resulting in
# segfaults from within Eigen.
#
# [1]: https://forums.developer.apple.com/thread/121887
function(detect_simd_hostile_stack_check_macos_xcode_pairing OUT_VAR)
  set(${OUT_VAR} FALSE PARENT_SCOPE)
  if (NOT APPLE)
    return()
  endif()

  execute_process(COMMAND sw_vers -productVersion
    OUTPUT_VARIABLE MACOS_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(COMMAND xcodebuild -version
    OUTPUT_VARIABLE XCODE_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH "Xcode [0-9\\.]+" XCODE_VERSION "${XCODE_VERSION}")
  string(REGEX REPLACE "Xcode ([0-9\\.]+)" "\\1" XCODE_VERSION "${XCODE_VERSION}")

  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_FLAGS "-mavx -O3")
  # Minimal test case taken from [1], author: snowcat, segfaults if
  # -fstack-check is enabled by default on affected macOS / Xcode pairing.
  check_cxx_source_runs(
    "int main(void) {
       register char a __asm(\"rbx\") = 0;
       char b[5000];
       char c[100] = {0};
       asm volatile(\"\" : : \"r,m\"(a), \"r,m\"(b), \"r,m\"(c) : \"memory\");
       return 0;
     }"
     BROKEN_STACK_CHECK_DISABLED_BY_DEFAULT)

   if (NOT BROKEN_STACK_CHECK_DISABLED_BY_DEFAULT)
     message("-- Detected macOS version: ${MACOS_VERSION} and Xcode version: "
       "${XCODE_VERSION} with SIMD-hostile -fstack-check enabled by default. "
       "Unless -fno-stack-check is used, segfaults may occur in any code that "
       "uses SIMD instructions.")
     set(${OUT_VAR} TRUE PARENT_SCOPE)
   endif()
endfunction()
