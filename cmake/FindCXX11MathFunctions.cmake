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
#

# FindCXX11MathFunctions.cmake - Find C++11 math functions.
#
# This module defines the following variables:
#
# CXX11_MATH_FUNCTIONS_FOUND: TRUE if C++11 math functions are found.

macro(find_cxx11_math_functions)
  # To support CXX11 option, clear the results of all check_xxx() functions
  # s/t we always perform the checks each time, otherwise CMake fails to
  # detect that the tests should be performed again after CXX11 is toggled.
  unset(CXX11_MATH_FUNCTIONS_FOUND CACHE)

  # Verify that all C++11-specific math functions used by jet.h exist.
  check_cxx_source_compiles("#include <cmath>
                             #include <cstddef>
                             static constexpr size_t kMaxAlignBytes = alignof(std::max_align_t);
                             int main() {
                                std::cbrt(1.0);
                                std::exp2(1.0);
                                std::log2(1.0);
                                std::hypot(1.0, 1.0);
                                std::fmax(1.0, 1.0);
                                std::fmin(1.0, 1.0);
                                return 0;
                             }"
                             CXX11_MATH_FUNCTIONS_FOUND)
endmacro()
