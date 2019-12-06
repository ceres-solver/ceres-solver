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
# Author: darius.rueckert@fau.de (Darius Rueckert)
#
# Testing the AutoDiffCodegen system is more complicated, because function- and
# constructor calls have side-effects. In C++ the evaluation order and
# the elision of copies is implementation defined. Between different compilers,
# some expression might be evaluated in a different order or some copies might be
# removed.
#
# Therefore, we run tests that expect a particular compiler behaviour only on gcc.
#
# The semantic tests, which check the correctness by executing the generated code
# are still run on all platforms.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CXX_FLAGS_OLD ${CMAKE_CXX_FLAGS})
    # From the man page:
    #   The C++ standard allows an implementation to omit creating a
    #   temporary which is only used to initialize another object of the
    #   same type.  Specifying -fno-elide-constructors disables that
    #   optimization, and forces G++ to call the copy constructor in all cases.
    # We use this flag to get the same results between different versions of
    # gcc and different optimization levels. Without this flag, testing would
    # be very painfull and might break when a new compiler version is released.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-elide-constructors")
    ceres_test(expression)
    ceres_test(expression_graph)
    ceres_test(expression_ref)
    ceres_test(code_generator)
    set(CMAKE_CXX_FLAGS ${CXX_FLAGS_OLD})
endif()
