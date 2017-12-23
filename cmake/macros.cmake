# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2017 Google Inc. All rights reserved.
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
# Author: sergey.vfx@gmail.com (Sergey Sharybin)

function(ADD_CHECK_CXX_COMPILER_FLAG CXXFLAGS CACHE_VAR FLAG)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("${FLAG}" "${CACHE_VAR}")
  if (${CACHE_VAR})
    set(${CXXFLAGS} "${${CXXFLAGS}} ${FLAG}" PARENT_SCOPE)
  endif (${CACHE_VAR})
endfunction(ADD_CHECK_CXX_COMPILER_FLAG)

function(APPEND_SOURCE_FILE_COMPILER_FLAGS SOURCEFILE FLAGS)
  # NOTE: Only works since CMake 2.8.6.
  set_property(SOURCE ${SOURCEFILE} APPEND_STRING PROPERTY COMPILE_FLAGS " ${FLAGS} ")
endfunction(APPEND_SOURCE_FILE_COMPILER_FLAGS)
