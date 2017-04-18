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
# Author: alexs.mac@gmail.com (Alex Stewart)

# Adds Ceres' C++11 requirements to a target such that they are exported when
# the target is exported (if the version of CMake supports it).
#
#    add_ceres_cxx11_requirements_to_target( [target1 [target2 [...]]] )
function(add_ceres_cxx11_requirements_to_target)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11" COMPILER_HAS_CXX11_FLAG)

  foreach(TARGET ${ARGN})
    if (NOT TARGET ${TARGET})
      message(FATAL_ERROR "Specified target to append Ceres C++11 requirements "
        "to: ${TARGET} is not a declared CMake target.")
    endif()

    if (COMPILER_HAS_CXX11_FLAG)
      # IMPORTANT: It is not sufficient to specify the
      #            CXX_STANDARD/CXX_STANDARD_REQUIRED target properties
      #            as these target properties are NOT exported.
      if (CMAKE_VERSION VERSION_LESS "2.8.12")
        # CMake version < 2.8.12 does not support target_compile_options(), warn
        # user that they will have to add compile flags to their own projects
        # manually.
        message(WARNING "-- Warning: Detected CMake version: ${CMAKE_VERSION} "
          "< 2.8.12, which is the minimum required for compile options to be "
          "included in an exported CMake target and the detected. compiler "
          "requires -std=c++11. The client is responsible for adding "
          "-std=c++11 when linking against: ${TARGET}.")
      elseif (COMMAND target_compile_features)
        # CMake >= 3.1, use new target_compile_features() to specify Ceres'
        # C++11 requirements as used in the public API.  This assumes that
        # C++11 STL features are available if the specified features are
        # available.  We do not use the cxx_std_11 feature to specify this as
        # this did not come in until CMake 3.8.
        #
        # The reason to prefer using target_compile_features() if it exists is
        # that this handles 'upgrading' of the C++ standard required more
        # gracefully, e.g. if a client of Ceres requires C++14, but Ceres was
        # compiled against C++11 then target_compile_options() may not work as
        # expected.
        target_compile_features(
          ${TARGET} PUBLIC cxx_alignas cxx_alignof cxx_constexpr)
      else()
        # CMake version >= 2.8.12 && < 3.1 supports target_compile_options()
        # but not target_compile_features(). For these intermediary versions,
        # we use target_compile_options() to manually specify the C++11 flag and
        # export it for client targets that depend on the target iff they are
        # NOT compiling for C.  We check for not C, rather than C++ as
        # LINKER_LANGUAGE is often NOTFOUND and then uses the default (C++).
        target_compile_options(${TARGET} PUBLIC
          $<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,C>>:-std=c++11>)
      endif()
    endif()
  endforeach()
endfunction()
