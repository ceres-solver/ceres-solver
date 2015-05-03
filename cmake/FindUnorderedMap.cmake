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

# FindUnorderedMap.cmake - Find unordered_map header and namespace.
#
# This module defines the following variables:
#
# UNORDERED_MAP_FOUND: TRUE if unordered_map is found.
# HAVE_UNORDERED_MAP_IN_STD_NAMESPACE: Use <unordered_map> & std.
# HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE: Use <unordered_map> & std::tr1.
# HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE: <tr1/unordered_map> & std::tr1.
MACRO(FIND_UNORDERED_MAP)
  # To support CXX11 option, clear the results of all check_xxx() functions
  # s/t we always perform the checks each time, otherwise CMake fails to
  # detect that the tests should be performed again after CXX11 is toggled.
  UNSET(HAVE_STD_UNORDERED_MAP_HEADER CACHE)
  UNSET(HAVE_UNORDERED_MAP_IN_STD_NAMESPACE CACHE)
  UNSET(HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE CACHE)
  UNSET(HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE CACHE)

  SET(UNORDERED_MAP_FOUND FALSE)
  INCLUDE(CheckIncludeFileCXX)
  CHECK_INCLUDE_FILE_CXX(unordered_map HAVE_STD_UNORDERED_MAP_HEADER)
  IF (HAVE_STD_UNORDERED_MAP_HEADER)
    # Finding the unordered_map header doesn't mean that unordered_map
    # is in std namespace.
    #
    # In particular, MSVC 2008 has unordered_map declared in std::tr1.
    # In order to support this, we do an extra check to see which
    # namespace should be used.
    INCLUDE(CheckCXXSourceCompiles)
    CHECK_CXX_SOURCE_COMPILES("#include <unordered_map>
                               int main() {
                                 std::unordered_map<int, int> map;
                                 return 0;
                               }"
                               HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
    IF (HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
      SET(UNORDERED_MAP_FOUND TRUE)
      MESSAGE("-- Found unordered_map/set in std namespace.")
    ELSE (HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
      CHECK_CXX_SOURCE_COMPILES("#include <unordered_map>
                                 int main() {
                                   std::tr1::unordered_map<int, int> map;
                                   return 0;
                                 }"
                                 HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
      IF (HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
        SET(UNORDERED_MAP_FOUND TRUE)
        MESSAGE("-- Found unordered_map/set in std::tr1 namespace.")
      ELSE (HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
        MESSAGE("-- Found <unordered_map> but cannot find either "
          "std::unordered_map or std::tr1::unordered_map.")
      ENDIF (HAVE_UNORDERED_MAP_IN_TR1_NAMESPACE)
    ENDIF (HAVE_UNORDERED_MAP_IN_STD_NAMESPACE)
  ELSE (HAVE_STD_UNORDERED_MAP_HEADER)
    CHECK_INCLUDE_FILE_CXX("tr1/unordered_map"
      HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
    IF (HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
      SET(UNORDERED_MAP_FOUND TRUE)
      MESSAGE("-- Found tr1/unordered_map/set in std::tr1 namespace.")
    ELSE (HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
      MESSAGE("-- Unable to find <unordered_map> or <tr1/unordered_map>.")
    ENDIF (HAVE_TR1_UNORDERED_MAP_IN_TR1_NAMESPACE)
  ENDIF (HAVE_STD_UNORDERED_MAP_HEADER)
ENDMACRO(FIND_UNORDERED_MAP)
