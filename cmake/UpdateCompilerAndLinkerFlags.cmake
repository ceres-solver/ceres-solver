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

# Remove all duplicate flags from ${FLAGS_VAR}, which is assumed to contain a
# whitespace separated list of compiler or linker flags.  The de-duplicated
# version is written back to ${FLAGS_VAR} in the caller's scope.
FUNCTION(REMOVE_DUPLICATE_FLAGS FLAGS_VAR)
  # Protect against the input flags variable containing semicolons, if this
  # is the case, then substitute them for something that should never appear
  # in flags, and then substitute back again at the end s/t our list operations
  # using semicolons do not affect the flags.
  SET(MAGIC_SEMICOLON_SUBSTITUTE "#$%@!")
  IF (${FLAGS_VAR} MATCHES ".*;.*")
    SET(INPUT_FLAGS_CONTAINED_SEMICOLONS TRUE)
    STRING(REPLACE ";" "${MAGIC_SEMICOLON_SUBSTITUTE}"
      ${FLAG_VAR} "${${FLAG_VAR}}")
  ENDIF()

  # Convert (assumed) whitespace separated list of flags into a CMake semicolon
  # separated list s/t we can use REMOVE_DUPLICATES.
  STRING(REPLACE " " ";" ${FLAG_VAR} "${${FLAG_VAR}}")
  LIST(REMOVE_DUPLICATES ${FLAG_VAR})
  # Convert the removed
  STRING(REPLACE ";" " " ${FLAG_VAR} "${${FLAG_VAR}}")

  # If the input contained semicolons initially, substitute them back in
  # place of the magic placeholder.
  IF (INPUT_FLAGS_CONTAINED_SEMICOLONS)
    STRING(REPLACE "${MAGIC_SEMICOLON_SUBSTITUTE}" ";"
      ${FLAG_VAR} "${${FLAG_VAR}}")
  ENDIF()

  SET(${FLAGS_VAR} "${${FLAGS_VAR}}" PARENT_SCOPE)
ENDFUNCTION()

# Update cached CMake C/CXX compiler & linker flags with their latest local
# values (if shadowed by a local variable) after removal of any duplicates.
#
# In CMake, any locally declared variable with the same name as a variable in
# the CMake cache shadows the cache variable.  This is problematic for flags
# variables such as CMAKE_CXX_FLAGS, as any changes to them via set() which
# are not reflected in the cache are used, but not shown in the CMake GUI.
# This function ensures that the cached flags variables are always up to date,
# and do not contain any duplicates, which otherwise occurs whenever CMake
# is configured multiple times and the flags variables are updated using:
# set(FLAGS_VAR "${FLAGS_VAR} -my-amazing-flag")
FUNCTION(UPDATE_COMPILER_AND_LINKER_FLAGS)
  # Get all variables declared in the CMake cache s/t we can find the
  # compiler & linker flag variables.
  GET_CMAKE_PROPERTY(ALL_CACHED_VARS CACHE_VARIABLES)

  # Find all CMAKE_C_FLAGS* & CMAKE_CXX_FLAGS* variables.
  STRING(REGEX MATCHALL "CMAKE_C[X]?[X]?_FLAGS[^;]*;?"
    ALL_CMAKE_C_CXX_FLAG_VARS "${ALL_CACHED_VARS}")
  FOREACH(FLAG_VAR ${ALL_CMAKE_C_CXX_FLAG_VARS})
    REMOVE_DUPLICATE_FLAGS(${FLAG_VAR})
    # Update the local version of the variable (which is the one used) to the
    # de-duplicated version, in addition to updating the cached version.
    SET(${FLAG_VAR} "${${FLAG_VAR}}" PARENT_SCOPE)
    UPDATE_CACHE_VARIABLE(${FLAG_VAR} "${${FLAG_VAR}}")
  ENDFOREACH()

  # Find all CMAKE_EXE_LINKER_FLAGS* variables.
  STRING(REGEX MATCHALL "CMAKE_EXE_LINKER_FLAGS[^;]*;?"
    ALL_CMAKE_EXE_LINKER_FLAG_VARS "${ALL_CACHED_VARS}")
  FOREACH(FLAG_VAR ${ALL_CMAKE_EXE_LINKER_FLAG_VARS})
    REMOVE_DUPLICATE_FLAGS(${FLAG_VAR})
    # Update the local version of the variable (which is the one used) to the
    # de-duplicated version, in addition to updating the cached version.
    SET(${FLAG_VAR} "${${FLAG_VAR}}" PARENT_SCOPE)
    UPDATE_CACHE_VARIABLE(${FLAG_VAR} "${${FLAG_VAR}}")
  ENDFOREACH()

ENDFUNCTION()
