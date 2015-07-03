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
# Authors: keir@google.com (Keir Mierle)
#          alexs.mac@gmail.com (Alex Stewart)

# Set up the git hook to make Gerrit Change-Id: lines in commit messages.
FUNCTION(ADD_GERRIT_COMMIT_HOOK)
  UNSET (LOCAL_GIT_DIRECTORY)
  IF (EXISTS ${CMAKE_SOURCE_DIR}/.git)
    IF (IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.git)
      # .git directory can be found on Unix based system, or on Windows with
      # Git Bash (shipped with msysgit).
      SET (LOCAL_GIT_DIRECTORY ${CMAKE_SOURCE_DIR}/.git)
    ELSE(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/.git)
      # .git is a file, this means Ceres is a git submodule of another project
      # and our .git file contains the path to the git directory which manages
      # Ceres, so we should add the gerrit hook there.
      FILE(READ ${CMAKE_SOURCE_DIR}/.git GIT_SUBMODULE_FILE_CONTENTS)
      # Strip any trailing newline characters, s/t we get a valid path.
      STRING(REGEX REPLACE "gitdir:[ ]*([^$].*)[\n].*" "${CMAKE_SOURCE_DIR}/\\1"
        GIT_SUBMODULE_GIT_DIRECTORY_PATH "${GIT_SUBMODULE_FILE_CONTENTS}")
      GET_FILENAME_COMPONENT(GIT_SUBMODULE_GIT_DIRECTORY_PATH
        "${GIT_SUBMODULE_GIT_DIRECTORY_PATH}" ABSOLUTE)
      IF (EXISTS ${GIT_SUBMODULE_GIT_DIRECTORY_PATH}
          AND IS_DIRECTORY ${GIT_SUBMODULE_GIT_DIRECTORY_PATH})
        SET(LOCAL_GIT_DIRECTORY "${GIT_SUBMODULE_GIT_DIRECTORY_PATH}")
      ENDIF()
    ENDIF()
  ELSE (EXISTS ${CMAKE_SOURCE_DIR}/.git)
    # TODO(keir) Add proper Windows support.
  ENDIF (EXISTS ${CMAKE_SOURCE_DIR}/.git)

  IF (EXISTS ${LOCAL_GIT_DIRECTORY})
    IF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
      MESSAGE(STATUS "Detected Ceres being used as a git submodule, adding "
        "commit hook for Gerrit to: ${LOCAL_GIT_DIRECTORY}")
      # Download the hook only if it is not already present.
      FILE(DOWNLOAD https://ceres-solver-review.googlesource.com/tools/hooks/commit-msg
        ${CMAKE_BINARY_DIR}/commit-msg)

      # Make the downloaded file executable, since it is not by default.
      FILE(COPY ${CMAKE_BINARY_DIR}/commit-msg
        DESTINATION ${LOCAL_GIT_DIRECTORY}/hooks/
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)
    ENDIF (NOT EXISTS ${LOCAL_GIT_DIRECTORY}/hooks/commit-msg)
  ENDIF (EXISTS ${LOCAL_GIT_DIRECTORY})
ENDFUNCTION()
