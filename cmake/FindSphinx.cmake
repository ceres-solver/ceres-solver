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
# Author: pablo.speciale@gmail.com (Pablo Speciale)
#

# Find the Sphinx documentation generator
#
# This modules defines
#  SPHINX_EXECUTABLE
#  SPHINX_FOUND

find_program(SPHINX_EXECUTABLE
             NAMES sphinx-build
             PATHS
               /usr/bin
               /usr/local/bin
               /opt/local/bin
             DOC "Sphinx documentation generator")

if (SPHINX_EXECUTABLE)

  find_package(PythonInterp 3)

  if(PYTHONINTERP_FOUND)
    # Check for sphinx theme dependency for documentation
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sphinx_rtd_theme"
      RESULT_VARIABLE SPHINX_RTD_THEME
      OUTPUT_QUIET
      ERROR_QUIET
    )
  endif ()

  if (${SPHINX_RTD_THEME} EQUAL 0)
    set(SPHINX_RTD_THEME TRUE)
  else ()
    set(SPHINX_RTD_THEME FALSE)
  endif ()

endif ()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE SPHINX_RTD_THEME)

mark_as_advanced(SPHINX_EXECUTABLE)
