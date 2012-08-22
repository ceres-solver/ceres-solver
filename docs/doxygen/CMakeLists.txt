# Ceres Solver - A fast non-linear least squares minimizer
# Copyright 2012 Google Inc. All rights reserved.
# http://code.google.com/p/ceres-solver/
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
# Author: arnaudgelas@gmail.com (Arnaud Gelas)

INCLUDE(${CMAKE_ROOT}/Modules/Documentation.cmake)

# -------------------------------------------------------------
#
# Generate Doxygen configuration file in all configuration file
#
IF (${DOXYGEN_DOT_FOUND})
  set(HAVE_DOT "YES")
ELSE ()
  SET(HAVE_DOT "NO")
ENDIF ()

OPTION(DOXYGEN_HTML "Doxygen will generate HTML Output" ON)
OPTION(DOXYGEN_DOCSET "Doxygen will generate additional index files for XCode 3" OFF)
OPTION(DOXYGEN_CHM "Doxygen will generate Microsoft HTML help" OFF)
OPTION(DOXYGEN_QHP "Doxygen will generate Qt Compressed Help" OFF)
OPTION(DOXYGEN_ECLIPSEHELP "Doxygen will generate Eclipse help plugin" OFF)
OPTION(DOXYGEN_LATEX "Doxygen will generate LaTeX output" OFF)
OPTION(DOXYGEN_RTF "Doxygen will generate RTF output optimized for Word 97" OFF)
OPTION(DOXYGEN_XML "Doxygen will generate XML output" OFF)

MARK_AS_ADVANCED(
  DOXYGEN_HTML
  DOXYGEN_DOCSET
  DOXYGEN_CHM
  DOXYGEN_QHP
  DOXYGEN_ECLIPSEHELP
  DOXYGEN_LATEX
  DOXYGEN_RTF
  DOXYGEN_XML
)

SET(DOXYGEN_OUTPUT
  HTML
  DOCSET
  CHM
  QHP
  ECLIPSEHELP
  LATEX
  RTF
  XML
)

FOREACH (var ${DOXYGEN_OUTPUT})
  IF (DOXYGEN_${var})
    SET(DOXYGEN_GEN_${var} "YES")
  ELSE (DOXYGEN_${var})
    SET(DOXYGEN_GEN_${var} "NO")
  ENDIF (DOXYGEN_${var})
ENDFOREACH (var ${DOXYGEN_OUTPUT})

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in
               ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
               @ONLY)

FIND_PACKAGE( LATEX )

IF (NOT LATEX_COMPILER)
  MESSAGE( "Warning: LaTeX not found. Formulas will not be generated in documentation." )
ENDIF (NOT LATEX_COMPILER)

ADD_CUSTOM_TARGET(Doxygen ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/docs/doxygen/doxyfile
  MAIN_DEPENDENCY ${CMAKE_BINARY_DIR}/docs/doxygen/doxyfile
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen
  )
