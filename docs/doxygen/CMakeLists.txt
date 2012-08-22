include (${CMAKE_ROOT}/Modules/Documentation.cmake)

# -------------------------------------------------------------
#
# Generate Doxygen configuration file in all configuration file
#
if( ${DOXYGEN_DOT_FOUND} )
  set( HAVE_DOT "YES" )
else()
  set( HAVE_DOT "NO" )
endif()

option( DOXYGEN_HTML "Doxygen will generate HTML Output" ON )
option( DOXYGEN_DOCSET "Doxygen will generate additional index files for XCode 3" OFF )
option( DOXYGEN_CHM "Doxygen will generate Microsoft HTML help" OFF )
option( DOXYGEN_QHP "Doxygen will generate Qt Compressed Help" OFF )
option( DOXYGEN_ECLIPSEHELP "Doxygen will generate Eclipse help plugin" OFF )
option( DOXYGEN_LATEX "Doxygen will generate LaTeX output" OFF )
option( DOXYGEN_RTF "Doxygen will generate RTF output optimized for Word 97" OFF )
option( DOXYGEN_XML "Doxygen will generate XML output" OFF )

mark_as_advanced( DOXYGEN_HTML
  DOXYGEN_DOCSET
  DOXYGEN_CHM
  DOXYGEN_QHP
  DOXYGEN_ECLIPSEHELP
  DOXYGEN_LATEX
  DOXYGEN_RTF
  DOXYGEN_XML
)

set( DOXYGEN_OUTPUT
  HTML
  DOCSET
  CHM
  QHP
  ECLIPSEHELP
  LATEX
  RTF
  XML
)

foreach( var ${DOXYGEN_OUTPUT} )
  if( DOXYGEN_${var} )
    set( DOXYGEN_GEN_${var} "YES" )
  else()
    set( DOXYGEN_GEN_${var} "NO" )
  endif()
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in
               ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
               @ONLY)

find_package( LATEX )

if( NOT LATEX_COMPILER )
  message( "Warning: LaTeX not found. Formulas will not be generated in documentation." )
endif()

add_custom_target(Doxygen ALL
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/docs/doxygen/doxyfile
  MAIN_DEPENDENCY ${CMAKE_BINARY_DIR}/docs/doxygen/doxyfile
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen
  )
