# Find the Sphinx documentation generator
#
# This modules defines
#  SPHINX_EXECUTABLE
#  SPHINX_FOUND

FIND_PROGRAM(SPHINX_EXECUTABLE
             NAMES sphinx-build
             PATHS
               /usr/bin
               /usr/local/bin
               /opt/local/bin
             DOC "Sphinx documentation generator")

IF ( NOT SPHINX_EXECUTABLE )
  SET(_Python_VERSIONS 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0 1.6 1.5)

  FOREACH ( _version ${_Python_VERSIONS} )
    SET( _sphinx_NAMES sphinx-build-${_version} )

    FIND_PROGRAM(SPHINX_EXECUTABLE
                 NAMES ${_sphinx_NAMES}
                 PATHS
                   /usr/bin
                   /usr/local/bin
                   /opt/loca/bin
                 DOC "Sphinx documentation generator")
  ENDFOREACH ()
ENDIF ()

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)

MARK_AS_ADVANCED(SPHINX_EXECUTABLE)
