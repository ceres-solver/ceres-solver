FIND_PACKAGE(Sphinx REQUIRED)

IF (NOT DEFINED SPHINX_THEME)
  SET (SPHINX_THEME armstrong)
ENDIF ()

IF (NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_themes")
ENDIF ()

# Configured documentation tools and intermediate build results
SET (BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
SET (SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
SET (SPHINX_HTML_DIR "${CMAKE_BINARY_DIR}/docs/html")

CONFIGURE_FILE ("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
                "${BINARY_BUILD_DIR}/conf.py"
                @ONLY)

ADD_CUSTOM_TARGET (my_project_docs ALL
                   ${SPHINX_EXECUTABLE}
                   -q -b html
                   -c "${BINARY_BUILD_DIR}"
                   -d "${SPHINX_CACHE_DIR}"
                   "${CMAKE_CURRENT_SOURCE_DIR}"
                   "${SPHINX_HTML_DIR}"
                   COMMENT "Building HTML documentation with Sphinx")
