FIND_PACKAGE(Sphinx REQUIRED)

IF (NOT DEFINED SPHINX_THEME)
  SET(SPHINX_THEME armstrong)
ENDIF ()

IF (NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_themes")
ENDIF ()

# Configured documentation tools and intermediate build results
SET(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
SET(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
SET(SPHINX_HTML_DIR "${CMAKE_BINARY_DIR}/docs/html")

# Install documentation
INSTALL(DIRECTORY ${SPHINX_HTML_DIR}
        DESTINATION ceres/doc
        COMPONENT Doc
        PATTERN "${SPHINX_HTML_DIR}/*")

CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
               "${BINARY_BUILD_DIR}/conf.py"
               @ONLY)

# Building using 'make_docs.py' python script
ADD_CUSTOM_TARGET(my_project_docs ALL
                  python
                  "${CMAKE_SOURCE_DIR}/scripts/make_docs.py"
                  "${CMAKE_CURRENT_SOURCE_DIR}"
                  "${BINARY_BUILD_DIR}"
                  "${SPHINX_CACHE_DIR}"
                  "${SPHINX_HTML_DIR}"
                  COMMENT "Building HTML documentation with Sphinx")
