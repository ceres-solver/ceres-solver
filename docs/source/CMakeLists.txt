# HTML output directory
set(SPHINX_HTML_DIR "${Ceres_BINARY_DIR}/docs/html")
set(SPHINX_DOCTREES_DIR "${Ceres_BINARY_DIR}/docs/doctrees")

# Install documentation
install(DIRECTORY ${SPHINX_HTML_DIR}
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT Doc
        PATTERN "${SPHINX_HTML_DIR}/*")

add_custom_target(ceres_docs ALL
                  "${Sphinx_BUILD_EXECUTABLE}"
                  -n -a -b html
                  -d "${SPHINX_DOCTREES_DIR}"
                  "${CMAKE_CURRENT_SOURCE_DIR}"
                  "${SPHINX_HTML_DIR}"
                  USES_TERMINAL
                  COMMENT "Building HTML documentation with Sphinx")
