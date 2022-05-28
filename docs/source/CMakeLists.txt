find_package(Sphinx REQUIRED)

# HTML output directory
set(SPHINX_HTML_DIR "${Ceres_BINARY_DIR}/docs/html")

# Install documentation
install(DIRECTORY ${SPHINX_HTML_DIR}
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT Doc
        PATTERN "${SPHINX_HTML_DIR}/*")

# Find python 3
find_package(PythonInterp 3 REQUIRED)

# Building using 'make_docs.py' python script
add_custom_target(ceres_docs ALL
                  ${PYTHON_EXECUTABLE}
                  "${Ceres_SOURCE_DIR}/scripts/make_docs.py"
                  "${Ceres_SOURCE_DIR}"
                  "${Ceres_BINARY_DIR}/docs"
                  "${SPHINX_EXECUTABLE}"
                  USES_TERMINAL
                  COMMENT "Building HTML documentation with Sphinx")
