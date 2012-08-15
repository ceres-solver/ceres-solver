FIND_PACKAGE( LATEX REQUIRED )

FILE( DOWNLOAD http://minted.googlecode.com/files/minted.sty
  ${CMAKE_BINARY_DIR}/docs/minted.sty
  SHOW_PROGRESS
)

ADD_CUSTOM_TARGET( UserGuide ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/docs ${CMAKE_BINARY_DIR}/docs
  COMMAND ${PDFLATEX_COMPILER}
    -shell-escape ${CMAKE_BINARY_DIR}/docs/ceres-solver.tex
    -output-directory ${CMAKE_BINARY_DIR}/docs
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs
  COMMAND ${BIBTEX_COMPILER} ${CMAKE_BINARY_DIR}/docs/ceres-solver.aux
  COMMAND ${PDFLATEX_COMPILER}
    -shell-escape ${CMAKE_BINARY_DIR}/docs/ceres-solver.tex
    -output-directory ${CMAKE_BINARY_DIR}/docs
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs
  COMMAND ${PDFLATEX_COMPILER}
    -shell-escape ${CMAKE_BINARY_DIR}/docs/ceres-solver.tex
    -output-directory ${CMAKE_BINARY_DIR}/docs
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs

)



