ADD_SUBDIRECTORY(userguide)

OPTION(BUILD_DOXYGEN
       "Build doxygen documentation."
       OFF)

IF(${BUILD_DOXYGEN})
  MESSAGE("-- Build doxygen documentation.")
  ADD_SUBDIRECTORY(doxygen)
ENDIF(${BUILD_DOXYGEN})
