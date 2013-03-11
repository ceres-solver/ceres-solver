# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /usr/local/homebrew/lib) # Mac OS X
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /usr/local/homebrew/include) # Mac OS X
LIST(APPEND SEARCH_HEADERS /opt/local/include)

# Locations to search for Eigen
SET(EIGEN_SEARCH_HEADERS ${SEARCH_HEADERS})
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/include/eigen3) # Ubuntu 10.04's default location.
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/include/eigen3)
LIST(APPEND EIGEN_SEARCH_HEADERS /usr/local/homebrew/include/eigen3)  # Mac OS X
LIST(APPEND EIGEN_SEARCH_HEADERS /opt/local/var/macports/software/eigen3/opt/local/include/eigen3) # Mac OS X

# Google Flags
OPTION(GFLAGS
       "Enable Google Flags."
       ON)

OPTION(BUILD_ANDROID
       "Build for Android. Use build_android.sh instead of setting this."
       OFF)

IF (${GFLAGS})
  MESSAGE("-- Check for Google Flags")
  FIND_LIBRARY(GFLAGS_LIB NAMES gflags PATHS ${SEARCH_LIBS})
  IF (NOT EXISTS ${GFLAGS_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_LIB=...")
  ENDIF (NOT EXISTS ${GFLAGS_LIB})
  MESSAGE("-- Found Google Flags library: ${GFLAGS_LIB}")
  FIND_PATH(GFLAGS_INCLUDE NAMES gflags/gflags.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GFLAGS_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Flags. Please specify: "
            "-DGFLAGS_INCLUDE=...")
  ENDIF (NOT EXISTS ${GFLAGS_INCLUDE})
  MESSAGE("-- Found Google Flags header in: ${GFLAGS_INCLUDE}")
ELSE (${GFLAGS})
  MESSAGE("-- Google Flags disabled; no tests or tools will be built!")
  ADD_DEFINITIONS(-DCERES_NO_GFLAGS)
ENDIF (${GFLAGS})

# Google Logging
IF (NOT ${BUILD_ANDROID})
  MESSAGE("-- Check for Google Log")
  FIND_LIBRARY(GLOG_LIB NAMES glog PATHS ${SEARCH_LIBS})
  IF (NOT EXISTS ${GLOG_LIB})
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: "
            "-DGLOG_LIB=...")
  ENDIF (NOT EXISTS ${GLOG_LIB})
  MESSAGE("-- Found Google Log library: ${GLOG_LIB}")

  FIND_PATH(GLOG_INCLUDE NAMES glog/logging.h PATHS ${SEARCH_HEADERS})
  IF (NOT EXISTS ${GLOG_INCLUDE})
    MESSAGE(FATAL_ERROR
            "Can't find Google Log. Please specify: "
            "-DGLOG_INCLUDE=...")
  ENDIF (NOT EXISTS ${GLOG_INCLUDE})
  MESSAGE("-- Found Google Log header in: ${GLOG_INCLUDE}")
ELSE (NOT ${BUILD_ANDROID})
  SET(GLOG_LIB miniglog)
  MESSAGE("-- Using minimal Glog substitute for Android (library): ${GLOG_LIB}")
  SET(GLOG_INCLUDE internal/ceres/miniglog)
  MESSAGE("-- Using minimal Glog substitute for Android (include): ${GLOG_INCLUDE}")
ENDIF (NOT ${BUILD_ANDROID})

# Eigen
MESSAGE("-- Check for Eigen 3.0")
FIND_PATH(EIGEN_INCLUDE NAMES Eigen/Core PATHS ${EIGEN_SEARCH_HEADERS})
IF (NOT EXISTS ${EIGEN_INCLUDE})
  MESSAGE(FATAL_ERROR "Can't find Eigen. Try passing -DEIGEN_INCLUDE=...")
ENDIF (NOT EXISTS ${EIGEN_INCLUDE})
MESSAGE("-- Found Eigen 3.0: ${EIGEN_INCLUDE}")


INCLUDE_DIRECTORIES(
  ${GLOG_INCLUDE}
  ${EIGEN_INCLUDE}
  )
