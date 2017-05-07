cmake_minimum_required( VERSION 2.8.3 )
project( modprop )

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules")

if (NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

add_definitions( -Wall -std=c++0x )

find_package(catkin REQUIRED 
    COMPONENTS      roscpp
                    rospy
)

## Finding 3rd party packages
# Find Boost C++ Utility libraries
find_package( Boost REQUIRED COMPONENTS random )
list( APPEND modprop_LIBRARIES ${Boost_LIBRARIES} )
list( APPEND modprop_INCLUDES ${Boost_INCLUDE_DIRS} )

# Find Eigen linear algebra library
find_package( Eigen3 REQUIRED )
message( STATUS "Eigen found at: ${EIGEN3_INCLUDE_DIR}" )
list( APPEND modprop_INCLUDES ${EIGEN3_INCLUDE_DIR} )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" )

catkin_package(
    INCLUDE_DIRS    include
					${EIGEN3_INCLUDE_DIR}
    
	LIBRARIES       modprop

    CATKIN_DEPENDS  roscpp 
                    
    DEPENDS         Boost 
                    system_lib
                    Eigen3
)

catkin_python_setup()

include_directories( ${modprop_INCLUDES} ${catkin_INCLUDE_DIRS} include )

add_library( modprop 
	src/BasicModules.cpp
	src/GaussianLogLikelihood.cpp
	src/ModulesCore.cpp
	src/PredictModule.cpp
	src/UpdateModule.cpp )

# Test binaries
# add_executable( test_regression tests/RegressionTest.cpp )
# target_link_libraries( test_regression 
#                        modprop
#                        ${modprop_LIBRARIES} 
#                        ${catkin_LIBRARIES} )

# add_executable( test_derivative tests/DerivativeTest.cpp )
# target_link_libraries( test_derivative
#                        modprop
#                        ${modprop_LIBRARIES}
#                        ${catkin_LIBRARIES}  )

add_executable( test_kalman tests/KalmanTest.cpp )
target_link_libraries( test_kalman
                       modprop
                       ${modprop_LIBRARIES}
                       ${catkin_LIBRARIES}  )

install( TARGETS test_kalman
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)