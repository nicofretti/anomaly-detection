cmake_minimum_required(VERSION 2.8.3)
project(summit_xl_controller)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  controller_interface
  nav_msgs
  tf
  urdf
  robotnik_msgs
)



###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES summit_xl_controller
#  CATKIN_DEPENDS controller_interface nav_msgs tf urdf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(summit_xl_controller
  src/summit_xl_controller.cpp
)

add_dependencies(summit_xl_controller robotnik_msgs_generate_messages_cpp)

target_link_libraries(summit_xl_controller
   ${catkin_LIBRARIES}
)

install(TARGETS summit_xl_controller
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(FILES summit_xl_controller_plugins.xml
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


