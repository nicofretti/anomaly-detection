cmake_minimum_required(VERSION 2.8.3)
project(robotnik_pad)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  robotnik_msgs
  rcomponent
  sensor_msgs
  geometry_msgs
  ackermann_msgs
  pluginlib
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES robotnik_pad
  CATKIN_DEPENDS roscpp rcomponent robotnik_msgs sensor_msgs geometry_msgs ackermann_msgs pluginlib
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(robotnik_pad
  src/robotnik_pad.cpp
)

add_dependencies(robotnik_pad ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(robotnik_pad_node src/robotnik_pad_node.cpp)
add_dependencies(robotnik_pad_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(robotnik_pad ${catkin_LIBRARIES})

target_link_libraries(robotnik_pad_node
  ${catkin_LIBRARIES}
  robotnik_pad
)

#############
## Install ##
#############

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS robotnik_pad robotnik_pad_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
