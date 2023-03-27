cmake_minimum_required(VERSION 2.8.3)
project(robotnik_base_hw_sim)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  std_msgs
  controller_interface
  robotnik_msgs
  rospy
  message_generation
  geometry_msgs
)

add_message_files(
    DIRECTORY msg
    FILES
    PickState.msg
    PickStates.msg
)
add_service_files(
    DIRECTORY srv
    FILES
    Pick.srv
    Place.srv
    SimplePick.srv
    SimplePlace.srv
)

catkin_python_setup()

generate_messages(
DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   geometry_msgs
)


catkin_package()

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
add_library(robotnik_elevator_controller
  src/robotnik_elevator_controller.cpp
  include/robotnik_base_hw_sim/robotnik_elevator_controller.h
)

add_dependencies(robotnik_elevator_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} robotnik_msgs_generate_messages_cpp)

target_link_libraries(robotnik_elevator_controller
   ${catkin_LIBRARIES}
)

install(TARGETS robotnik_elevator_controller
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

install(FILES robotnik_elevator_controller_plugins.xml
DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(PROGRAMS
   src/robotnik_base_hw_sim/robotnik_base_hw_sim_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
