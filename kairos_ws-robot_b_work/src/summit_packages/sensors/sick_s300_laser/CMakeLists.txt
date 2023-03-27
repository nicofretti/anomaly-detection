cmake_minimum_required(VERSION 2.8.3)
project(sick_s300_laser)

find_package(catkin REQUIRED COMPONENTS diagnostic_msgs roscpp sensor_msgs std_msgs)

find_package(Boost REQUIRED COMPONENTS date_time thread)

catkin_package()

### BUILD ###
include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
  src/ScannerSickS300.cpp
  src/SerialIO.cpp
  src/${PROJECT_NAME}.cpp
)


add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

### INSTALL ###
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
