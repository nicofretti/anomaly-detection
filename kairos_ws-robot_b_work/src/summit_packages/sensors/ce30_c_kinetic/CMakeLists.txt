cmake_minimum_required(VERSION 2.8.3)
project(ce30c_driver)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(PCL REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ce30c_driver
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)

link_directories("${PROJECT_SOURCE_DIR}")
add_executable(ce30_cloud src/ce30_cloud.cpp)

target_link_libraries(ce30_cloud ${catkin_LIBRARIES} ${PCL_LIBRARIES} libbw_ce30v2.0.so)