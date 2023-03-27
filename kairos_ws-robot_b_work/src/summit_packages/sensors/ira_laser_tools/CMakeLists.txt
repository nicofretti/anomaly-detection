cmake_minimum_required(VERSION 2.8.3)
project(ira_laser_tools)

add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS laser_geometry roscpp sensor_msgs std_msgs tf dynamic_reconfigure pcl_ros)

find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED)

generate_dynamic_reconfigure_options(cfg/LaserscanMultiMerger.cfg cfg/laserscan_virtualizer.cfg)


catkin_package(
   INCLUDE_DIRS
#  LIBRARIES laser_merger
#  CATKIN_DEPENDS laser_geometry roscpp sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(laserscan_multi_merger src/laserscan_multi_merger.cpp)
add_dependencies(laserscan_multi_merger ${PROJECT_NAME}_gencfg)
target_link_libraries(laserscan_multi_merger ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(laserscan_virtualizer src/laserscan_virtualizer.cpp)
add_dependencies(laserscan_virtualizer ${PROJECT_NAME}_gencfg)
target_link_libraries(laserscan_virtualizer ${catkin_LIBRARIES} ${PCL_LIBRARIES})

install(TARGETS laserscan_multi_merger laserscan_virtualizer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

