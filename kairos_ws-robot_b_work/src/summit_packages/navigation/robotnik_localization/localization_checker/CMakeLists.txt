cmake_minimum_required(VERSION 2.8.3)
project(localization_checker)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  laser_geometry
  pcl_conversions
  pcl_ros
  roscpp
  tf
  tf_conversions
)

generate_dynamic_reconfigure_options(
  cfg/LocalizationChecker.cfg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES localization_checker
#  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp tf
#  DEPENDS system_lib
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)

add_executable(localization_checker_node src/localization_checker_node.cpp src/localization_checker.cpp)

add_dependencies(localization_checker_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(localization_checker_node
  ${catkin_LIBRARIES}
)

install( DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

install(TARGETS localization_checker_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


#set(CATKIN_ENABLE_TESTING 0)

if(CATKIN_ENABLE_TESTING)
  #add_executable(raytracer_test test/raytracer_test.cpp)
  #add_dependencies(raytracer_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  #target_link_libraries(raytracer_test
  #  ${catkin_LIBRARIES}
  #)
  
  #add_executable(matcher_test test/matcher_test.cpp)
  #add_dependencies(matcher_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  #target_link_libraries(matcher_test
  #  ${catkin_LIBRARIES}
  #)
  
  catkin_add_gtest(raytracer_test test/raytracer_test.cpp)
  target_link_libraries(raytracer_test ${catkin_LIBRARIES})

  catkin_add_gtest(matcher_test test/matcher_test.cpp)
  target_link_libraries(matcher_test ${catkin_LIBRARIES})

  catkin_add_gtest(localization_scorer_test test/localization_scorer_test.cpp)
  target_link_libraries(localization_scorer_test ${catkin_LIBRARIES})

endif()
