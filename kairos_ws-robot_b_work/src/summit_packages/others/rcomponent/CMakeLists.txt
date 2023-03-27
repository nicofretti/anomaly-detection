cmake_minimum_required(VERSION 2.8.3)
project(rcomponent)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  robotnik_msgs
  diagnostic_msgs
  diagnostic_updater
  self_test
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    ${PROJECT_NAME}
  CATKIN_DEPENDS
    robotnik_msgs
  DEPENDS
)

include_directories(
   include
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
add_library(rcomponent
   include/rcomponent/rcomponent.h
   include/rcomponent/rcomponent_log_macros.h
   src/rcomponent.cpp
)

## Declare a cpp executable
add_executable(rcomponent_server src/simple_service_component.cpp)
add_executable(rcomponent_simple src/simple_component.cpp)
add_executable(rcomponent_simple_async src/simple_component_async.cpp)
add_executable(rcomponent_diagnostic src/simple_diagnostics_component.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
add_dependencies(rcomponent robotnik_msgs_generate_messages_cpp)
add_dependencies(rcomponent_server robotnik_msgs_generate_messages_cpp)
add_dependencies(rcomponent_simple robotnik_msgs_generate_messages_cpp)
add_dependencies(rcomponent_simple_async robotnik_msgs_generate_messages_cpp)
add_dependencies(rcomponent_diagnostic robotnik_msgs_generate_messages_cpp)


target_link_libraries(rcomponent_server
   rcomponent
   ${catkin_LIBRARIES}
)
target_link_libraries(rcomponent_simple
   rcomponent
   ${catkin_LIBRARIES}
)
target_link_libraries(rcomponent_simple_async
   rcomponent
   ${catkin_LIBRARIES}
)
target_link_libraries(rcomponent_diagnostic
   rcomponent
   ${catkin_LIBRARIES}
)

#############
## Install ##
#############


## Mark executables and/or libraries for installation
install(TARGETS rcomponent
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
)



#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    find_package(rosunit REQUIRED)

##    catkin_add_gtest(rosunit_rcomponent test/rosunit_rcomponent.cpp)
##    target_link_libraries(rosunit_rcomponent ${catkin_LIBRARIES} rcomponent)
    add_rostest_gtest(rosunit_rcomponent_log test/rosunit_rcomponent_log.test test/rosunit_rcomponent_log.cpp)
    target_link_libraries(rosunit_rcomponent_log ${catkin_LIBRARIES} procedures)

    add_executable(test_namespace test/test_namespace.cpp)
    target_link_libraries(test_namespace ${catkin_LIBRARIES} rcomponent)
endif()
