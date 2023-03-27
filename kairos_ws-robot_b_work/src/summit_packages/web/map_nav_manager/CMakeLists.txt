cmake_minimum_required(VERSION 2.8.3)
project(map_nav_manager)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    roscpp
    std_srvs
    message_generation
    tf2_web_republisher
    rosbridge_server
    amcl
    move_base
    gmapping
    robotnik_msgs
    interactive_marker_proxy
    poi_manager
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()


add_message_files(
   FILES
   State.msg
)


## Generate services in the 'srv' folder
add_service_files(
   FILES
   SetFilename.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
   robotnik_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES map_nav_manager
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)
