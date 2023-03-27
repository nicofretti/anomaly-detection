cmake_minimum_required(VERSION 2.8.3)
project(robot_command_scheduler)

find_package(catkin REQUIRED COMPONENTS
  rospy
  actionlib
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_command_scheduler
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS src/robot_command_scheduler/__init__.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS src/robot_command_scheduler/command_scheduler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


catkin_install_python(PROGRAMS src/robot_command_scheduler/command_scheduler_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
