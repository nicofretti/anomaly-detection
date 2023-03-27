cmake_minimum_required(VERSION 2.8.3)
project(robot_complex_command_sequencer)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_complex_command_sequencer
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS src/robot_complex_command_sequencer/__init__.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS src/robot_complex_command_sequencer/complex_command_sequencer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


catkin_install_python(PROGRAMS src/robot_complex_command_sequencer/complex_command_sequencer_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
