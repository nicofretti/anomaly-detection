cmake_minimum_required(VERSION 2.8.3)
project(poi_manager)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
  visualization_msgs
  tf
  robotnik_msgs
  robot_local_control_msgs
)

catkin_python_setup()

catkin_package()


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination

catkin_install_python(PROGRAMS src/poi_manager/__init__.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS src/poi_manager/poi_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS src/poi_manager/poi.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS src/poi_manager/poi_marker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


install(
  DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
