cmake_minimum_required(VERSION 2.8.3)
project(summit_xl_bringup)

find_package(catkin REQUIRED COMPONENTS 
    message_runtime 
    std_msgs    
    std_srvs
    std_msgs
    tf
    mavros
    map_nav_manager
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

install( DIRECTORY calibration/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/calibration
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY env/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/env
)

install(DIRECTORY install/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/install
)

install( DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install( DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)

install( DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)
