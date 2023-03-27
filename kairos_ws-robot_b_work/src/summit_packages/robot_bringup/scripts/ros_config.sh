#!/bin/bash
robot_params="/home/robot/kairos_ws/src/summit_packages/robot_bringup/env"


echo "All the env variables configuration are located at $robot_params"

source $robot_params/bringup.env
source $robot_params/robot_params.env
source $robot_params/sensors_params.env
source $robot_params/battery_params.env
source $robot_params/localization_params.env
source $robot_params/manipulation_params.env
source $robot_params/navigation_params.env
source $robot_params/perception_params.env
source $robot_params/rlc_params.env
source $robot_params/simulation_params.env
source $robot_params/hmi_params.env

export ROS_MASTER_URI=http://$HOSTNAME:11311
export ROS_HOSTNAME=$HOSTNAME
source /opt/ros/melodic/setup.bash
source $ROBOT_WORKSPACE/devel/setup.bash

echo ""
echo -e "ROS_MASTER_URI\t= $ROS_MASTER_URI"
echo -e "ROS_HOSTNAME\t= $ROS_HOSTNAME"
echo -e "WORKSPACE \t= $ROBOT_WORKSPACE/devel/setup.bash"
echo ""
