#!/bin/sh

OUTPUT_RED='\033[0;31m'
OUTPUT_NC='\033[0m'
OUTPUT_GREEN='\033[0;32m'
OUTPUT_ORANGE='\033[0;33m'
OUTPUT_YELLOW='\033[1;33m'

echo "${OUTPUT_YELLOW} THIS SCRIPT HAS TO BE RUN JUST ONCE! otherwise delete the added to the ~/.bashrc ${OUTPUT_NC}"

echo "${OUTPUT_ORANGE}"
echo "###################################################################################################################"
echo "This script will copy the most common parameters for the different models, although you should revise it later..."
echo "Please select the robot model:"
echo "1) summit_xl"
echo "2) summit_xls"
echo "3) summit_xl_gen"
echo "4) rbkairos"
echo "Any other key if you are not sure"
echo "####################################################################################################################"
echo -e "${OUTPUT_NC}"

read response
if [ $response = "1" ]
then
  robot_params_file=summit_xl_params.env
elif [ $response = "2" ]
then
  robot_params_file=summit_xls_params.env
elif [ $response = "3" ]
then
  robot_params_file=summit_xl_gen_params.env
elif [ $response = "4" ]
then
  robot_params_file=rbkairos_params.env
else
  robot_params_file=robot_params.env
fi

config_folder=`rospack find summit_xl_bringup`/env
target_folder=~
robot_params_target_file=robot_params.env
autorun_file=autorun.bash

cp -v $config_folder/$robot_params_file $target_folder/$robot_params_target_file
cp -v $config_folder/$autorun_file $target_folder

echo "source $target_folder/robot_params.env" >> ~/.bashrc
echo "source $target_folder/autorun.bash" >> ~/.bashrc
