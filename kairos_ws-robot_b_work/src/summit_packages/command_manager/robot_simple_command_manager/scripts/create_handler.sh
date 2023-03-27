#!/bin/sh

OUTPUT_RED='\033[0;31m'
OUTPUT_NC='\033[0m'
OUTPUT_GREEN='\033[0;32m'
OUTPUT_ORANGE='\033[0;33m'
OUTPUT_YELLOW='\033[1;33m'

echo "${OUTPUT_YELLOW} THIS SCRIPT HAS TO BE RUN ONCE THE WORKSPACE HAS BEEN COMPILED ${OUTPUT_NC}"
echo "${OUTPUT_ORANGE}"
echo "###################################################################################################################"
echo "This script will create an initial version (from a template) of the handler you want to add."
echo "Please select the handler type number. It will be client of a:"
echo "1) action"
echo "2) service"
echo "3) procedure"
echo "4) subscriber"
echo "Any other key to finish"
echo "####################################################################################################################"
echo "${OUTPUT_NC}"

read response
if [ $response = "1" ]
then
  handler_type=action
elif [ $response = "2" ]
then
  handler_type=service
elif [ $response = "3" ]
then
  handler_type=procedure
elif [ $response = "4" ]
then
  handler_type=subscriber
else
  echo "See you next time!"
  exit
fi

echo "\`$handler_type\` has been selected"
echo 

target_folder=`rospack find robot_simple_command_manager`/src/robot_simple_command_manager/handlers/$handler_type\s/
templates_folder=`rospack find robot_simple_command_manager`/scripts/templates/
template_file=$templates_folder\template_$handler_type\_interface.py

echo "${OUTPUT_ORANGE}"
echo "###################################################################################################################"
echo "Write the name of your handler. Keep in mind:"
echo "1.- Use snake case"
echo "2.- Try to use a descriptive name"
echo "3.- The script will append the type of the handler to create the handler file"
echo "4.- The script will automatically locate the handler inside the handlers folder"
echo "###################################################################################################################"
echo "${OUTPUT_NC}"

read response
target_file=$response\_$handler_type\_interface.py
echo "The handler file will be $target_file"
temp_file=$templates_folder$target_file
cp $template_file $temp_file

target_class=$(echo $response | sed -r 's/(^|_)([a-z]|[0-9])/\U\2/g')
tag_name=$(echo $response | tr [a-z] [A-Z])
sed -i -e "s/TEMPLATE/$tag_name/g" $temp_file
sed -i -e "s/Template/$target_class/g" $temp_file
sed -i -e "s/template/$response/g" $temp_file
cp -v $temp_file $target_folder$target_file
rm -rfv $temp_file
