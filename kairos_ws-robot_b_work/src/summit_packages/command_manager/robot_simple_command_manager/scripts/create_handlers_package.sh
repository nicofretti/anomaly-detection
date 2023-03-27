#!/bin/sh

OUTPUT_RED='\033[0;31m'
OUTPUT_NC='\033[0m'
OUTPUT_GREEN='\033[0;32m'
OUTPUT_ORANGE='\033[0;33m'
OUTPUT_YELLOW='\033[1;33m'

target_folder=`rospack find robot_simple_command_manager`/../
templates_folder=`rospack find robot_simple_command_manager`/scripts/templates/template_custom_commands_package

echo "${OUTPUT_ORANGE}"
echo "###################################################################################################################"
echo "Write the name of your custom handlers package. Keep in mind:"
echo "1.- Use snake case"
echo "2.- Try to use a descriptive name"
echo "3.- The script will automatically locate the package inside the parent folder of robot_simple_command_manager"
echo "###################################################################################################################"
echo "${OUTPUT_NC}"

read response
new_package=$target_folder$response

cp -r $templates_folder $new_package

cd $new_package
rm CATKIN_IGNORE

package_file=$new_package/package.xml
sed -i -e "s/template_custom_commands_package/$response/g" $package_file

cmake_file=$new_package/CMakeLists.txt
sed -i -e "s/template_custom_commands_package/$response/g" $cmake_file

pyton_setup_file=$new_package/setup.py
sed -i -e "s/template_custom_commands_package/$response/g" $pyton_setup_file

mv $new_package/src/template_custom_commands_package $new_package/src/$response