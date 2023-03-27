#! /bin/bash

echo "ROBOTNIK GAZEBO MODELS INSTALLATION"
mkdir -p ~/.gazebo/models

cp -r ./ar_tags/* ~/.gazebo/models/
cp -r . ~/.gazebo/models/
rm -r ~/.gazebo/models/ar_tags/

echo "Complete!!!"
