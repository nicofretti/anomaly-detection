#!/bin/bash

# Open the terminal of a running container, default is ros-master

container="ros-master"
# If there is the first argument, it is the name of the container
if [ $# -eq 1 ]; then
    container=$1
fi

docker_name=$(docker ps | grep $container | awk '{print $1}')
docker exec -it $docker_name bash
