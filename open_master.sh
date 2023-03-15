#!/bin/bash
docker_name=$(docker ps | grep "ros-visualizer" | awk '{print $1}')
docker exec -it $docker_name bash
