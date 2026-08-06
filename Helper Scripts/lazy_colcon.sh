#!/bin/bash

echo "=============================="
echo " ROS 2 Pkg Rebuilds"
echo "=============================="

echo "Colcon is Building..."
colcon build

echo "ros2 run ece687_project_pkg..."
ros2 run ece687_project_pkg

echo "Source it now..."
source install/setup.bash

echo "now it is running... Check out robot"
ros2 run ece687_project_pkg project_node

exec bash
