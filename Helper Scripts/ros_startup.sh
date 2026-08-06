#!/bin/bash

echo "=============================="
echo " ROS 2 Container Startup"
echo "=============================="

echo "[1] Stopping old ROS daemon..."
pkill -9 -f ros2-daemon || true

echo "[2] Removing ROS daemon cache..."
rm -rf ~/.ros/ros2cli

echo "[3] Starting ROS daemon..."
ros2 daemon start

echo "[4] Checking ROS daemon status..."
ros2 daemon status

echo "[5] Container IP Address:"
hostname -I

echo "[6] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[7] Sourcing RoboMaster workspace..."
source /opt/ros/ws/setup.bash

echo "=============================="
echo " ROS Environment Ready"
echo "=============================="

exec bash
