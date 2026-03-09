#!/bin/bash
sudo chmod -R 777 /dev/tty*
source ./devel/setup.bash 
roslaunch livelybot_bringup arm_task_space_impedance.launch
