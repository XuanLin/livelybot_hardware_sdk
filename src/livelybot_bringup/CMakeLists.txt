cmake_minimum_required(VERSION 3.0.2)
project(livelybot_bringup)

# add_definitions(-DDEBUG)

set(pinocchio_DIR /opt/ros/noetic/lib/cmake/pinocchio)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  serial  
  livelybot_msg  
  livelybot_serial
  livelybot_power
  pinocchio
)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs
  # DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(canboard_update src/canboard_update.cpp)
target_link_libraries(canboard_update ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(motor_feedback src/motor_feedback.cpp)
target_link_libraries(motor_feedback ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(motor_rerun src/motor_rerun.cpp)
target_link_libraries(motor_rerun ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(motor_move_zero src/motor_move_zero.cpp)
target_link_libraries(motor_move_zero ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(motor_set_zero src/motor_set_zero.cpp)
target_link_libraries(motor_set_zero ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(motor_timeout src/motor_timeout.cpp)
target_link_libraries(motor_timeout ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

# =================================================================================
add_executable(arm_joint_survey src/arm_joint_survey.cpp)
target_link_libraries(arm_joint_survey ${catkin_LIBRARIES})

add_executable(arm_motor_sign_test src/arm_motor_sign_test.cpp)
target_link_libraries(arm_motor_sign_test ${catkin_LIBRARIES})
target_include_directories(arm_motor_sign_test PRIVATE /usr/include/eigen3)
target_link_libraries(arm_motor_sign_test ${catkin_LIBRARIES} pinocchio::pinocchio)

add_executable(arm_gravity_comp src/arm_gravity_comp.cpp)
target_link_libraries(arm_gravity_comp ${catkin_LIBRARIES})
target_include_directories(arm_gravity_comp PRIVATE /usr/include/eigen3)
target_link_libraries(arm_gravity_comp ${catkin_LIBRARIES} pinocchio::pinocchio)

add_executable(arm_traj_following src/arm_traj_following.cpp)
target_link_libraries(arm_traj_following ${catkin_LIBRARIES})
target_include_directories(arm_traj_following PRIVATE /usr/include/eigen3)
target_link_libraries(arm_traj_following ${catkin_LIBRARIES} pinocchio::pinocchio)

add_executable(arm_sysid_trajectory src/arm_sysid_trajectory.cpp)
target_include_directories(arm_sysid_trajectory PRIVATE /usr/include/eigen3)
target_link_libraries(arm_sysid_trajectory ${catkin_LIBRARIES} pinocchio::pinocchio)

add_executable(arm_task_space_impedance src/arm_task_space_impedance.cpp)
target_link_libraries(arm_task_space_impedance ${catkin_LIBRARIES})
target_include_directories(arm_task_space_impedance PRIVATE /usr/include/eigen3)
target_link_libraries(arm_task_space_impedance ${catkin_LIBRARIES} pinocchio::pinocchio)
# =================================================================================

add_executable(right_arm_torque_logger src/right_arm_torque_logger.cpp)
target_link_libraries(right_arm_torque_logger ${catkin_LIBRARIES} ${SERIALPORT_LIBRARY})

add_executable(enable_motor_power src/enable_motor_power.cpp)
target_link_libraries(enable_motor_power ${catkin_LIBRARIES})


install(TARGETS canboard_update
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS motor_feedback
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS motor_move_zero
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS motor_rerun
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS motor_set_zero
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(TARGETS motor_timeout
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
 
install(DIRECTORY script/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/script
)
