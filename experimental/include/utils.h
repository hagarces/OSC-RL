

#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <franka/gripper.h>

#include <franka_gripper/GraspAction.h>
#include <franka_gripper/HomingAction.h>
#include <franka_gripper/MoveAction.h>
#include <franka_gripper/StopAction.h>

#include <actionlib/client/simple_action_client.h>

#include <ros/init.h>
#include <ros/node_handle.h>


#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>

#include <franka_control_test/SetPointQ.h>
#include <franka_control_test/SetPointXYZ.h>
#include <franka_control_test/SetPointVXYZ.h>
#include <controller_manager_msgs/LoadController.h>
#include <controller_manager_msgs/UnloadController.h>
#include <controller_manager_msgs/SwitchController.h>



void actionctrlCallBack(const std_msgs::String::ConstPtr& msg);

void load_controllers(ros::ServiceClient* load_client);

void switch_controllers(ros::ServiceClient* switch_client);

void wait_results(ros::NodeHandle& nh, float time_out);


void call_setpoint_q(ros::NodeHandle& nh, ros::ServiceClient* servidor, double q_delta_ref[7], float time, bool global_reference, float time_out);

void call_setpoint_xyz(ros::NodeHandle& nh, ros::ServiceClient* servidor, double x_delta_ref, double y_delta_ref,
		      double z_delta_ref, float time, float time_out);

void set_array(double q_delta_ref[7], double q0, double q1, double q2, double q3, double q4, double q5, double q6);


#endif

