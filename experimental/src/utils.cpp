#include <franka_control_test/utils.h>
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


using franka_gripper::GraspAction;
using franka_gripper::HomingAction;
using franka_gripper::MoveAction;
using franka_gripper::StopAction;


using GraspClient = actionlib::SimpleActionClient<GraspAction>;
using HomingClient = actionlib::SimpleActionClient<HomingAction>;
using MoveClient = actionlib::SimpleActionClient<MoveAction>;
using StopClient = actionlib::SimpleActionClient<StopAction>;

using hominggoal_ = franka_gripper::HomingGoal; 
using stopgoal_ = franka_gripper::StopGoal;

bool ActionReady = true;

void actionctrlCallBack(const std_msgs::String::ConstPtr& msg){
  ActionReady = true;
}

void load_controllers(ros::ServiceClient* load_client){

  controller_manager_msgs::LoadController load_msg;
  
  load_msg.request.name = "robot_q_controller";
  load_client->call(load_msg);
  
  load_msg.request.name = "robot_xyz_controller";
  load_client->call(load_msg);
  
  load_msg.request.name = "robot_vxyz_controller";
  load_client->call(load_msg);

}

void switch_controllers(ros::ServiceClient* switch_client){

  controller_manager_msgs::SwitchController switch_msg;
}


void wait_results(ros::NodeHandle& nh, float time_out){

  ros::Time end = ros::Time::now() + ros::Duration(time_out);
  while (!ActionReady && nh.ok()){
    if (ros::Time::now() >= end){
      ROS_ERROR("la accion tomo mas tiempo que el timeout!");
      return;
    }
  }
  ROS_INFO("accion terminada");
}



void call_setpoint_q(ros::NodeHandle& nh, ros::ServiceClient* servidor, double q_delta_ref[7], float time, bool global_reference, float time_out){
  
  franka_control_test::SetPointQ setpoint_q_msg;
  
  setpoint_q_msg.request.action_time = time;
  setpoint_q_msg.request.global_reference = global_reference;
  
  for (size_t i = 0; i < 7; ++i) {
  setpoint_q_msg.request.q_delta_ref[i] = q_delta_ref[i];
  }
  
  ActionReady = false;
  servidor->call(setpoint_q_msg);
  
  wait_results(nh, time_out);
}

void call_setpoint_xyz(ros::NodeHandle& nh, ros::ServiceClient* servidor, double x_delta_ref, double y_delta_ref, double z_delta_ref, float time, float time_out){
  
  franka_control_test::SetPointXYZ setpoint_xyz_msg;
  
  setpoint_xyz_msg.request.action_time = time;
  
  setpoint_xyz_msg.request.x_delta_ref = x_delta_ref;
  setpoint_xyz_msg.request.y_delta_ref = y_delta_ref;
  setpoint_xyz_msg.request.z_delta_ref = z_delta_ref;
  
  ActionReady = false;
  servidor->call(setpoint_xyz_msg);
  
  wait_results(nh, time_out);
}

void  set_array(double q_delta_ref[7], double q0, double q1, double q2, double q3, double q4, double q5, double q6){
    q_delta_ref[0] = q0;
    q_delta_ref[1] = q1;
    q_delta_ref[2] = q2;
    q_delta_ref[3] = q3;
    q_delta_ref[4] = q4;
    q_delta_ref[5] = q5;
    q_delta_ref[6] = q6;
}






