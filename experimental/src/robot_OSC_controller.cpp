#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include "std_msgs/String.h"
#include <array>

#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <pluginlib/class_list_macros.h>

#include <franka_control_test/SetPointXYZ.h>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>





namespace robot_controllers {

void TorqueSafetyLimiter(std::array<double, 7> &tau_d_array,
                         std::array<double, 7> &min_torque, 
                         std::array<double, 7> &max_torque) {
  for (size_t i = 0; i < tau_d_array.size(); i++) {
    if (tau_d_array[i] < min_torque[i]) {
      tau_d_array[i] = min_torque[i];
    } else if (tau_d_array[i] > max_torque[i]) {
      tau_d_array[i] = max_torque[i];
    }
  }
}

void TorqueMinMovement(std::array<double, 7> &tau_d_array,
                         std::array<double, 7> &min_plus_torque, 
                         std::array<double, 7> &min_minus_torque,
                         double near_rate) {
                         
                         
  double min_kp_tau = 0.02;
                      
  for (size_t i = 0; i < tau_d_array.size(); i++) {
    
    if (tau_d_array[i] > min_kp_tau){
      tau_d_array[i] = tau_d_array[i] + min_plus_torque[i] * near_rate;
    
    } else if (tau_d_array[i] < -min_kp_tau) {
      tau_d_array[i] = tau_d_array[i] + min_minus_torque[i] * near_rate;
    }
    else{
      if (tau_d_array[i] > 0){
        tau_d_array[i] = (tau_d_array[i]/min_kp_tau) * (tau_d_array[i] + min_plus_torque[i] * near_rate);
      }     
      else{
        tau_d_array[i] = (tau_d_array[i]/-min_kp_tau) * (tau_d_array[i] + min_minus_torque[i] * near_rate); 
      }
    }
  }
}


void TorqueAmplifier(std::array<double, 7> &tau_d_array,
                         std::array<double, 7> &acumulated_tau, 
                         double error_dist,
                         double success_thresh,
                         double dist_thresh_acumulation,
                         double min_torque_for_acumulation,
                         double acumulated_smooth_factor,
                         double max_acumulated_tau){
                     
   
  if (error_dist > dist_thresh_acumulation){
    for (size_t i = 0; i < tau_d_array.size(); i++) {
      if (acumulated_tau[i] > 0){
        acumulated_tau[i] = acumulated_tau[i] - acumulated_smooth_factor;
      }
      else if (acumulated_tau[i] < 0){
        acumulated_tau[i] = acumulated_tau[i] + acumulated_smooth_factor; 
      }
      tau_d_array[i] = tau_d_array[i] + acumulated_tau[i];
    }
  }
    

  else if (error_dist < success_thresh){
    for (size_t i = 0; i < tau_d_array.size(); i++) {
      tau_d_array[i] = tau_d_array[i] + acumulated_tau[i];
    }
  }
  
  
  else{
    for (size_t i = 0; i < tau_d_array.size(); i++) {
      if (tau_d_array[i] > min_torque_for_acumulation and acumulated_tau[i] < max_acumulated_tau){
        acumulated_tau[i] = acumulated_tau[i] + acumulated_smooth_factor; 
      }
      else if (tau_d_array[i] < -min_torque_for_acumulation and acumulated_tau[i] > -max_acumulated_tau){
        acumulated_tau[i] = acumulated_tau[i] - acumulated_smooth_factor; 
      }
      tau_d_array[i] = tau_d_array[i] + acumulated_tau[i];
    }
  }                      
}


void JointSafetyLimiter(std::array<double, 7> &tau_d_array,
			 std::array<double, 7> &q,
			 double safety_dist,
                        std::array<double, 7> &q_max_lim, 
                        std::array<double, 7> &q_min_lim) {
  for (size_t i = 0; i < tau_d_array.size(); i++) {
    

    if (q[i] > (q_max_lim[i] - safety_dist)){    

      if (tau_d_array[i] > 0){
        tau_d_array[i] = tau_d_array[i]  * std::max(((q_max_lim[i] - q[i])/(safety_dist)), -1.0);
        
        ROS_INFO("Joint: %zu is near its upper limit!, limiting torque...", i + 1);
      }
    }

    else if (q[i] < (q_min_lim[i] + safety_dist)){    

      if (tau_d_array[i] < 0){
        tau_d_array[i] = tau_d_array[i]  * std::max(((q[i] - q_min_lim[i])/(safety_dist)), -1.0);
        ROS_INFO("Joint: %zu is near its lower limit!, limiting torque...", i + 1);
      }
    }
  }
}

void CartesianVelocityLimiter(Eigen::Matrix<double, 3, 1> &xyz_error,
			       Eigen::Matrix<double, 3, 1> &tau_task,
			       double v_max, double kp, double kv) {

  double xyz_error_norm = xyz_error.norm();
  double sat_gain_xyz = (v_max / kp) * kv;
  Eigen::MatrixXd scale = xyz_error;
  
  if(xyz_error_norm > sat_gain_xyz){
    scale = scale * (sat_gain_xyz / xyz_error_norm);
  }
  
  tau_task = kp * scale;
  
}

void DTorqueSafetyLimiter(std::array<double, 7> &tau_d_array,
			   std::array<double, 7> &tau_d_array_previous,
			   double max_d_torque, double dt){
			   
  for (size_t i = 0; i < tau_d_array.size(); i++) {
    double d_tau = (tau_d_array[i] - tau_d_array_previous[i])/dt;
    
    tau_d_array[i] = tau_d_array_previous[i] + std::max(std::min(d_tau, max_d_torque), -max_d_torque) * dt;

  }  
}


void Mx_calculator(Eigen::MatrixXd &M_inv, Eigen::MatrixXd &Mx, Eigen::MatrixXd &jacobian, double epsilon = 0.00025) {



  Eigen::MatrixXd Mx_inv(3, 3);
  Mx_inv << jacobian * M_inv * jacobian.transpose();
  
  const float det_threshold = 0.001;
  double Mx_inv_det = abs(Mx_inv.determinant());
  
  if (Mx_inv_det >= det_threshold){
    Mx = Mx_inv.inverse();
    
  } else{

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Mx_inv, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType singular_vals =
        svd.singularValues();
        
    Eigen::MatrixXd Mx_inv_aux = Mx_inv;
    
    Mx_inv_aux.setZero();
    
    for (int i = 0; i < singular_vals.size(); i++) {
      if (singular_vals(i) < epsilon) {
        Mx_inv_aux(i, i) = 0.;
      } else {
        Mx_inv_aux(i, i) = 1. / singular_vals(i);
      }
    }
      
     
    Mx = Eigen::MatrixXd(svd.matrixV() * Mx_inv_aux * svd.matrixU().transpose()); 
  }
}
  

class RobotOSCController
    : public controller_interface::MultiInterfaceController<franka_hw::FrankaModelInterface,
                                   			      hardware_interface::EffortJointInterface,
                                   			      franka_hw::FrankaStateInterface> {                                         
 private:
 
  hardware_interface::EffortJointInterface* effort_joint_interface;
  franka_hw::FrankaStateInterface* state_interface;
  franka_hw::FrankaModelInterface* model_interface;
  
  
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  

  ros::Duration elapsed_time_; 
  std::array<double, 16> initial_pose_{}; 
  Eigen::Matrix<double, 3, 1> desired_xyz;
  Eigen::Matrix<double, 7, 1> tau_null;
  
  std::array<double, 7> tau_max{87, 87, 87, 87, 12, 12, 12};
  std::array<double, 7> tau_min{-87, -87, -87, -87, -12, -12, -12};
  
  
  // q_max_lim nominal: {2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159}
  std::array<double, 7> q_max_lim{2.3, 1.3, 2.5, -0.5, 2.4, 4.1, 2.7};
  
  //q_min_lim nominal: {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159}
  std::array<double, 7> q_min_lim{-2.3, -1.3, -2.5, -2.7, -2.4, 0.9, -2.7};
  
  double safety_dist = 0.2;

  std::array<double, 7> acumulated_tau{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  double max_acumulated_tau = 0.7;
  double acumulated_smooth_factor = 0.0002;
  double dist_thresh_acumulation = 0.5;
  double min_torque_for_acumulation = 0.5;
  double success_dist_thresh = 0.03;
  double dt = 0.001; // 1 [ms]
  double safety_rate = 0.2;
  double d_tau_lim = 1000 * safety_rate;
  

  std::array<double, 7> tau_min_mov_plus{ 0.8,   0.9,  0.9,  1.4,  0.3,  0.65, 0};
  std::array<double, 7> tau_min_mov_minus{-0.8, -0.9, -0.9, -0.9, -0.3, -0.45, 0};
  
  double kv = 2;
  double kp = 3;
  double v_max = 2; 
  
  ros::ServiceServer setpoint_server;
  ros::Publisher result_pub;
  
  bool action_finished = true;
  
 public:
  
  bool init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
    
  
    auto setpoint_handler = [&](franka_control_test::SetPointXYZ::Request& request,
  			      franka_control_test::SetPointXYZ::Response& response) -> bool {

      franka::RobotState robot_state_ = state_handle_->getRobotState();
      Eigen::Affine3d T_EE_in_base_frame(
      Eigen::Matrix4d::Map(robot_state_.O_T_EE.data()));
      Eigen::Vector3d current_xyz(T_EE_in_base_frame.translation());
      
      this->desired_xyz << current_xyz[0] + request.x_delta_ref,
      			    current_xyz[1] + request.y_delta_ref,
      			    current_xyz[2] + request.z_delta_ref;
      
      if (request.tau_null_bool){
        this->tau_null << request.tau_null[0], request.tau_null[1], request.tau_null[2],
      			   request.tau_null[3], request.tau_null[4], request.tau_null[5],
      			   request.tau_null[6];
      } else{
        this->tau_null << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
      }
      
      response.success = 1u;
      response.message = "Setpoint correctly setted";
      
     
      this->elapsed_time_ = ros::Duration(0.0);
      this->action_finished = false;
    
      return true;
  };
  
  setpoint_server = node_handle.advertiseService<franka_control_test::SetPointXYZ::Request, franka_control_test::SetPointXYZ::Response>("setpoint", setpoint_handler);
  result_pub = node_handle.advertise<std_msgs::String>("result", 100);
  
  ROS_INFO("Controlador OSC levantando servidor de setpoints y topico de resultados");
  
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "Error: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  model_interface = robot_hardware->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "Error: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "Error: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "Error: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "Error: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  effort_joint_interface = robot_hardware->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "Error: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "Error: Exception getting joint handles: " << ex.what());
      return false;
    }
  }
  
  return true;
}


  void starting(const ros::Time& /* time */) {
    initial_pose_ = state_handle_->getRobotState().O_T_EE_d;
    elapsed_time_ = ros::Duration(0.0);
    tau_null.setZero();
    desired_xyz << initial_pose_[12], initial_pose_[13], initial_pose_[14];
    ROS_INFO("Posicion inicial: (%f, %f, %f)", desired_xyz[0], desired_xyz[1], desired_xyz[2]);

  }

  void update(const ros::Time& time,
  	      const ros::Duration& period) {
  	      
    franka::RobotState robot_state = state_handle_->getRobotState();
    
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
    Eigen::Affine3d T_EE_in_base_frame(
      Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
      
    Eigen::Vector3d current_xyz(T_EE_in_base_frame.translation());
    Eigen::Quaterniond current_quat(T_EE_in_base_frame.linear());
    
    std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
    std::array<double, 42> jacobian_array =
        model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
    std::array<double, 49> mass_array = model_handle_->getMass();
    
    Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
    Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_(jacobian_array.data());
    Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass_array.data());
    
    Eigen::MatrixXd jacobian_pos(3, 7);
    jacobian_pos << jacobian_.block(0, 0, 3, 7);
    
    Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d_previous(robot_state.tau_J_d.data());
    
    Eigen::MatrixXd M_inv(7, 7);
    M_inv = M.inverse();
    
    Eigen::MatrixXd Mx(3, 3);
    Mx_calculator(M_inv, Mx, jacobian_pos);
    
    Eigen::Matrix<double, 7, 3> J_inv;
    J_inv << M_inv * jacobian_pos.transpose() * Mx;
    
    Eigen::Matrix<double, 7, 7> Null_projection;
    Null_projection << Eigen::MatrixXd::Identity(7, 7) -
                   jacobian_pos.transpose() * J_inv.transpose();

    Eigen::Matrix<double, 7, 1> tau_d, tau_n;
    
    Eigen::Matrix<double, 3, 1> xyz_error, tau_task;
    
    xyz_error << desired_xyz - current_xyz;
    

    

    CartesianVelocityLimiter(xyz_error, tau_task, v_max, kp, kv);
   
    tau_d << - kv * (M * dq);
    
    tau_n << Null_projection * tau_null;
        
    tau_d << jacobian_pos.transpose() * Mx * tau_task + coriolis + tau_d + tau_n;// + Null_projection * tau_null;
    

    std::array<double, 7> tau_d_array{};
    std::array<double, 7> tau_d_previous_array{};
    std::array<double, 7> q_array{};
    
    Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;
    Eigen::VectorXd::Map(&tau_d_previous_array[0], 7) = tau_J_d_previous;
    Eigen::VectorXd::Map(&q_array[0], 7) = q;  
    
    double aux_rate = std::min(xyz_error.norm()/success_dist_thresh, 1.0);
    TorqueMinMovement(tau_d_array, tau_min_mov_plus, tau_min_mov_minus, aux_rate);
    
    TorqueAmplifier(tau_d_array, acumulated_tau, xyz_error.norm(), success_dist_thresh,
                                 dist_thresh_acumulation, min_torque_for_acumulation, acumulated_smooth_factor, 
                                 max_acumulated_tau);
    
    

    JointSafetyLimiter(tau_d_array, q_array, safety_dist, q_max_lim,  q_min_lim);
    
                         
    //  [WARNING]: DONT CHANGE ANYTHING BELOW THIS
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~{SECURITY - DONT CHANGE}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    // -------  LIMITAR D_TORQUE -------  
    DTorqueSafetyLimiter(tau_d_array, tau_d_previous_array, d_tau_lim, dt);
    
    // -------  LIMITAR TORQUE ------- 
    TorqueSafetyLimiter(tau_d_array, tau_min, tau_max);
                      
    // Setear comandos de torque a cada articulacion
    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d_array[i]);
    }
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~{SECURITY - DONT CHANGE}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    if (xyz_error.norm() < success_dist_thresh){
      if (!action_finished){
        action_finished = true;
      
        std_msgs::String msg;
        msg.data = "finished";
        result_pub.publish(msg);
      
      }
    }
  }
};
}// robot_controllers namespace

PLUGINLIB_EXPORT_CLASS(robot_controllers::RobotOSCController,
                       controller_interface::ControllerBase)
