#ifndef OP3_WEBOTS_ROS2_OP3_EXTERN_CONTROLLER_HPP
#define OP3_WEBOTS_ROS2_OP3_EXTERN_CONTROLLER_HPP

#include <rclcpp/rclcpp.hpp>

#include <thread>

#include <std_msgs/msg/float64.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <webots/Supervisor.hpp>

#include <yaml-cpp/yaml.h>

#include <onnxruntime_cxx_api.h>

#define N_MOTORS (20)

namespace robotis_op
{

class OP3ExternController : public webots::Supervisor, public rclcpp::Node
{
private:
// ONNX Variables
  Ort::Env env;
  std::unique_ptr<Ort::Session> session;
  std::vector<const char*> input_names = {"input"}; // Match your model's input name
  std::vector<const char*> output_names = {"output"}; 
  // Constants - Adjust these based on your training!
  const float ACTION_SCALING = 0.5; // Example: if policy outputs -1 to 1 and range is 0.5 rad
  const int SINGLE_OBS_DIM = 49;           // Ensure this matches your model input size
  const int HISTORY_LEN = 3;
  // Buffer for observations (e.g., 49 values)
  std::deque<std::vector<float>> obs_history_;
  std::vector<float> input_tensor_values;
  std::vector<int64_t> input_shape = {1, SINGLE_OBS_DIM};

  // In op3_extern_controller.hpp private members:
  double phase_[2] = {0.0, M_PI}; // Phase oscillators
  double gait_freq_ = 1.5;
  double phase_dt_; // Will be calculated in initialize
  std::vector<float> default_angles_;
  std::vector<float> last_action_;
  float cmd_x_ = 0.0;   // Example: Walking forward command
  float cmd_y_ = 0.0;
  float cmd_yaw_ = 0.0;
  int warmup_ticks_ = 0;
  const int WARMUP_THRESHOLD = 500; // 50 ticks * 20ms = 1 second
  std::vector<float> build_observation();

public:
  OP3ExternController();
  virtual ~OP3ExternController();

  void run();

  void initialize(std::string gain_file_path);

  void process();

  void stepWebots();

  void setDesiredJointAngles();
  void getPresentJointAngles();
  void getPresentJointTorques();

  void getCameraInfo();
  
  void getCurrentRobotCOM();
  void getIMUOutput();

  void publishPresentJointStates();
  void publishIMUOutput();
  void publishCOMData();
  void publishCameraData();

  void posCommandCallback(const std_msgs::msg::Float64::SharedPtr msg, const int &joint_idx);
  
  void queueThread();
  
  bool parsePIDGainYAML(std::string gain_file_path);


  int time_step_ms_;
  double time_step_sec_;

  double current_time_sec_; // control time

  // for motor angle
  double desired_joint_angle_rad_[N_MOTORS];
  double current_joint_angle_rad_[N_MOTORS];
  double current_joint_torque_Nm_[N_MOTORS];
  sensor_msgs::msg::JointState joint_state_msg_;

  // center of mass
  double current_com_m_[3];
  double previous_com_m_[3];
  double current_com_vel_mps_[3];
  geometry_msgs::msg::Vector3 com_m_;
  
  // imu
  double torso_xyz_m_[3];
  sensor_msgs::msg::Imu imu_data_;

  // camera
  sensor_msgs::msg::CameraInfo camera_info_msg_;
  sensor_msgs::msg::Image image_data_;
  rclcpp::Time last_camera_publish_time_;
  double camera_publish_period_sec_;

  // publishers
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr com_data_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr present_joint_state_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_data_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_image_publisher_;
  
  // devices
  webots::Camera* camera_;
  webots::LED* head_led_;
  webots::LED* body_led_;
  webots::Speaker* speaker_;
  webots::Keyboard* key_board_; 
  
  webots::Gyro *gyro_;
  webots::Accelerometer *acc_;
  webots::InertialUnit *iu_;

  webots::Motor* motors_[N_MOTORS];
  webots::PositionSensor* encoders_[N_MOTORS];
  
  webots::Node *torso_node_;
  webots::Node *rf_node_, *lf_node_;

  bool desired_joint_angle_rcv_flag_[20];
  
  // thread
  std::thread queue_thread_;
};

}

#endif
