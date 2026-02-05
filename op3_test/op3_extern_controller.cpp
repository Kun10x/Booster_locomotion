#include "op3_webots/op3_extern_controller.hpp"

#include <sensor_msgs/image_encodings.hpp>

#include <webots/Device.hpp>  // keep this include near the top
#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/Camera.hpp>
#include <webots/Motor.hpp>
#include <webots/LED.hpp>
#include <webots/Speaker.hpp>
#include <webots/Camera.hpp>
#include <webots/Gyro.hpp>
#include <webots/Accelerometer.hpp>
#include <webots/InertialUnit.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
namespace robotis_op
{

struct gains
{
  double p_gain;
  double i_gain;
  double d_gain;
  bool initialized;
};

std::string op3_joint_names[20] = {
    "r_sho_pitch", "l_sho_pitch", "r_sho_roll", "l_sho_roll", "r_el", "l_el",
    "r_hip_yaw", "l_hip_yaw", "r_hip_roll", "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch", "r_knee", "l_knee",
    "r_ank_pitch", "l_ank_pitch", "r_ank_roll", "l_ank_roll",
    "head_pan", "head_tilt"};

std::string webots_joint_names[20] = {
    "ShoulderR" /*ID1 */, "ShoulderL" /*ID2 */, "ArmUpperR" /*ID3 */, "ArmUpperL" /*ID4 */, "ArmLowerR" /*ID5 */, "ArmLowerL" /*ID6 */, 
    "PelvYR" /*ID7 */, "PelvYL" /*ID8 */, "PelvR" /*ID9 */, "PelvL" /*ID10*/,
    "LegUpperR" /*ID11*/, "LegUpperL" /*ID12*/, "LegLowerR" /*ID13*/, "LegLowerL" /*ID14*/, 
    "AnkleR" /*ID15*/, "AnkleL" /*ID16*/, "FootR" /*ID17*/, "FootL" /*ID18*/, 
    "Neck" /*ID19*/, "Head" /*ID20*/
};

gains joint_gains[20];

}

using namespace std;
using namespace robotis_op;

OP3ExternController::OP3ExternController() : Node("op3_webots_extern_controller")
{
  time_step_ms_  = 20;
  time_step_sec_ = 0.020;
  camera_publish_period_sec_ = 0.1;  // 10 Hz
  last_camera_publish_time_ = this->get_clock()->now() - rclcpp::Duration::from_seconds(camera_publish_period_sec_);

  current_time_sec_ = 0; // control time

  // for motor angle
  for (int i = 0; i < N_MOTORS; i++)
  {
    desired_joint_angle_rad_[i] = 0;
    current_joint_angle_rad_[i] = 0;
    current_joint_torque_Nm_[i] = 0;
  }  

  // center of mass
  for (int i = 0; i < 3; i++)
  {
    current_com_m_[i]       = 0;
    previous_com_m_[i]      = 0;
    current_com_vel_mps_[i] = 0;
  }  

}

OP3ExternController::~OP3ExternController()
{
  queue_thread_.join();
}

void OP3ExternController::initialize(std::string gain_file_path)
{
  parsePIDGainYAML(gain_file_path);

  // get basic time step 
  time_step_ms_ = getBasicTimeStep();
  time_step_sec_ = time_step_ms_ * 0.001;

  // --- DEBUG: list all devices once ---
  int n = getNumberOfDevices();
  for (int i = 0; i < n; ++i) {
    webots::Device *dev = getDeviceByIndex(i);
    if (!dev) continue;
    RCLCPP_INFO(this->get_logger(), "Webots device %02d: %s", i, dev->getName().c_str());
  }
  
  // initialize webots' devices
  head_led_ = getLED("HeadLed");
  body_led_ = getLED("BodyLed");
  camera_ = getCamera("Camera");
  
  gyro_ = getGyro("Gyro");
  acc_  = getAccelerometer("Accelerometer");
  iu_ = getInertialUnit("inertial unit");

  speaker_ = getSpeaker("Speaker");
  
  gyro_->enable(time_step_ms_);
  acc_->enable(time_step_ms_);
  iu_->enable(time_step_ms_);
  camera_->enable(time_step_ms_);

  // initialize image_data_ and camera_info
  image_data_.header.stamp = this->get_clock()->now();
  image_data_.header.frame_id = "cam_link";
  image_data_.width  = camera_->getWidth();
  image_data_.height = camera_->getHeight();
  image_data_.is_bigendian = false;
  image_data_.step = sizeof(uint8_t) * 4 * camera_->getWidth();
  image_data_.data.resize(4*camera_->getWidth()*camera_->getHeight());
  image_data_.encoding = sensor_msgs::image_encodings::BGRA8;

  camera_info_msg_.header.stamp = this->get_clock()->now();
  camera_info_msg_.header.frame_id = "cam_link";
  camera_info_msg_.width = camera_->getWidth();
  camera_info_msg_.height = camera_->getHeight();
  camera_info_msg_.distortion_model = "plumb_bob"; // need to check what plumb_bob is

  double forcal_length = camera_->getWidth() / (2 * tan(camera_->getFov() * 0.5));
  camera_info_msg_.d = {0.0, 0.0, 0.0, 0.0, 0.0};
  camera_info_msg_.r = {1.0, 0.0, 0.0, 
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0};
  camera_info_msg_.k = {forcal_length, 0.0,           camera_->getWidth() * 0.5,
                        0.0,           forcal_length, camera_->getHeight() * 0.5,
                        0.0,           0.0,           1.0};
  camera_info_msg_.p = {forcal_length,
                        0.0,
                        camera_->getWidth() * 0.5,
                        0.0,
                        0.0,
                        forcal_length,
                        camera_->getHeight() * 0.5,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0};

  // initialize motors
  for (int i = 0; i < N_MOTORS; i++) {
    // get motors
    motors_[i] = getMotor(webots_joint_names[i]);

    // enable torque feedback
    motors_[i]->enableTorqueFeedback(time_step_ms_);

    if (joint_gains[i].initialized == true)
      motors_[i]->setControlPID(joint_gains[i].p_gain, joint_gains[i].i_gain, joint_gains[i].d_gain);

    // initialize encoders
    std::string sensorName = webots_joint_names[i];
    sensorName.push_back('S');
    encoders_[i] = getPositionSensor(sensorName);
    encoders_[i]->enable(time_step_ms_);
  }

  // set publishers and subscribers and spin
  queue_thread_ = std::thread(&OP3ExternController::queueThread, this);
  // Load ONNX Model
  std::string package_share_directory = ament_index_cpp::get_package_share_directory("op3_webots");
  std::string model_path = package_share_directory + "/Op3_policy.onnx";
  session = std::make_unique<Ort::Session>(env, model_path.c_str(), Ort::SessionOptions{nullptr});
  RCLCPP_INFO(this->get_logger(), "received command: %s", model_path.c_str());
  phase_dt_ = 2.0 * M_PI * gait_freq_ * 0.020;
  last_action_.assign(20, 0.0f);
  // default_angles_ = {
  //     // 1. Head (Neck, Head)
  //     0.0f, -0.1745f, 

  //     // 2. Left Arm (ShoulderL, ArmUpperL, ArmLowerL)
  //     -0.0873f, 0.7854f, -0.7854f, 

  //     // 3. Right Arm (ShoulderR, ArmUpperR, ArmLowerR)
  //     0.0873f, -0.7854f, 0.7854f,

  //     // 4. Left Leg (PelvYL, PelvL, LegUpperL, LegLowerL, AnkleL, FootL)
  //     0.0f, -0.0398f, -0.5147f, 1.0694f, 0.6420f, -0.0398f,

  //     // 5. Right Leg (PelvYR, PelvR, LegUpperR, LegLowerR, AnkleR, FootR)
  //     0.0f, 0.0398f, 0.5147f, -1.0694f, -0.6420f, 0.0398f
  // };
  // default_angles_ = {
  //     0.087266,  // ShoulderR
  //   -0.534072,  // ShoulderL
  //   -0.785398,  // ArmUpperR
  //     0.785398,  // ArmUpperL
  //     0.785398,  // ArmLowerR
  //   -0.785398,  // ArmLowerL
  //     0.000000,  // PelvYR
  //     0.000000,  // PelvYL
  //     0.039794,  // PelvR
  //   -0.039794,  // PelvL
  //     0.514697,  // LegUpperR
  //   -0.514697,  // LegUpperL
  //   -1.069363,  // LegLowerR
  //     1.069363,  // LegLowerL
  //   -0.641926,  // AnkleR
  //     0.641926,  // AnkleL
  //     0.039794,  // FootR
  //   -0.039794,  // FootL
  //     0.000000,  // Neck
  //   0.0   // Head
  // };
  default_angles_ = {
      0.534072,   // ShoulderR
    -0.534072,   // ShoulderL
    -0.879648,   // ArmUpperR
      0.879648,   // ArmUpperL
      0.62832,    // ArmLowerR
    -0.62832,    // ArmLowerL
      0.0,        // PelvYR
      0.0,        // PelvYL
      0.0,        // PelvR
      0.0,        // PelvL
      0.596904,   // LegUpperR
    -0.596904,   // LegUpperL
    -1.13098,    // LegLowerR
      1.13098,    // LegLowerL
    -0.534072,   // AnkleR
      0.534072,   // AnkleL
      0.0,        // FootR
      0.0,        // FootL
      0.0,        // Neck
      0.0         // Head
  };
  std::vector<float> zero_obs(49, 0.0f);
  for(int i = 0; i < 3; i++) {
      obs_history_.push_back(zero_obs);
  }
}

void OP3ExternController::process()
{
  // setDesiredJointAngles();


  getPresentJointAngles();
  // getPresentJointTorques();
  // getCurrentRobotCOM();
  getIMUOutput();

  // Create the 49-dim vector for the current frame
  std::vector<float> current_obs = build_observation();
  //frame stacking history of 3  (49*3=147dims)
  obs_history_.push_back(current_obs);
  if (obs_history_.size() > 3) obs_history_.pop_front();
  // --- STEP A: WARMUP / INITIALIZATION ---
  if (warmup_ticks_ < WARMUP_THRESHOLD) {
    for (int i = 0; i < 20; i++) {
        motors_[i]->setPosition(default_angles_[i]);
    }
    warmup_ticks_++;
    stepWebots();
    return; // Exit here! Don't run inference yet.
  }
  


  // Flatten history for ONNX (147 dims)
  std::vector<float> flattened_input;
  for (const auto& frame : obs_history_) {
      flattened_input.insert(flattened_input.end(), frame.begin(), frame.end());
  }
  // 3. INFERENCE
  try {
      Ort::AllocatorWithDefaultOptions allocator;
      auto in_name_ptr = session->GetInputNameAllocated(0, allocator);
      auto out_name_ptr = session->GetOutputNameAllocated(0, allocator);
      const char* in_name = in_name_ptr.get();
      const char* out_name = out_name_ptr.get();

      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<int64_t> input_shape = {1, 147};
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, flattened_input.data(), flattened_input.size(),
          input_shape.data(), input_shape.size());

      auto output_tensors = session->Run(Ort::RunOptions{nullptr}, &in_name, &input_tensor, 1, &out_name, 1);
      float* output_arr = output_tensors[0].GetTensorMutableData<float>();

      // 4. APPLY TO MOTORS & STORE LAST ACTION
      int map_mj_to_wb[20] = {
          18, 19,                     // Head
          1, 3, 5,                    // L Arm
          0, 2, 4,                    // R Arm
          7, 9, 11, 13, 15, 17,       // L Leg
          6, 8, 10, 12, 14, 16        // R Leg
      };
      if (std::isnan(output_arr[0])) {
          RCLCPP_WARN(this->get_logger(), "Policy returned NaN! Skipping motor update.");
      } else {
        
          for (int i = 0; i < 20; i++) {
            int wb_idx = map_mj_to_wb[i];
            last_action_[i] = output_arr[i]; 
            // Control = Action * Scale + Default
            double target_pos = (double)output_arr[wb_idx] * 0.3 + default_angles_[i];
            motors_[i]->setPosition(target_pos);
          }
      }

  } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX Error: %s", e.what());
  }

  // {
  //   auto logger = this->get_logger();
  //   RCLCPP_INFO(logger, "---------------------------------------------------");
  //   RCLCPP_INFO(logger, "FRAME DEBUG [Time Step: %d ms]", time_step_ms_);

  //   // 1. IMU Data
    
  //   RCLCPP_INFO(logger, "GYRO  [x,y,z]: [%.3f, %.3f, %.3f]", imu_data_.angular_velocity.x, imu_data_.angular_velocity.y, imu_data_.angular_velocity.z);
  //   RCLCPP_INFO(logger, "ACCEL [x,y,z]: [%.3f, %.3f, %.3f]", imu_data_.linear_acceleration.x, imu_data_.linear_acceleration.y, imu_data_.linear_acceleration.z);
    

  //   // 2. Commands
  //   RCLCPP_INFO(logger, "CMD   [x,y,yaw]: [%.2f, %.2f, %.2f]", com_m_.x, com_m_.y, com_m_.z);

  //   // // 3. Joint Angles (Relative to Default)
  //   // // Printing in rows of 5 for readability
  //   // RCLCPP_INFO(logger, "JOINT RELATIVE POSITIONS (Current - Default):");
  //   // for (int i = 0; i < 20; i += 5) {
  //   //     RCLCPP_INFO(logger, "  [%02d-%02d]: %6.3f %6.3f %6.3f %6.3f %6.3f",
  //   //         i, i+4,
  //   //         (float)encoders_[i]->getValue() - default_angles_[i],
  //   //         (float)encoders_[i+1]->getValue() - default_angles_[i+1],
  //   //         (float)encoders_[i+2]->getValue() - default_angles_[i+2],
  //   //         (float)encoders_[i+3]->getValue() - default_angles_[i+3],
  //   //         (float)encoders_[i+4]->getValue() - default_angles_[i+4]);
  //   // }

  //   // // 4. Last Actions (Policy Raw Outputs)
  //   // RCLCPP_INFO(logger, "LAST ACTIONS (Policy Outputs -1 to 1):");
  //   // for (int i = 0; i < 20; i += 5) {
  //   //     RCLCPP_INFO(logger, "  [%02d-%02d]: %6.3f %6.3f %6.3f %6.3f %6.3f",
  //   //         i, i+4,
  //   //         last_action_[i], last_action_[i+1], last_action_[i+2], 
  //   //         last_action_[i+3], last_action_[i+4]);
  //   // }
  //   RCLCPP_INFO(logger, "---------------------------------------------------");
  // }

  publishPresentJointStates();
  // publishIMUOutput();
  // publishCOMData();
  publishCameraData();

  stepWebots();
}
std::vector<float> OP3ExternController::build_observation() {
    std::vector<float> obs;

    // 1. Gyro (3)
    obs.push_back((float)imu_data_.angular_velocity.x);
    obs.push_back((float)imu_data_.angular_velocity.y);
    obs.push_back((float)imu_data_.angular_velocity.z);

    // 2. Gravity Vector (3) - Using the orientation quaternion
    float qx = (float)imu_data_.orientation.x;
    float qy = (float)imu_data_.orientation.y;
    float qz = (float)imu_data_.orientation.z;
    float qw = (float)imu_data_.orientation.w;

    obs.push_back(2 * (qx * qz - qw * qy));
    obs.push_back(2 * (qw * qx + qy * qz));
    obs.push_back(qw * qw - qx * qx - qy * qy + qz * qz);

    // 3. Command (3)
    obs.push_back(cmd_x_);
    obs.push_back(cmd_y_);
    obs.push_back(cmd_yaw_);

    int map_mj_to_wb[20] = {
        18, 19,                     // Head
        1, 3, 5,                    // L Arm
        0, 2, 4,                    // R Arm
        7, 9, 11, 13, 15, 17,       // L Leg
        6, 8, 10, 12, 14, 16        // R Leg
    };

    // 4. Joint Angles - RELATIVE (20)
    for (int i = 0; i < 20; i++) {
        int wb_idx = map_mj_to_wb[i];
        float relative_angle = (float)current_joint_angle_rad_[wb_idx] - default_angles_[i];
        obs.push_back(relative_angle);
    }

    // 5. Last Action (20)
    for (int i = 0; i < 20; i++) {
        obs.push_back(last_action_[i]);
    }

    return obs; // Exactly 49 dimensions
}
void OP3ExternController::setDesiredJointAngles()
{
  for (int joint_idx = 0; joint_idx < N_MOTORS; joint_idx++)
  {
    motors_[joint_idx]->setPosition(desired_joint_angle_rad_[joint_idx]);
  }
}

void OP3ExternController::getPresentJointAngles()
{
  for (int joint_idx = 0; joint_idx < N_MOTORS; joint_idx++)
  {
    current_joint_angle_rad_[joint_idx] = encoders_[joint_idx]->getValue();
  }
}

void OP3ExternController::getPresentJointTorques()
{
  for (int joint_idx = 0; joint_idx < N_MOTORS; joint_idx++)
  {
    current_joint_torque_Nm_[joint_idx] = motors_[joint_idx]->getTorqueFeedback();
  }
}

void OP3ExternController::getCurrentRobotCOM()
{
  const double* com = this->getSelf()->getCenterOfMass();

  previous_com_m_[0] = current_com_m_[0];
  previous_com_m_[1] = current_com_m_[1];
  previous_com_m_[2] = current_com_m_[2];
  
  current_com_m_[0] = com[0];
  current_com_m_[1] = com[1];
  current_com_m_[2] = com[2];

  current_com_vel_mps_[0] = (current_com_m_[0] - previous_com_m_[0]) / time_step_sec_;
  current_com_vel_mps_[1] = (current_com_m_[1] - previous_com_m_[1]) / time_step_sec_;
  current_com_vel_mps_[2] = (current_com_m_[2] - previous_com_m_[2]) / time_step_sec_;

  com_m_.x = current_com_m_[0];
  com_m_.y = current_com_m_[1];
  com_m_.z = current_com_m_[2];
}

void OP3ExternController::getIMUOutput()
{
  const double* gyro_rps = gyro_->getValues();
  const double* acc_mps2 = acc_->getValues();
  const double* quat = iu_->getQuaternion();

  imu_data_.angular_velocity.x = gyro_rps[0];
  imu_data_.angular_velocity.y = gyro_rps[1];
  imu_data_.angular_velocity.z = gyro_rps[2];

  imu_data_.linear_acceleration.x = acc_mps2[0];
  imu_data_.linear_acceleration.y = acc_mps2[1];
  imu_data_.linear_acceleration.z = acc_mps2[2];

  imu_data_.orientation.x = quat[0];
  imu_data_.orientation.y = quat[1];
  imu_data_.orientation.z = quat[2];
  imu_data_.orientation.w = quat[3];
}

void OP3ExternController::publishPresentJointStates()
{
  joint_state_msg_.name.clear();
  joint_state_msg_.position.clear();
  joint_state_msg_.velocity.clear();
  joint_state_msg_.effort.clear();

  joint_state_msg_.header.stamp = rclcpp::Clock().now();
    
  for(int i = 0; i < 20; i++)
  {
    joint_state_msg_.name.push_back(op3_joint_names[i]);
    joint_state_msg_.position.push_back(current_joint_angle_rad_[i]);
    joint_state_msg_.velocity.push_back(0);
    joint_state_msg_.effort.push_back(current_joint_torque_Nm_[i]);
  }

  present_joint_state_publisher_->publish(joint_state_msg_);
}

void OP3ExternController::publishIMUOutput()
{
  imu_data_publisher_->publish(imu_data_);
}

void OP3ExternController::publishCOMData()
{
  com_data_publisher_->publish(com_m_);
}

void OP3ExternController::publishCameraData()
{
  const auto now = this->get_clock()->now();
  if ((now - last_camera_publish_time_) < rclcpp::Duration::from_seconds(camera_publish_period_sec_)) {
    return;
  }
  last_camera_publish_time_ = now;

  image_data_.header.stamp = now;
  camera_info_msg_.header.stamp = image_data_.header.stamp;
  
  if (camera_info_publisher_->get_subscription_count() > 0)
    camera_info_publisher_->publish(camera_info_msg_);

  auto image = camera_->getImage();
    
  if (image)
  {
    memcpy(image_data_.data.data(), image, image_data_.data.size());
    camera_image_publisher_->publish(image_data_);
  }
}

void OP3ExternController::queueThread()
{
  auto executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor->add_node(this->get_node_base_interface());


  /* Publishers, Subsribers, and Service Clients */
  present_joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/robotis_op3/joint_states", 1);
  imu_data_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("/robotis_op3/imu", 1);
  com_data_publisher_ = this->create_publisher<geometry_msgs::msg::Vector3>("/robotis_op3/com", 1);

  camera_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera/camera_info", 1);
  camera_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", rclcpp::SensorDataQoS().reliable());

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr goal_pos_subs[N_MOTORS];

  for (int i = 0; i < N_MOTORS; i++)
  {
    // make subscribers for the joint position topic from robotis framework
    std::string goal_pos_topic_name = "/robotis_op3/" + op3_joint_names[i] + "_position/command";

    std::function<void(const std_msgs::msg::Float64::SharedPtr)> callback = 
         std::bind(&OP3ExternController::posCommandCallback, this, std::placeholders::_1, i);
    goal_pos_subs[i] = this->create_subscription<std_msgs::msg::Float64>(goal_pos_topic_name, 1, callback);
  }

  rclcpp::Rate rate(1000.0 / 20.0);
  while (rclcpp::ok())
  {
    executor->spin_some();
    //this->publishCameraData();
    rate.sleep();
  }
}

void OP3ExternController::posCommandCallback(const std_msgs::msg::Float64::SharedPtr msg, const int &joint_idx)
{
  desired_joint_angle_rad_[joint_idx] = msg->data;
  desired_joint_angle_rcv_flag_[joint_idx] = true;
}

void OP3ExternController::stepWebots() {
  if (step(time_step_ms_) == -1)
    exit(EXIT_SUCCESS);
}

bool OP3ExternController::parsePIDGainYAML(std::string gain_file_path)
{
  if (gain_file_path == "")
    return false;

  YAML::Node doc;

  try
  {
    doc = YAML::LoadFile(gain_file_path.c_str());
    for (int joint_idx = 0; joint_idx < N_MOTORS; joint_idx++)
    {
      joint_gains[joint_idx].initialized = false;
      YAML::Node gains;
      if (gains = doc[webots_joint_names[joint_idx]])
      {
        joint_gains[joint_idx].p_gain = gains["p_gain"].as<double>();
        joint_gains[joint_idx].i_gain = gains["i_gain"].as<double>();
        joint_gains[joint_idx].d_gain = gains["d_gain"].as<double>();
        joint_gains[joint_idx].initialized = true;
      }
      else
      {
        RCLCPP_WARN_STREAM(this->get_logger(), "there is not pre-defined gains for " << webots_joint_names[joint_idx]);
      }
    }

    return true;
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_STREAM(this->get_logger(), "gain file not found: " << gain_file_path);
    return false;
  }
}
