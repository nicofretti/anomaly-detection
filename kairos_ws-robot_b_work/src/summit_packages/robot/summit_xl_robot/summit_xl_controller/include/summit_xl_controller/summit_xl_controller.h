#include <boost/circular_buffer.hpp>

#include <controller_interface/controller.h>
#include <controller_interface/controller_base.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>

#include <std_msgs/Bool.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <tf/tfMessage.h>
#include <tf/transform_broadcaster.h>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <robotnik_msgs/enable_disable.h>
#include <robotnik_msgs/get_mode.h>
#include <robotnik_msgs/set_mode.h>
#include <robotnik_msgs/set_odometry.h>

#include <std_msgs/Bool.h>

namespace summit_xl_controller
{
enum
{
  FRONT_RIGHT_TRACTION_JOINT = 0,
  FRONT_LEFT_TRACTION_JOINT = 1,
  BACK_RIGHT_TRACTION_JOINT = 2,
  BACK_LEFT_TRACTION_JOINT = 3,
  BEGIN_TRACTION_JOINT = 0,
  END_TRACTION_JOINT = 4,
  NUMBER_OF_JOINTS = 4,
  NUMBER_OF_WHEELS = 4
};

class SummitXLController : public controller_interface::Controller<hardware_interface::VelocityJointInterface>

/* The standard way on indigo is to inherit from
controller_interface::Controller<T>,
where T is a JointInterface. So a controller can access to only one type of
JointInterface

As a hack, if we inherit from ControllerBase instead, we can access to two
different JointInterfaces
In that case, we need to implement:
    initRequest, which receives a RobotHW instead of a
hardware_interface::XJointInterface, from where the interfaces can be accessed
    getHardwareInterfaceType, which returns a string with the main type of
JointInterface. In our case, it is a VelocityJointInterface
*/

{
public:
  SummitXLController();

  /**
  */

  bool init(hardware_interface::VelocityJointInterface* hw, ros::NodeHandle& root_nh, ros::NodeHandle& controller_nh);

  /**
   * \brief Updates controller, i.e. computes the odometry and sets the new
   * velocity commands
   * \param time   Current time
   * \param period Time since the last called to update
   */
  void update(const ros::Time& time, const ros::Duration& period);

  /**
   * \brief Starts controller
   * \param time Current time
   */
  void starting(const ros::Time& time);

  /**
   * \brief Stops controller
   * \param time Current time
   */
  void stopping(const ros::Time& /*time*/);

  virtual std::string getHardwareInterfaceType() const;

private:
  // Control stuff
  // joint handles: to read current state and send commands
  std::vector<hardware_interface::JointHandle> joints_;
  // joint names: to get the handle of each joint
  std::vector<std::string> joint_names_;

  std::vector<std::pair<double, double>> joint_limits_;  // lower, upper limits
  // current joint state: position or velocity
  std::vector<double> joint_states_;
  std::vector<double> joint_states_mean_;  // current joint state mean: used to
                                           // calculate the reference according
                                           // to the constraints
  std::vector<double> joint_references_;   // current reference for each joint
  std::vector<double> joint_commands_;     // current command to be sent: may differ
                                           // from reference is the wheels are not
                                           // in position or if the watchdog times
                                           // out
  // used to calculate the current joint state as the
  // mean of the previous joint states
  std::vector<boost::circular_buffer<double>> joint_states_history_;

  unsigned int joint_states_history_size_;  // size of the joint history

  // Data
  geometry_msgs::Twist received_cmd_;  // holds last velocity command
  geometry_msgs::Twist current_cmd_;   // hold current used command (limited). it
                                       // is updated on the limitCommand function
                                       // and in the writeJointCommands, because
                                       // if the direction wheels are not in
                                       // position, the traction reference is 0
  nav_msgs::Odometry odom_;            // holds odometry
  geometry_msgs::Pose2D robot_pose_;   // holds robot position
  ros::Time cmd_last_stamp_;           // holds last velocity command time stamp, used to
                                       // check the watchdog
  bool cmd_watchdog_timedout_;         // true is the watchdog has been activated
                                       // time that has to pass to activate the watchdog
  ros::Duration cmd_watchdog_duration_;

  ros::Time imu_last_stamp_;    // holds last velocity command time stamp, used to
                                // check the watchdog
  bool imu_watchdog_timedout_;  // true is the watchdog has been activated
  // time that has to pass to activate the watchdog
  ros::Duration imu_watchdog_duration_;
  double in_motion_timer_;

  bool odom_broadcast_tf_;  // if true, odom must be published also in tf
  // time between odometry publication updates
  ros::Duration odom_publish_period_;
  ros::Time odom_last_sent_;    // to check if the odometry must be sent
  ros::Time odom_last_update_;  // to use in the odometry calculation

  // Wheels configuration
  double wheel_base_;      // distance between front and rear axles
  double track_width_;     // distance between right and left wheels
  double wheel_diameter_;  // wheel diamater, to convert from angular speed to
                           // linear

  // Speed and acceleration limits
  double linear_speed_limit_;
  double linear_acceleration_limit_;
  double angular_speed_limit_;
  double angular_acceleration_limit_;

  bool hard_brake_;
  bool e_stop_;

  int active_kinematic_mode_;
  bool motion_odometry_;
  std::string kinematic_mode_;  // "skid"=only skid or "omni"=permits both

  std::string imu_topic_;  // name of the topic which is publishing
                           // sensor_msgs/Imu messages
  std::string e_stop_topic_;  // name of the topic which is publishing emergency stop
  bool bFirstYawRead_;
  bool bYawSensor_;
  double init_yaw_;    // Init yaw from the yaw sensor (gyro or imu)
  double last_yaw_;    // Last yaw read from the yaw sensor (gyro or imu)
  double delta_yaw_;   // Yaw increment in one period
  double motion_yaw_;  // Variable that integrates yaw only when the robot is in
                       // motion
  double imu_yaw_;

  bool enabled_;

  ros::Time e_stop_last_stamp_;		   // Saves the time when the estop is active
  // ROS stuff
  std::string controller_name_;             // node name,
  std::string command_topic_;               // topic from where velocity commands are read
  std::string odom_topic_;                  // name of the topic where the odometry is published
  std::string odom_frame_;                  // name of the frame associated to the odometry
  std::string robot_base_frame_;            // name of the frame associated to the robot.
                                            // odom_frame_ is published as it's parent
  std::string front_right_traction_joint_;  // Name of the wheel joints
  std::string front_left_traction_joint_;
  std::string back_right_traction_joint_;
  std::string back_left_traction_joint_;

  double front_right_traction_joint_max_speed_;  // Name of the wheel joints
  double front_left_traction_joint_max_speed_;
  double back_right_traction_joint_max_speed_;
  double back_left_traction_joint_max_speed_;

  double joint_speed_noise_threshold_;  // Min. joint speed to discard noise from motors

  // topic publisher where the odometry is published
  ros::Publisher odom_publisher_;
  // topic publisher to publish if the controller is enabled
  ros::Publisher enabled_pub_;
  ros::Publisher in_motion_publisher_;
  // topic subscriber to receive velocity commands
  ros::Subscriber cmd_vel_subscriber_;
  tf::TransformBroadcaster* transform_broadcaster_;  // to publish odom frame

  ros::Subscriber imu_sub_;
  ros::Subscriber e_stop_sub_;

  // Services
  ros::ServiceServer srv_SetOdometry_;
  ros::ServiceServer srv_GetMode_;
  ros::ServiceServer enable_srv_;

  //
  void readJointStates();
  void writeJointCommands();
  void limitCommand(double period);
  void updateJointStateHistoryMean();
  void updateJointReferences();
  void setJointVelocityReferenceBetweenLimits(std::vector<double>& wheel_speed);
  void updateOdometry();
  bool InMotion();
  void publishOdometry();

  void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& cmd_msg);
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);
  void eStopCallback(const std_msgs::BoolConstPtr& msg);

  bool srvCallback_GetMode(robotnik_msgs::get_mode::Request& request, robotnik_msgs::get_mode::Response& response);
  bool srvCallback_SetOdometry(robotnik_msgs::set_odometry::Request& request,
                               robotnik_msgs::set_odometry::Response& response);
  bool enableSrvCallback(robotnik_msgs::enable_disable::Request& req, robotnik_msgs::enable_disable::Response& res);
};
PLUGINLIB_EXPORT_CLASS(summit_xl_controller::SummitXLController, controller_interface::ControllerBase);
}
