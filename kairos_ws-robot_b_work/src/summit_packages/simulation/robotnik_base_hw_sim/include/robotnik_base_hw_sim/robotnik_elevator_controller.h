#include <boost/circular_buffer.hpp>

#include <controller_interface/controller.h>
#include <controller_interface/controller_base.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <robotnik_msgs/SetElevator.h>
#include <robotnik_msgs/ElevatorStatus.h>
#include <robotnik_msgs/ElevatorAction.h>
#include <robotnik_msgs/enable_disable.h>
#include <robotnik_msgs/SetElevatorAction.h>
#include <robotnik_msgs/inputs_outputs.h>
#include <robotnik_msgs/set_digital_output.h>
#include <robotnik_base_hw_sim/SimplePick.h>
#include <robotnik_base_hw_sim/SimplePlace.h>


#include <actionlib/server/simple_action_server.h>

#include <string>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>

using namespace std;

namespace robotnik_elevator_controller
{


class RobotnikElevatorController : public controller_interface::Controller<hardware_interface::EffortJointInterface>

{
public:
  RobotnikElevatorController();

  /**
  */

  bool init(hardware_interface::EffortJointInterface* hw, ros::NodeHandle& root_nh, ros::NodeHandle& controller_nh);

  /**
   * \brief Updates controller, i.e. computes the odometry and sets the new velocity commands
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
  // Services

  // Control stuff
  hardware_interface::JointHandle joint_;  // joint handles: to read current state and send commands
  std::vector<std::string> joint_names_;                 // joint names: to get the handle of each joint

  std::vector<std::pair<double, double> > joint_limits_;  // lower, upper limits

  double joint_states_;       // current joint state: position or velocity
                                           // constraints
  std::vector<double> joint_commands_;  // current command to be sent: may differ from reference is the wheels are not
                                        // in position or if the watchdog times out

  robotnik_msgs::inputs_outputs received_io_;  // holds last io msg
  ros::Time io_last_stamp_;                    // holds the last io msg stamp

 
  // ROS stuff
  std::string controller_name_;   // node name,
  
  std::string set_digital_output_service_hw_;  // name of the topic where the odometry is published
  // Publishers
  ros::Publisher elevator_status_pub_;         // topic publisher for the elevator
  ros::Publisher enabled_pub_;            		// topic publisher to publish if the controller is enabled
  ros::Publisher elevator_pub_;
  // Subscribers
  ros::Subscriber io_sub_;       // I/O subscriber
 // Service Clients
  ros::ServiceClient set_digital_output_client;
  ros::ServiceClient gazebo_simple_pick_service_client;
  ros::ServiceClient gazebo_simple_place_service_client;
  // Service Servers
  ros::ServiceServer set_elevator_srv_;
  ros::ServiceServer enable_srv_;

  bool setElevatorSrvCallback(robotnik_msgs::SetElevator::Request& req, robotnik_msgs::SetElevator::Response& res);
  bool enableSrvCallback(robotnik_msgs::enable_disable::Request& req, robotnik_msgs::enable_disable::Response& res);

  // Elevator
  // Strnig param elevator joint
  std::string elevator_joint_name_;
  ros::Time last_elevator_action_time;  // Time of the last elevator action
  
  robotnik_msgs::ElevatorAction elevator_action_, elevator_ongoing_action_;  // saves the action to carry out
  robotnik_msgs::ElevatorStatus elevator_status_;      // saves the elevator status for publishing purposes
  ros::Time elevator_loop_last_execution_;             // to check if the elevator status must be sent
  ros::Time elevator_init_action_time_;                // Time when the action
  ros::Duration elevator_status_control_loop_period_;  // control loop period
  ros::Duration elevator_action_command_timeout_;      // timeout processing an action received
  int elevator_digital_output_up_;
  int elevator_digital_output_down_;
  int elevator_digital_input_up_;
  int elevator_digital_input_down_;
  ros::Time elevator_action_init_time;  // Time when the action starts
  double elevator_position_up_, elevator_position_down_;
  double elevator_current_position_;  // The position to publish
  bool enabled_; 						//Flag to accept any type of command
  bool init_ok; // flag true after init
  double elevation_action_time_;	// time needed to perform the any elevation action
  
  // GAZEBO FAKE ELEVATOR PARAMS
  bool disable_gazebo_physics_for_pickup_;	//
  std::string gazebo_simple_pick_service_name_;
  std::string gazebo_simple_place_service_name_;
  std::string gazebo_robot_model_;
  geometry_msgs::Pose elevation_offset;
  
  void readJointStates();
  void writeJointCommands();
  void publishElevatorStatus();
  void elevatorControlLoop(const ros::Time& time);
  void switchToElevatorState(string new_state);

  void ioCallback(const robotnik_msgs::inputs_outputsConstPtr& msg);
  int setDigitalOutput(int number, bool value);
  int pickCart();
  int placeCart();

  actionlib::SimpleActionServer<robotnik_msgs::SetElevatorAction>* elevator_action_server_;
  robotnik_msgs::SetElevatorFeedback elevator_action_feedback_;
  robotnik_msgs::SetElevatorResult elevator_action_result_;

  void executeElevatorCallback(const robotnik_msgs::SetElevatorGoalConstPtr& goal);
};
PLUGINLIB_EXPORT_CLASS(robotnik_elevator_controller::RobotnikElevatorController, controller_interface::ControllerBase);
}
