#include <boost/circular_buffer.hpp>
#include <math.h>

#include <controller_interface/multi_interface_controller.h>
#include <controller_interface/controller.h>
#include <controller_interface/controller_base.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose2D.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/tfMessage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>

#include <robotnik_msgs/set_mode.h>
#include <robotnik_msgs/get_mode.h>
#include <robotnik_msgs/set_odometry.h>
#include <robotnik_msgs/set_height.h>


namespace summit_x_controller
{
    
    enum {
        FRONT_RIGHT_TRACTION_JOINT=1,
        FRONT_LEFT_TRACTION_JOINT=2,
        BACK_RIGHT_TRACTION_JOINT=3,
        BACK_LEFT_TRACTION_JOINT=4,
        BEGIN_TRACTION_JOINT=1,
        END_TRACTION_JOINT=5,
        
        BEGIN_POSITION_JOINT=0,
        END_POSITION_JOINT=1,
        UPPER_PLATTFORM_BASE_JOINT=0,
        
        NUMBER_OF_JOINTS=5,
    };

    class SummitXController 
    : public controller_interface::MultiInterfaceController<hardware_interface::VelocityJointInterface,
                                                            hardware_interface::PositionJointInterface>
    
    /* The standard way on indigo is to inherit from controller_interface::Controller<T>,
    where T is a JointInterface. So a controller can access to only one type of JointInterface

    As a hack, if we inherit from ControllerBase instead, we can access to two different JointInterfaces
    In that case, we need to implement:
        initRequest, which receives a RobotHW instead of a hardware_interface::XJointInterface, from where the interfaces can be accessed
        getHardwareInterfaceType, which returns a string with the main type of JointInterface. In our case, it is a VelocityJointInterface
    */

{
    public:
    SummitXController();

    
    /**
    */ 

    bool init(hardware_interface::RobotHW* const robot_hw, ros::NodeHandle& root_nh, ros::NodeHandle& controller_nh);

    bool initController(ros::NodeHandle root_nh, ros::NodeHandle controller_nh);

    bool initVelocityInterface(hardware_interface::VelocityJointInterface* hw, ros::NodeHandle& root_nh,
                                    ros::NodeHandle& controller_nh);

    bool initPositionInterface(hardware_interface::PositionJointInterface* hw, ros::NodeHandle& root_nh,
                                    ros::NodeHandle& controller_nh);

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

    // Control stuff
    std::vector<hardware_interface::JointHandle> joints_; // joint handles: to read current state and send commands
    std::vector<std::string> joint_names_; // joint names: to get the handle of each joint

    std::vector<std::pair<double, double> > joint_limits_; //lower, upper limits
   
	std::vector<double> joint_states_; // current joint state: position or velocity
    std::vector<double> joint_states_mean_; // current joint state mean: used to calculate the reference according to the constraints
    std::vector<double> joint_references_; // current reference for each joint
	std::vector<double> joint_commands_; // current command to be sent: may differ from reference is the wheels are not in position or if the watchdog times out

    std::vector<boost::circular_buffer<double> > joint_states_history_; // used to calculate the current joint state as the mean of the previous joint states
    unsigned int joint_states_history_size_; // size of the joint history
    
    // Data
    geometry_msgs::Twist received_cmd_; // holds last velocity command
    geometry_msgs::Twist current_cmd_;  // hold current used command (limited). it is updated on the limitCommand function and in the writeJointCommands, because if the direction wheels are not in position, the traction reference is 0 
    double scissor_cmd_;
    nav_msgs::Odometry odom_; // holds odometry
    geometry_msgs::Pose2D robot_pose_; // holds robot position
    ros::Time cmd_last_stamp_; // holds last velocity command time stamp, used to check the watchdog
    bool cmd_watchdog_timedout_; // true is the watchdog has been activated
    ros::Duration cmd_watchdog_duration_; // time that has to pass to activate the watchdog
    
    ros::Time imu_last_stamp_; // holds last velocity command time stamp, used to check the watchdog
    bool imu_watchdog_timedout_; // true is the watchdog has been activated
    ros::Duration imu_watchdog_duration_; // time that has to pass to activate the watchdog
    double in_motion_timer_;

    bool odom_broadcast_tf_; // if true, odom must be published also in tf
    ros::Duration odom_publish_period_; // time between odometry publication updates
    ros::Time odom_last_sent_; // to check if the odometry must be sent
    ros::Time odom_last_update_; // to use in the odometry calculation 

    // Wheels configuration
    double wheel_base_;     // distance between front and rear axles 
    double track_width_;    // distance between right and left wheels 
    double wheel_diameter_; // wheel diamater, to convert from angular speed to linear 

    // Speed and acceleration limits
    double linear_speed_limit_;
    double linear_acceleration_limit_;
    double angular_speed_limit_;
    double angular_acceleration_limit_;

	int active_kinematic_mode_;
	bool motion_odometry_;
    std::string kinematic_mode_;  	// "skid"=only skid or "omni"=permits both
	
	std::string imu_topic_;			// name of the topic which is publishing sensor_msgs/Imu messages
	bool bFirstYawRead_;
	bool bYawSensor_;		
	double init_yaw_;  		// Init yaw from the yaw sensor (gyro or imu)
	double last_yaw_; 		// Last yaw read from the yaw sensor (gyro or imu)
	double delta_yaw_; 		// Yaw increment in one period
	double motion_yaw_; 	// Variable that integrates yaw only when the robot is in motion
	double imu_yaw_;
	
	

    // ROS stuff
    std::string controller_name_; // node name, 
    std::string command_topic_; // topic from where velocity commands are read 
    std::string odom_topic_; // name of the topic where the odometry is published 
    std::string odom_frame_; // name of the frame associated to the odometry 
    std::string robot_base_frame_; // name of the frame associated to the robot. odom_frame_ is published as it's parent 
	std::string front_right_traction_joint_;	// Name of the wheel joints
	std::string front_left_traction_joint_;
	std::string back_right_traction_joint_;
	std::string back_left_traction_joint_;
	std::string upper_base_joint_;
	std::string upper_platform_base_joint_;
	
    sensor_msgs::JointState joint_states_msg_;
	
    double upper_platform_base_current_position_;
	double front_right_traction_joint_max_speed_;	// Name of the wheel joints
	double front_left_traction_joint_max_speed_;
	double back_right_traction_joint_max_speed_;
	double back_left_traction_joint_max_speed_;
	double aux_upper_base_joint_max_position_;
	double upper_base_joint_max_position_;
    double linear_axis_relation_;
    double scissor_hypotenuse_;
    double scissor_default_height_;
    double scissor_default_linear_axis_;
	

    ros::Publisher odom_publisher_; // topic publisher where the odometry is published 
    ros::Publisher joint_states_pub_; 
    ros::Subscriber cmd_vel_subscriber_; // topic subscriber to receive velocity commands 
    tf::TransformBroadcaster *transform_broadcaster_; // to publish odom frame
    
    ros::Subscriber imu_sub_;

	// Services
	ros::ServiceServer srv_SetOdometry_;
	ros::ServiceServer srv_SetMode_;
	ros::ServiceServer srv_GetMode_;
    ros::ServiceServer srv_SetPosition_;

    //
    void readJointStates();
    void writeJointCommands();
    void limitCommand(double period);
    void updateJointStateHistoryMean();
    void updateJointReferences();
    void setJointVelocityReferenceBetweenLimits(std::vector<double> &wheel_speed);
    void updateOdometry();
    void updateCurrentPlatformPosition();
    bool InMotion();
    void publishOdometry();
    void publishJointStates();
 
    
    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& cmd_msg);
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);
    
	bool srvCallback_GetMode(robotnik_msgs::get_mode::Request& request, robotnik_msgs::get_mode::Response& response);
	bool srvCallback_SetOdometry(robotnik_msgs::set_odometry::Request &request, robotnik_msgs::set_odometry::Response &response );
    bool setPlatformPositionCallback(robotnik_msgs::set_height::Request &request, robotnik_msgs::set_height::Response &response);
      
};
PLUGINLIB_EXPORT_CLASS(summit_x_controller::SummitXController, controller_interface::ControllerBase);
}
