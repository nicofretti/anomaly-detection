#ifndef _ICP_POSE_CORRECTOR_ICP_POSE_CORRECTOR_
#define _ICP_POSE_CORRECTOR_ICP_POSE_CORRECTOR_

#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <nav_msgs/OccupancyGrid.h>

#include <std_srvs/Trigger.h>

#include <costmap_2d/cost_values.h>
#include <laser_geometry/laser_geometry.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>

#include <std_srvs/SetBool.h>

#include <dynamic_reconfigure/server.h>
#include <icp_pose_corrector/IcpPoseCorrectorConfig.h>

typedef pcl::PointXYZ PointType;

namespace Actions
{
enum Action
{
  Correct,
  Initialize
};
}

typedef Actions::Action Action;

class IcpPoseCorrector
{
public:
  IcpPoseCorrector();
  virtual ~IcpPoseCorrector();

  bool init();

private:
  ros::NodeHandle nh_, gnh_, pnh_;

  tf::TransformBroadcaster tf_broadcaster_;
  tf::TransformListener tf_listener_;

  laser_geometry::LaserProjection projector_;

  tf::StampedTransform corrected_transform_;

  std::string base_frame_;
  std::string odom_frame_;
  std::string global_frame_;  /// @brief frame that map is located in

  std::string map_topic_;
  std::string scan_topic_;
  std::string input_pose_topic_;
  std::string initial_pose_topic_;
  std::string set_initial_pose_topic_, set_correct_pose_topic_;

  ros::Subscriber map_sub_, map_update_sub_;
  ros::Subscriber scan_sub_;
  // input pose sub is used to trigger periodic registrations, where the output is used to
  // correct the underlying localization system.
  ros::Subscriber input_pose_sub_;
  // init pose sub is used to trigger an initialization of the localization, where the outuput is used
  // to initialize the underlying localization system
  ros::Subscriber initial_pose_sub_;
  ros::Publisher set_initial_pose_pub_, set_correct_pose_pub_;

  ros::ServiceServer request_correction_srv_;
  ros::ServiceServer toggle_correction_srv_;
  ros::ServiceServer register_localization_srv_;

  ros::Publisher test_pub_;

  // boost::mutex
  boost::mutex config_mutex_;
  void reconfigureCallback(icp_pose_corrector::IcpPoseCorrectorConfig& config);
  dynamic_reconfigure::Server<icp_pose_corrector::IcpPoseCorrectorConfig> reconfigure_server_;

  bool enabled_;

  double max_allowed_distance_correction_;
  double max_allowed_angular_correction_;

  double d_thresh_, a_thresh_;
  int inner_iterations_, outer_iterations_;
  pcl::IterativeClosestPoint<PointType, PointType> registration_algorithm_;

  // transform publishing data
  ros::SteadyTimer tf_broadcast_timer_;
  double tf_broadcast_rate_;

  // map data
  bool map_received_;
  nav_msgs::OccupancyGrid occupancy_;
  pcl::PointCloud<PointType>::Ptr map_;
  pcl::KdTree<PointType>::Ptr map_tree_;

  double inflation_distance_;
  unsigned char obstacle_value_;
  int minimum_coincidences_;
  double minimum_area_covered_;

  // scan data
  sensor_msgs::LaserScan current_scan_;
  bool scan_received_;

  // pose data
  tf::StampedTransform odom_pose_of_last_correction_;
  bool first_pose_received_;

  // callbacks for data
  void incomingMap(const nav_msgs::OccupancyGrid& new_map);
  void incomingScan(const sensor_msgs::LaserScan& scan);
  void incomingPose(const geometry_msgs::PoseWithCovarianceStamped& pose);
  void initialPoseReceived(const geometry_msgs::PoseWithCovarianceStamped& pose);

  // callbacks for services
  bool registerLocalizationCallback(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& res);
  bool requestCorrectionCallback(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& res);
  bool toggleCorrectionCallback(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& res);

  // new functions
  double getMatchingDistanceForPoint(const PointType& robot, const PointType& point);

  void occupancyGridToPointCloud(const nav_msgs::OccupancyGrid& occupancy, const unsigned char& obstacle_value,
                                 pcl::PointCloud<PointType>::Ptr& cloud);

  bool scanToPointCloud(const std::string& fixed_frame, const sensor_msgs::LaserScan& scan,
                        pcl::PointCloud<PointType>::Ptr& cloud);

  void matchScanPointsToMap(const tf::StampedTransform& transform, const pcl::KdTree<PointType>::Ptr& map_tree,
                            const pcl::PointCloud<PointType>::Ptr& scan_in_map_frame, const double inflation_distance,
                            pcl::PointCloud<PointType>::Ptr& matched_map,
                            pcl::PointCloud<PointType>::Ptr& matched_scan);

  bool registerPoseWithCurrentData(const tf::StampedTransform& map_pose_unregistered,
                                   tf::StampedTransform& map_pose_registered,
                                   pcl::PointCloud<PointType>::Ptr& matched_scan_in_map_frame,
                                   pcl::PointCloud<PointType>::Ptr& matched_map,
                                   std::vector<double>& registration_covariance);

  bool acceptRegistration(const tf::StampedTransform& map_pose_unregistered,
                          const tf::StampedTransform& map_pose_registered,
                          const pcl::PointCloud<PointType>::Ptr& matched_scan_in_map_frame,
                          const pcl::PointCloud<PointType>::Ptr& matched_map);

  bool applyOdomCorrection(const tf::StampedTransform& map_pose_registered,
                           const tf::StampedTransform& odom_pose_before, const tf::StampedTransform& odom_pose_after,
                           tf::StampedTransform& map_pose_corrected);

  bool hasTravelledEnough(const tf::StampedTransform& odom_pose_last_correction,
                          const tf::StampedTransform& odom_pose_current);

  double getAreaCoveredByRegistration(const tf::StampedTransform& map_pose,
                                      const pcl::PointCloud<PointType>::Ptr& matched_map);

  bool sendPose(const geometry_msgs::PoseWithCovarianceStamped& pose, const Action& action);

  bool lookupTransform(const std::string& target, const std::string& source, const ros::Time& time,
                       tf::StampedTransform& transform);

  bool lookupTransform(const std::string& target, const ros::Time& target_time, const std::string& source,
                       const ros::Time& source_time, const std::string& fixed_frame, tf::StampedTransform& transform);

  bool computePose(const geometry_msgs::PoseWithCovarianceStamped& pose_unregistered,
                   const bool check_allowed_correction, geometry_msgs::PoseWithCovarianceStamped& pose);

  bool getCurrentPoseInMapFrameAt(const ros::Time& map_time, geometry_msgs::PoseWithCovarianceStamped& pose);
  bool getCurrentPoseInMapFrame(geometry_msgs::PoseWithCovarianceStamped& pose);

  std::vector<double> getDefaultCovariance();
};

#endif  // _ICP_POSE_CORRECTOR_ICP_POSE_CORRECTOR_
