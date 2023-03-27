#ifndef _LOCALIZATION_CHECKER_LOCALIZATION_CHECKER_
#define _LOCALIZATION_CHECKER_LOCALIZATION_CHECKER_

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

#include <std_srvs/SetBool.h>

#include <dynamic_reconfigure/server.h>
#include <localization_checker/LocalizationCheckerConfig.h>

#include <localization_checker/localization_scorer.h>

typedef pcl::PointXYZI PointType;

class LocalizationChecker
{
public:
  LocalizationChecker();
  virtual ~LocalizationChecker();

  bool init();

private:
  ros::NodeHandle nh_, gnh_, pnh_;

  tf::TransformListener tf_listener_;

  laser_geometry::LaserProjection projector_;

  tf::StampedTransform corrected_transform_;

  std::string base_frame_;
  std::string odom_frame_;
  std::string global_frame_;  /// @brief frame that map is located in

  std::string map_topic_;
  std::string scan_topic_;
  std::string input_pose_topic_, output_pose_topic_;

  ros::Subscriber map_sub_, map_update_sub_;
  ros::Subscriber scan_sub_;
  ros::Subscriber input_pose_sub_;

  ros::Publisher output_pose_pub_;

  ros::Publisher localization_score_pub_;
  ros::Publisher localization_likelihood_pub_;

  ros::ServiceServer request_localization_check_srv_;
  ros::ServiceServer toggle_localization_check_srv_;

  ros::Publisher test_pub_;
  ros::Publisher likelihood_test_pub_;

  // boost::mutex
  boost::mutex config_mutex_;
  void reconfigureCallback(localization_checker::LocalizationCheckerConfig& config);
  dynamic_reconfigure::Server<localization_checker::LocalizationCheckerConfig> reconfigure_server_;

  bool enabled_;

  // map data
  bool map_received_;
  nav_msgs::OccupancyGrid occupancy_;
  pcl::PointCloud<PointType>::Ptr map_;
  pcl::KdTree<PointType>::Ptr map_tree_;

  double d_thresh_, a_thresh_;
  double inflation_distance_;
  double localization_distance_;
  unsigned char obstacle_value_;

  // scan data
  sensor_msgs::LaserScan current_scan_;
  bool scan_received_;

  LocalizationScorer<PointType>::Ptr localization_scorer_;
  Raytracer<PointType>::Ptr raytracer_;
  LocalizationMatcher<PointType>::Ptr matcher_;                       //(map_cloud, scan_cloud);
  LocalizationLikelihoodMatcher<PointType>::Ptr likelihood_matcher_;  //(map_cloud, scan_cloud);

  // pose data
  tf::StampedTransform odom_pose_of_last_correction_;
  geometry_msgs::PoseWithCovarianceStamped current_pose_;
  geometry_msgs::PoseWithCovarianceStamped corrected_pose_;
  bool pose_received_, first_pose_received_;

  // callbacks for data
  void incomingMap(const nav_msgs::OccupancyGrid& new_map);
  void incomingScan(const sensor_msgs::LaserScan& scan);
  void incomingPose(const geometry_msgs::PoseWithCovarianceStamped& pose);

  // callbacks for services
  bool requestLocalizationCheckCallback(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& res);
  bool toggleLocalizationCheckCallback(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& res);

  // localization estimation functions
  void raytraceLine(const pcl::KdTree<PointType>::Ptr& map_tree, const PointType& init_position,
                    const PointType& end_position);

  // new functions
  double getMatchingDistanceForPoint(const PointType& robot, const PointType& point);

  void occupancyGridToPointCloud(const nav_msgs::OccupancyGrid& occupancy, const unsigned char& obstacle_value,
                                 pcl::PointCloud<PointType>::Ptr& cloud);

  bool scanToPointCloud(const std::string& fixed_frame, const sensor_msgs::LaserScan& scan,
                        pcl::PointCloud<PointType>::Ptr& cloud);

  void matchScanPointsToMap(const tf::StampedTransform& transform, const pcl::KdTree<PointType>::Ptr& map_tree,
                            const pcl::PointCloud<PointType>::Ptr& scan, const double inflation_distance,
                            pcl::PointCloud<PointType>::Ptr& matched_scan);

  bool computeLocalizationScore(const tf::StampedTransform& map_pose_unregistered);
  bool computeLocalizationLikelihood(const tf::StampedTransform& map_pose_unregistered);

  bool acceptRegistration(const tf::StampedTransform& map_pose_unregistered,
                          const tf::StampedTransform& map_pose_registered);

  bool applyOdomCorrection(const tf::StampedTransform& map_pose_registered,
                           const tf::StampedTransform& odom_pose_before, const tf::StampedTransform& odom_pose_after,
                           tf::StampedTransform& map_pose_corrected);

  bool hasTravelledEnough(const tf::StampedTransform& odom_pose_last_correction,
                          const tf::StampedTransform& odom_pose_current);

  bool setNewPose(const tf::StampedTransform& map_pose, const tf::StampedTransform& odom_pose);

  bool lookupTransform(const std::string& source, const std::string& target, const ros::Time& time,
                       tf::StampedTransform& transform);

  bool computeLocalizationCheck();
};

#endif  // _LOCALIZATION_CHECKER_LOCALIZATION_CHECKER_
