#include <icp_pose_corrector/icp_pose_corrector.h>

#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/distances.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_2D.h>

#include <tf_conversions/tf_eigen.h>

IcpPoseCorrector::IcpPoseCorrector()
{
}

IcpPoseCorrector::~IcpPoseCorrector()
{
}

bool IcpPoseCorrector::init()
{
  pnh_ = ros::NodeHandle("~");
  nh_ = ros::NodeHandle();
  gnh_ = ros::NodeHandle();

  scan_received_ = false;
  first_pose_received_ = false;
  map_received_ = false;
  enabled_ = true;

  pnh_.param("base_frame", base_frame_, std::string("base_footprint"));
  pnh_.param("odom_frame", odom_frame_, std::string("odom"));
  pnh_.param("global_frame", global_frame_, std::string("map"));

  pnh_.param("scan_topic", scan_topic_, std::string("scan"));
  scan_sub_ = gnh_.subscribe(scan_topic_, 1, &IcpPoseCorrector::incomingScan, this);

  pnh_.param("input_pose_topic", input_pose_topic_, std::string("input_pose"));
  input_pose_sub_ = gnh_.subscribe(input_pose_topic_, 1, &IcpPoseCorrector::incomingPose, this);

  pnh_.param("initial_pose_topic", initial_pose_topic_, std::string("initialpose_with_registration"));
  initial_pose_sub_ = gnh_.subscribe(initial_pose_topic_, 1, &IcpPoseCorrector::initialPoseReceived, this);

  pnh_.param("map_topic", map_topic_, std::string("map"));
  pnh_.param("inflation_distance", inflation_distance_, 0.15);
  int ov;
  pnh_.param("obstacle_value", ov, 100);
  obstacle_value_ = ov < 255 ? ov : 255;

  map_sub_ = gnh_.subscribe(map_topic_, 1, &IcpPoseCorrector::incomingMap, this);

  pnh_.param("set_initital_pose_topic", set_initial_pose_topic_, std::string("initialpose"));
  set_initial_pose_pub_ = gnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(set_initial_pose_topic_, 1);

  pnh_.param("set_correct_pose_topic", set_correct_pose_topic_, std::string("correctpose"));
  set_correct_pose_pub_ = gnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(set_correct_pose_topic_, 1);

  pnh_.param("outer_iterations", outer_iterations_, 5);
  pnh_.param("inner_iterations", inner_iterations_, 250);

  pnh_.param("max_allowed_distance_correction", max_allowed_distance_correction_, 0.4);  // 0 all are allowed
  pnh_.param("max_allowed_angular_correction", max_allowed_angular_correction_, 0.25);

  register_localization_srv_ =
      gnh_.advertiseService("register_localization", &IcpPoseCorrector::registerLocalizationCallback, this);

  request_correction_srv_ =
      gnh_.advertiseService("request_correction", &IcpPoseCorrector::requestCorrectionCallback, this);

  toggle_correction_srv_ =
      gnh_.advertiseService("toggle_correction", &IcpPoseCorrector::toggleCorrectionCallback, this);

  pnh_.param("tf_broadcast_rate", tf_broadcast_rate_, 10.0);
  if (tf_broadcast_rate_ <= 0)
  {
    ROS_ERROR_STREAM("tf_broadcast_rate == " << tf_broadcast_rate_ << " that is bad. Set to default: 10 hz");
    tf_broadcast_rate_ = 10;
  }
  pnh_.param("minimum_coincidences", minimum_coincidences_, 5);
  pnh_.param("minimum_area_covered", minimum_area_covered_, 100.0);

  pnh_.param("update_min_d", d_thresh_, 0.4);
  pnh_.param("update_min_a", a_thresh_, 0.4);
  //  tf_broadcast_timer_ = gnh_.createSteadyTimer(ros::WallDuration(1/tf_broadcast_rate_),

  // debug topics

  test_pub_ = gnh_.advertise<pcl::PointCloud<PointType>>("test", 1);

  reconfigure_server_.setCallback(boost::bind(&IcpPoseCorrector::reconfigureCallback, this, _1));

  return true;
}

void IcpPoseCorrector::reconfigureCallback(icp_pose_corrector::IcpPoseCorrectorConfig& config)
{
  boost::mutex::scoped_lock l(config_mutex_);

  minimum_coincidences_ = config.minimum_coincidences;
  minimum_area_covered_ = config.minimum_area_covered;
  obstacle_value_ = config.obstacle_value;
  inner_iterations_ = config.inner_iterations;
  outer_iterations_ = config.outer_iterations;
  max_allowed_distance_correction_ = config.max_allowed_distance_correction;
  max_allowed_angular_correction_ = config.max_allowed_angular_correction;
  inflation_distance_ = config.inflation_distance;
  d_thresh_ = config.update_min_d;
  a_thresh_ = config.update_min_a;
  enabled_ = config.enabled;
}

bool IcpPoseCorrector::toggleCorrectionCallback(std_srvs::SetBool::Request& request,
                                                std_srvs::SetBool::Response& response)
{
  if (enabled_)
  {
    if (request.data == true)
    {
      response.success = true;
      response.message = "Already enabled, this call is useless";
      ROS_WARN_STREAM(response.message);
      return true;
    }
    else
    {
      enabled_ = false;
      response.success = true;
      response.message = "Register Pose Correction disabled";
      ROS_WARN_STREAM(response.message);
      return true;
    }
  }
  if (not enabled_)
  {
    if (request.data == false)
    {
      response.success = true;
      response.message = "Already disabled, this call is useless";
      ROS_WARN_STREAM(response.message);
      return true;
    }
    else
    {
      enabled_ = true;
      response.success = true;

      if (false == lookupTransform(odom_frame_, base_frame_, ros::Time(0), odom_pose_of_last_correction_))
      {
        // cannot read odom pose
        response.message = "Register Correction enabled. But could not read odom pose. May bad corrections be applied";
        ROS_WARN_STREAM(response.message);
      }
      else
      {
        response.message = "Register Correction enabled, and read odom pose. Expect to work properly";
        ROS_INFO_STREAM(response.message);
      }
    }
  }
}

bool IcpPoseCorrector::registerLocalizationCallback(std_srvs::Trigger::Request& request,
                                                    std_srvs::Trigger::Response& response)
{
  // TODO: registerLocalizationCallback, requestCorrectionCallback, initialPoseReceived and incomingPose should be
  // refactored because they almost do the same

  if (not scan_received_ or not map_received_)
  {
    response.success = false;
    response.message = "Registration cannot be computed due to a lack of input data (scan or map)";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  geometry_msgs::PoseWithCovarianceStamped unregistered_pose;
  bool got_unregistered_pose = getCurrentPoseInMapFrameAt(ros::Time(0), unregistered_pose);

  if (got_unregistered_pose == false)
  {
    response.success = false;
    response.message = "Pose registration could not read old map pose";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  bool check_allowed_correction = false;
  geometry_msgs::PoseWithCovarianceStamped registered_pose;

  bool successful_correction = computePose(unregistered_pose, check_allowed_correction, registered_pose);
  if (successful_correction == false)
  {
    response.success = false;
    response.message = "Registration had too much error. Discarded";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  bool successful_pose_publish = sendPose(registered_pose, Actions::Initialize);
  if (successful_pose_publish == false)
  {
    response.success = false;
    response.message = "Registration could not be sent!";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  response.success = true;
  response.message = "Registration has been successful!";
  ROS_INFO_STREAM(response.message);
  return true;
}

bool IcpPoseCorrector::requestCorrectionCallback(std_srvs::Trigger::Request& request,
                                                 std_srvs::Trigger::Response& response)
{
  // TODO: registerLocalizationCallback, requestCorrectionCallback, initialPoseReceived and incomingPose should be
  // refactored because they almost do the same

  if (not scan_received_ or not map_received_)
  {
    response.success = false;
    response.message = "Registration cannot be computed due to a lack of input data (scan or map)";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  geometry_msgs::PoseWithCovarianceStamped unregistered_pose;
  bool got_unregistered_pose = getCurrentPoseInMapFrameAt(ros::Time(0), unregistered_pose);

  if (got_unregistered_pose == false)
  {
    response.success = false;
    response.message = "Pose registration could not read old map pose";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  bool check_allowed_correction = false;
  geometry_msgs::PoseWithCovarianceStamped registered_pose;

  bool successful_correction = computePose(unregistered_pose, check_allowed_correction, registered_pose);
  if (successful_correction == false)
  {
    response.success = false;
    response.message = "Registration had too much error. Discarded";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  bool successful_pose_publish = sendPose(registered_pose, Actions::Correct);
  if (successful_pose_publish == false)
  {
    response.success = false;
    response.message = "Registration could not be sent!";
    ROS_WARN_STREAM(response.message);
    return true;
  }

  response.success = true;
  response.message = "Registration has been successful!";
  ROS_INFO_STREAM(response.message);
  return true;
}

void IcpPoseCorrector::initialPoseReceived(const geometry_msgs::PoseWithCovarianceStamped& unregistered_pose)
{
  // TODO: registerLocalizationCallback, requestCorrectionCallback, initialPoseReceived and incomingPose should be
  // refactored because they almost do the same

  if (not scan_received_ or not map_received_)
  {
    ROS_WARN_STREAM("Registration cannot be computed due to a lack of input data (scan or map)");
    return;
  }

  bool check_allowed_correction = false;
  geometry_msgs::PoseWithCovarianceStamped registered_pose;

  bool successful_correction = computePose(unregistered_pose, check_allowed_correction, registered_pose);
  if (successful_correction == false)
  {
    ROS_WARN_STREAM("Registration had too much error. Discarded");
    return;
  }

  bool successful_pose_publish = sendPose(registered_pose, Actions::Initialize);
  if (successful_pose_publish == false)
  {
    ROS_WARN_STREAM("Registration could not be sent!");
    return;
  }

  ROS_INFO_STREAM("Registration has been successful!");
  return;
}

void IcpPoseCorrector::incomingPose(const geometry_msgs::PoseWithCovarianceStamped& incoming_pose)
{
  // TODO: registerLocalizationCallback, requestCorrectionCallback, initialPoseReceived and incomingPose should be
  // refactored because they almost do the same

  if (not scan_received_ or not map_received_)
  {
    ROS_WARN_STREAM_THROTTLE(5, "Registration cannot be computed due to a lack of input data (scan or map)");
    return;
  }

  bool must_update_pose = false;

  if (first_pose_received_ == false)
  {
    if (false == lookupTransform(odom_frame_, base_frame_, ros::Time(0), odom_pose_of_last_correction_))
    {
      // ROS_INFO("This is really bad. I cannot read odom");
      return;
    }
    must_update_pose = true;
    first_pose_received_ = true;
  }
  else
  {
    tf::StampedTransform odom_pose_current;
    if (false == lookupTransform(odom_frame_, base_frame_, ros::Time(0), odom_pose_current))
    {
      // cannot read current odom
      return;
    }

    must_update_pose = hasTravelledEnough(odom_pose_of_last_correction_, odom_pose_current);
  }

  if (false == must_update_pose)
    return;

  // TODO: there is a race condition here: if amcl is initialized with init pose, it can publish through its topic the
  // new  pose, triggering the computation, but tf may not be updated, so the registration "moves back" the pose to
  // the previous one maybe solved by reading tf at different times, but then the normal operation does not work.

  geometry_msgs::PoseWithCovarianceStamped unregistered_pose;
  // bool got_unregistered_pose = getCurrentPoseInMapFrameAt(incoming_pose.header.stamp, up);
  bool got_unregistered_pose = getCurrentPoseInMapFrame(unregistered_pose);
  if (got_unregistered_pose == false)
  {
    ROS_WARN_STREAM("Pose registration could not read old map pose");
    return;
  }

  bool check_allowed_correction = true;
  geometry_msgs::PoseWithCovarianceStamped registered_pose;

  bool successful_correction = computePose(unregistered_pose, check_allowed_correction, registered_pose);
  if (successful_correction == false)
  {
    ROS_WARN_STREAM("Registration had too much error. Discarded");
    return;
  }

  bool successful_pose_publish = sendPose(registered_pose, Actions::Correct);
  if (successful_pose_publish == false)
  {
    ROS_WARN_STREAM("Registration could not be sent!");
    return;
  }

  ROS_INFO_STREAM("Registration has been successful!");
}

void IcpPoseCorrector::incomingScan(const sensor_msgs::LaserScan& scan)
{
  current_scan_ = scan;
  scan_received_ = true;
}

void IcpPoseCorrector::incomingMap(const nav_msgs::OccupancyGrid& occupancy)
{
  occupancy_ = occupancy;
  // TODO: check if should be reset here or what to do
  map_ = boost::make_shared<pcl::PointCloud<PointType>>();
  map_received_ = true;

  occupancyGridToPointCloud(occupancy_, obstacle_value_, map_);
  map_tree_.reset(new pcl::KdTreeFLANN<PointType>);
  map_tree_->setInputCloud(map_);

  ROS_WARN_STREAM_THROTTLE(5, "TODO: Implement map updates subscription");
  ROS_DEBUG("Received a %d X %d map at %f m/pix", occupancy_.info.width, occupancy_.info.height,
            occupancy_.info.resolution);
}

bool IcpPoseCorrector::hasTravelledEnough(const tf::StampedTransform& odom_pose_previous,
                                          const tf::StampedTransform& odom_pose_current)
{
  tf::Transform movement = odom_pose_previous.inverseTimes(odom_pose_current);
  return movement.getOrigin().length() > d_thresh_ || tf::getYaw(movement.getRotation()) > a_thresh_;
}

void IcpPoseCorrector::occupancyGridToPointCloud(const nav_msgs::OccupancyGrid& occupancy,
                                                 const unsigned char& obstacle_value,
                                                 pcl::PointCloud<PointType>::Ptr& cloud)
{
  int init_x, init_y, end_x, end_y;

  init_x = 0;
  init_y = 0;

  end_x = occupancy.info.width;
  end_y = occupancy.info.height;

  cloud->clear();
  for (unsigned int mx = init_x; mx < end_x; mx++)
  {
    for (unsigned int my = init_y; my < end_y; my++)
    {
      unsigned int index = my * occupancy.info.width + mx;

      unsigned char value = occupancy.data[index];

      if (value == obstacle_value)  // to not use functions from costmap
      {
        double wx, wy;
        wx = occupancy.info.origin.position.x + (mx + 0.5) * occupancy.info.resolution;
        wy = occupancy.info.origin.position.y + (my + 0.5) * occupancy.info.resolution;

        PointType point;
        point.x = wx;
        point.y = wy;
        point.z = 0;
        cloud->push_back(point);
      }
    }
  }
}

double IcpPoseCorrector::getMatchingDistanceForPoint(const PointType& robot, const PointType& point)
{
  //  double d = pcl::euclideanDistance(robot, point) * std::tan(max_allowed_angular_correction_);
  //  double max_inflation_distance_ = inflation_distance_ * 3;
  //  double min_inflation_distance_ = inflation_distance_;
  //
  //  return std::min(std::max(d, min_inflation_distance_), max_inflation_distance_);

  return inflation_distance_;
}

struct PointCmp
{
  bool operator()(const PointType& lhs, const PointType& rhs) const
  {
    return std::hypot(lhs.x, lhs.y) < std::hypot(rhs.x, rhs.y);
  }
};

void IcpPoseCorrector::matchScanPointsToMap(const tf::StampedTransform& map_pose,
                                            const pcl::KdTree<PointType>::Ptr& map_tree,
                                            const pcl::PointCloud<PointType>::Ptr& scan_in_map_frame,
                                            const double inflation_distance,
                                            pcl::PointCloud<PointType>::Ptr& matched_map,
                                            pcl::PointCloud<PointType>::Ptr& matched_scan)
{
  //
  std::set<PointType, PointCmp> matched_map_points;

  matched_map->clear();
  matched_map->header = scan_in_map_frame->header;

  matched_scan->clear();
  matched_scan->header = scan_in_map_frame->header;

  std::vector<int> nn_indices(1);
  std::vector<float> nn_sqr_dists(1);

  PointType robot;
  robot.x = map_pose.getOrigin().x();
  robot.y = map_pose.getOrigin().y();
  robot.z = map_pose.getOrigin().z();
  for (auto point_in_scan : *scan_in_map_frame)
  {
    ROS_WARN_STREAM_THROTTLE(5, "TODO: match points scan2map depending on the range of scan");
    if (map_tree->radiusSearch(point_in_scan, getMatchingDistanceForPoint(robot, point_in_scan), nn_indices,
                               nn_sqr_dists, 1) != 0)
    {
      matched_scan->push_back(point_in_scan);
      matched_map_points.insert(map_tree->getInputCloud()->at(nn_indices[0]));
    }
  }

  for (auto point_in_matched_map : matched_map_points)
    matched_map->push_back(point_in_matched_map);
}

bool IcpPoseCorrector::scanToPointCloud(const std::string& fixed_frame, const sensor_msgs::LaserScan& scan,
                                        pcl::PointCloud<PointType>::Ptr& cloud)
{
  sensor_msgs::PointCloud2 scan_in_fixed_frame, scan_in_base_frame;
  // convert first to a fixed and continuous frame (should be odom), to take into account displacement during capture
  projector_.transformLaserScanToPointCloud(fixed_frame, scan, scan_in_fixed_frame, tf_listener_);
  // convert back to the base frame of the robot
  pcl_ros::transformPointCloud(base_frame_, scan_in_fixed_frame, scan_in_base_frame, tf_listener_);
  pcl::fromROSMsg(scan_in_base_frame, *cloud);

  return true;
}

bool IcpPoseCorrector::computePose(const geometry_msgs::PoseWithCovarianceStamped& pose_unregistered,
                                   const bool check_allowed_correction,
                                   geometry_msgs::PoseWithCovarianceStamped& registered_pose)
{
  tf::StampedTransform map_pose_unregistered;  // current read map pose
  tf::StampedTransform map_pose_registered;    // map pose compute registering scan data to map
  tf::StampedTransform odom_pose_before;       // odometry pose when scan data was taken
  tf::StampedTransform odom_pose_after;        // odometry pose when registration was finished

  tf::StampedTransform map_pose;  // actual map pose after registration and odom correction

  // data used to calculate the final registration
  pcl::PointCloud<PointType>::Ptr matched_scan_in_map_frame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr matched_map(new pcl::PointCloud<PointType>());

  tf::poseMsgToTF(pose_unregistered.pose.pose, map_pose_unregistered);
  map_pose_unregistered.stamp_ = pose_unregistered.header.stamp;
  map_pose_unregistered.frame_id_ = pose_unregistered.header.frame_id;

  if (false == lookupTransform(odom_frame_, base_frame_, map_pose_unregistered.stamp_, odom_pose_before))
  {
    // cannot read odom pose
    ROS_WARN_STREAM("Cannot read odom before -> base tf");
    return false;
  }

  std::vector<double> registration_covariance;
  if (false == registerPoseWithCurrentData(map_pose_unregistered, map_pose_registered, matched_scan_in_map_frame,
                                           matched_map, registration_covariance))
  {
    // cannot compute registration
    ROS_WARN_STREAM("Cannot register pose");
    return false;
  }

  if (true == check_allowed_correction and
      false == acceptRegistration(map_pose_unregistered, map_pose_registered, matched_scan_in_map_frame, matched_map))
  {
    // correction is not likely to have matched
    ROS_WARN_STREAM("Error is too high");
    return false;
  }

  if (false == lookupTransform(odom_frame_, base_frame_, ros::Time(0), odom_pose_after))
  {
    // cannot read odom pose
    ROS_WARN_STREAM("Cannot read odom after -> base tf");
    return false;
  }

  if (false == applyOdomCorrection(map_pose_registered, odom_pose_before, odom_pose_after, map_pose))
  {
    // cannot apply correction
    ROS_WARN_STREAM("Cannot apply odom correction");
    return false;
  }

  tf::poseTFToMsg(map_pose, registered_pose.pose.pose);
  registered_pose.header.stamp = map_pose.stamp_;
  registered_pose.header.frame_id = map_pose.frame_id_;

  for (size_t i = 0; i < registered_pose.pose.covariance.size(); i++)
    registered_pose.pose.covariance[i] = registration_covariance[i];

  return true;
}

std::vector<double> IcpPoseCorrector::getDefaultCovariance()
{
  float factorPos = 0.03;
  float factorRot = 0.1;
  float cov = 0.5;

  std::vector<double> covariance(36);
  covariance[0] = (cov * cov) * factorPos;
  covariance[7] = (cov * cov) * factorPos;
  covariance[35] = (M_PI / 12.0 * M_PI / 12.0) * factorRot;
  return covariance;
}

bool IcpPoseCorrector::sendPose(const geometry_msgs::PoseWithCovarianceStamped& pose, const Action& action)
{
  switch (action)
  {
    case Actions::Initialize:
      set_initial_pose_pub_.publish(pose);
      break;
    case Actions::Correct:
      set_correct_pose_pub_.publish(pose);
      break;
    default:
      ROS_ERROR_STREAM("Should never reach here");
      return false;
  }

  // TODO: this is a candidate to be refactored, it is repeated three times in the code
  if (false == lookupTransform(odom_frame_, base_frame_, ros::Time(0), odom_pose_of_last_correction_))
  {
    // cannot read odom pose
    std::string message = "Sent pose to localization system, but could not read odom pose. May bad corrections be "
                          "applied";
    ROS_WARN_STREAM(message);
  }
  else
  {
    std::string message = "Sent pose to localization system, and read odom pose. Expect to work properly";
    ROS_INFO_STREAM(message);
  }

  return true;
}

bool IcpPoseCorrector::applyOdomCorrection(const tf::StampedTransform& map_pose_registered,
                                           const tf::StampedTransform& odom_pose_before,
                                           const tf::StampedTransform& odom_pose_after, tf::StampedTransform& map_pose)
{
  map_pose = map_pose_registered;
  map_pose.stamp_ = odom_pose_after.stamp_;
  map_pose.setData(map_pose * odom_pose_before.inverseTimes(odom_pose_after));
  return true;
}

bool IcpPoseCorrector::acceptRegistration(const tf::StampedTransform& map_pose_unregistered,
                                          const tf::StampedTransform& map_pose_registered,
                                          const pcl::PointCloud<PointType>::Ptr& matched_scan_in_map_frame,
                                          const pcl::PointCloud<PointType>::Ptr& matched_map)
{
  tf::Transform final_correction = map_pose_registered.inverseTimes(map_pose_unregistered);

  double distance_correction = final_correction.getOrigin().length();
  double angular_correction = tf::getYaw(final_correction.getRotation());

  if ((std::abs(distance_correction) > max_allowed_distance_correction_ and max_allowed_distance_correction_ > 0) or
      (std::abs(angular_correction) > max_allowed_angular_correction_ and max_allowed_angular_correction_ > 0))
  {
    ROS_WARN("Correction is too high in euclidean or angular distance.");
    return false;
  }

  if (getAreaCoveredByRegistration(map_pose_registered, matched_map) < minimum_area_covered_)
  {
    ROS_WARN("Correction has too few area covered.");
    return false;
  }

  return true;
}

bool IcpPoseCorrector::registerPoseWithCurrentData(const tf::StampedTransform& map_pose_unregistered,
                                                   tf::StampedTransform& map_pose_registered,
                                                   pcl::PointCloud<PointType>::Ptr& matched_scan_in_map_frame,
                                                   pcl::PointCloud<PointType>::Ptr& matched_map,
                                                   std::vector<double>& registration_covariance)
{
  pcl::PointCloud<PointType>::Ptr scan_in_robot_frame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr scan_in_map_frame(new pcl::PointCloud<PointType>());

  scanToPointCloud(odom_frame_, current_scan_, scan_in_robot_frame);

  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  ROS_WARN_STREAM_THROTTLE(5, "TODO: Check parameters for inner registration algorithm");
  ROS_WARN_STREAM_THROTTLE(5, "TODO: Check for data type: ptr or not");
  registration_algorithm_.setTransformationEpsilon(1e-6);
  registration_algorithm_.setMaximumIterations(inner_iterations_);

  //  pcl::registration::TransformationEstimation2D<PointType, PointType>::Ptr transformation_estimation_(
  //      new pcl::registration::TransformationEstimation2D<PointType, PointType>());
  //  registration_algorithm_.setTransformationEstimation(transformation_estimation_);

  double distance_threshold = 1.0;
  map_pose_registered = map_pose_unregistered;
  for (int i = 0; i < outer_iterations_; i++)
  {
    scan_in_map_frame->clear();

    // transform input scan to current map pose estimation
    pcl_ros::transformPointCloud(*scan_in_robot_frame, *scan_in_map_frame, map_pose_registered);
    // setting header should be done by pcl_ros::transformPointCloud, but it isn't...
    scan_in_map_frame->header.frame_id = map_pose_registered.frame_id_;
    // stamp in pcl is in microseconds, in ros in nanosecs
    scan_in_map_frame->header.stamp = pcl_conversions::toPCL(map_pose_registered.stamp_);

    matched_map->clear();
    matched_scan_in_map_frame->clear();
    // leave only points that are potential map features
    matchScanPointsToMap(map_pose_registered, map_tree_, scan_in_map_frame, inflation_distance_, matched_map,
                         matched_scan_in_map_frame);

    int points_to_register = scan_in_map_frame->width * scan_in_map_frame->height;
    if (points_to_register < minimum_coincidences_)
    {
      // not enough points to register scan
      ROS_WARN_STREAM_THROTTLE(5, "Scan has " << points_to_register << ", I need at least " << minimum_coincidences_
                                              << ". Stopping computation!");
      return false;
    }

    test_pub_.publish(*matched_scan_in_map_frame);

    // calculate registration
    distance_threshold /= 2.0;
    registration_algorithm_.setMaxCorrespondenceDistance(distance_threshold);
    registration_algorithm_.setRANSACOutlierRejectionThreshold(distance_threshold / 2.0);

    registration_algorithm_.setInputTarget(map_);
    registration_algorithm_.setInputSource(matched_scan_in_map_frame);

    pcl::PointCloud<PointType> registered_scan_in_map;
    registration_algorithm_.align(registered_scan_in_map);
    if (false == registration_algorithm_.hasConverged())
    {
      ROS_WARN_STREAM_THROTTLE(5, "Registration algorithm has not converged. Stopping computation!");
      return false;
    }

    // get map pose estimation
    Eigen::Matrix4d m = registration_algorithm_.getFinalTransformation().cast<double>();
    // conversion from Eigen::Matrix4f should be done by tf_conversions,
    // but it can only manage rotations or translations...
    // Do it here manually, and just in case is done more times, refactor to a method
    tf::StampedTransform partial_correction;
    tf::matrixEigenToTF(m.block<3, 3>(0, 0), partial_correction.getBasis());
    tf::vectorEigenToTF(m.block<3, 1>(0, 3), partial_correction.getOrigin());

    map_pose_registered.setData(partial_correction * map_pose_registered);
  }

  registration_covariance = getDefaultCovariance();

  return true;
}

double IcpPoseCorrector::getAreaCoveredByRegistration(const tf::StampedTransform& map_pose,
                                                      const pcl::PointCloud<PointType>::Ptr& matched_map)
{
  double area = 0;
  PointType current_point;
  current_point.x = map_pose.getOrigin().x();
  current_point.y = map_pose.getOrigin().y();
  for (auto p : *matched_map)
  {
    area += std::sqrt(std::pow(p.x - current_point.x, 2) + std::pow(p.y - current_point.y, 2));
  }

  ROS_INFO("Covered area in pose (%f, %f, %f) is %f", current_point.x, current_point.y,
           tf::getYaw(map_pose.getRotation()), area);

  return area;
}

bool IcpPoseCorrector::getCurrentPoseInMapFrameAt(const ros::Time& map_time,
                                                  geometry_msgs::PoseWithCovarianceStamped& pose)
{
  tf::StampedTransform map_pose;  // current read map pose
  if (false == lookupTransform(global_frame_, map_time, base_frame_, current_scan_.header.stamp, odom_frame_, map_pose))
  {  // cannot read map pose
    ROS_WARN_STREAM("Cannot read global -> base tf");
    return false;
  }

  tf::poseTFToMsg(map_pose, pose.pose.pose);
  pose.header.frame_id = map_pose.frame_id_;
  pose.header.stamp = map_pose.stamp_;

  return true;
}

bool IcpPoseCorrector::getCurrentPoseInMapFrame(geometry_msgs::PoseWithCovarianceStamped& pose)
{
  tf::StampedTransform map_pose;  // current read map pose

  if (false == lookupTransform(global_frame_, base_frame_, current_scan_.header.stamp, map_pose))
  {  // cannot read map pose
    ROS_WARN_STREAM("Cannot read global -> base tf");
    return false;
  }

  tf::poseTFToMsg(map_pose, pose.pose.pose);
  pose.header.frame_id = map_pose.frame_id_;
  pose.header.stamp = map_pose.stamp_;

  return true;
}

bool IcpPoseCorrector::lookupTransform(const std::string& target, const std::string& source, const ros::Time& time,
                                       tf::StampedTransform& transform)
{
  tf::StampedTransform read_transform;
  try
  {
    // we could check for the result of waitForTransform, but as what we have to do is the same as for the
    // lookupTransform, do nothing and wait for the exception to trigger it
    tf_listener_.waitForTransform(target, source, time, ros::Duration(2));
    tf_listener_.lookupTransform(target, source, time, read_transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  transform = read_transform;
  return true;
}

bool IcpPoseCorrector::lookupTransform(const std::string& target, const ros::Time& target_time,
                                       const std::string& source, const ros::Time& source_time,
                                       const std::string& fixed_frame, tf::StampedTransform& transform)
{
  tf::StampedTransform read_transform;
  try
  {
    // we could check for the result of waitForTransform, but as what we have to do is the same as for the
    // lookupTransform, do nothing and wait for the exception to trigger it
    tf_listener_.waitForTransform(target, target_time, source, source_time, fixed_frame, ros::Duration(2));
    tf_listener_.lookupTransform(target, target_time, source, source_time, fixed_frame, read_transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }
  transform = read_transform;
  return true;
}
