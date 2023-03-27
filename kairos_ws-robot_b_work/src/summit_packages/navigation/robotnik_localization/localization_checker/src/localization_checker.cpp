#include <localization_checker/localization_checker.h>
#include <localization_checker/raytrace_iterator.h>
#include <localization_checker/raytrace_matcher.h>

#include <sensor_msgs/point_cloud_conversion.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/distances.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_2D.h>

#include <tf_conversions/tf_eigen.h>

#include <std_msgs/Float64.h>

LocalizationChecker::LocalizationChecker()
{
}

LocalizationChecker::~LocalizationChecker()
{
}

bool LocalizationChecker::init()
{
  pnh_ = ros::NodeHandle("~");
  nh_ = ros::NodeHandle();
  gnh_ = ros::NodeHandle();

  scan_received_ = false;
  pose_received_ = false;
  first_pose_received_ = false;
  map_received_ = false;
  enabled_ = true;

  pnh_.param("base_frame", base_frame_, std::string("base_footprint"));
  pnh_.param("odom_frame", odom_frame_, std::string("odom"));
  pnh_.param("global_frame", global_frame_, std::string("map"));

  pnh_.param("scan_topic", scan_topic_, std::string("scan"));
  scan_sub_ = gnh_.subscribe(scan_topic_, 1, &LocalizationChecker::incomingScan, this);

  pnh_.param("input_pose_topic", input_pose_topic_, std::string("amcl_pose"));
  input_pose_sub_ = gnh_.subscribe(input_pose_topic_, 1, &LocalizationChecker::incomingPose, this);

  pnh_.param("map_topic", map_topic_, std::string("map"));
  pnh_.param("inflation_distance", inflation_distance_, 0.15);
  pnh_.param("localization_distance", localization_distance_, 10.0);

  int ov;
  pnh_.param("obstacle_value", ov, 100);
  obstacle_value_ = ov < 255 ? ov : 255;

  map_sub_ = gnh_.subscribe(map_topic_, 1, &LocalizationChecker::incomingMap, this);

  pnh_.param("output_pose_topic", output_pose_topic_, std::string("initialpose"));
  output_pose_pub_ = gnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(output_pose_topic_, 1);

  request_localization_check_srv_ =
      gnh_.advertiseService("request_estimation", &LocalizationChecker::requestLocalizationCheckCallback, this);

  toggle_localization_check_srv_ =
      gnh_.advertiseService("toggle_estimation", &LocalizationChecker::toggleLocalizationCheckCallback, this);

  localization_score_pub_ = pnh_.advertise<std_msgs::Float64>("score", 1);
  localization_likelihood_pub_ = pnh_.advertise<std_msgs::Float64>("likelihood", 1);

  // debug topics
  test_pub_ = gnh_.advertise<pcl::PointCloud<PointType>>("test", 1);
  likelihood_test_pub_ = gnh_.advertise<pcl::PointCloud<PointType>>("likelihood_test", 1);

  reconfigure_server_.setCallback(boost::bind(&LocalizationChecker::reconfigureCallback, this, _1));

  return true;
}

void LocalizationChecker::reconfigureCallback(localization_checker::LocalizationCheckerConfig& config)
{
  boost::mutex::scoped_lock l(config_mutex_);

  obstacle_value_ = config.obstacle_value;
  inflation_distance_ = config.inflation_distance;
  localization_distance_ = config.localization_distance;

  d_thresh_ = config.update_min_d;
  a_thresh_ = config.update_min_a;
  enabled_ = config.enabled;
}

bool LocalizationChecker::toggleLocalizationCheckCallback(std_srvs::SetBool::Request& request,
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
      response.message = "Register Pose LocalizationCheck disabled";
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
      response.message = "Register LocalizationCheck enabled";
      ROS_WARN_STREAM(response.message);
      return true;
    }
  }
}

bool LocalizationChecker::requestLocalizationCheckCallback(std_srvs::Trigger::Request& request,
                                                           std_srvs::Trigger::Response& response)
{
  ROS_WARN_STREAM("Remember to add again the request estimation callback to its state");
  if (scan_received_ and map_received_)
  {
    bool check_allowed_correction = false;
    bool successful_estimation = computeLocalizationCheck();

    if (successful_estimation == true)
    {
      response.success = true;
      response.message = "LocalizationCheck has been successful!";
    }
    else
    {
      response.success = false;
      response.message = "LocalizationCheck had too much error. Discarded";
    }
  }
  else
  {
    response.success = false;
    response.message = "LocalizationCheck cannot be computed due to a lack of input data (scan or laser)";
  }

  return true;
}

void LocalizationChecker::incomingScan(const sensor_msgs::LaserScan& scan)
{
  current_scan_ = scan;
  scan_received_ = true;
}

void LocalizationChecker::incomingPose(const geometry_msgs::PoseWithCovarianceStamped& pose)
{
  ROS_INFO_STREAM("Remember to read incomingPose localization estimation");
  // now we only use incoming pose as a trigger for computation
  current_pose_ = pose;
  pose_received_ = true;

  bool must_check_localization = false;

  if (first_pose_received_ == false)
  {
    must_check_localization = true;
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

    must_check_localization = hasTravelledEnough(odom_pose_of_last_correction_, odom_pose_current);
  }

  // Trigger Pose LocalizationCheck each time we receive a pose
  if (scan_received_ and map_received_ and enabled_ and must_check_localization)
  {
    computeLocalizationCheck();
  }
}

void LocalizationChecker::incomingMap(const nav_msgs::OccupancyGrid& occupancy)
{
  occupancy_ = occupancy;
  map_received_ = true;

  ROS_WARN_STREAM("Add different types of localization scorer");
  localization_scorer_ = boost::make_shared<LocalizationScorerLaserScan<PointType>>(occupancy_);
  localization_scorer_->init();

  ROS_WARN_STREAM_THROTTLE(5, "TODO: Implement map updates subscription");
  ROS_DEBUG("Received a %d X %d map at %f m/pix", occupancy_.info.width, occupancy_.info.height,
            occupancy_.info.resolution);
}

bool LocalizationChecker::hasTravelledEnough(const tf::StampedTransform& odom_pose_last_correction,
                                             const tf::StampedTransform& odom_pose_current)
{
  tf::Transform movement = odom_pose_last_correction.inverseTimes(odom_pose_current);
  return movement.getOrigin().length() > d_thresh_ || tf::getYaw(movement.getRotation()) > a_thresh_;
}

bool LocalizationChecker::scanToPointCloud(const std::string& fixed_frame, const sensor_msgs::LaserScan& scan,
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

bool LocalizationChecker::lookupTransform(const std::string& source, const std::string& target, const ros::Time& time,
                                          tf::StampedTransform& transform)
{
  try
  {
    // we could check for the result of waitForTransform, but as what we have to do is the same as for the
    // lookupTransform, do nothing and wait for the exception to trigger it
    tf_listener_.waitForTransform(source, target, time, ros::Duration(0.5));
    tf_listener_.lookupTransform(source, target, time, transform);
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
    return false;
  }

  return true;
}

bool LocalizationChecker::computeLocalizationCheck()
{
  tf::StampedTransform map_pose;   // actual map pose after registration and odom correction
  tf::StampedTransform odom_pose;  // odometry pose when registration was finished

  if (false == lookupTransform(global_frame_, base_frame_, current_scan_.header.stamp, map_pose))
  {  // cannot read map pose
    ROS_WARN_STREAM("Cannot read global -> base tf");
    return false;
  }

  if (false == lookupTransform(odom_frame_, base_frame_, current_scan_.header.stamp, odom_pose))
  {
    // cannot read odom pose
    ROS_WARN_STREAM("Cannot read odom before -> base tf");
    return false;
  }

  if (false == computeLocalizationLikelihood(map_pose))
  {
    // cannot compute registration
    ROS_WARN_STREAM("Cannot compute localization likelihood");
    return false;
  }

  if (false == computeLocalizationScore(map_pose))
  {
    // cannot compute registration
    ROS_WARN_STREAM("Cannot compute localization score");
    return false;
  }

  //  if (false == acceptRegistration(map_pose_unregistered, map_pose_registered))
  //  {
  //    // correction is not likely to have matched
  //    ROS_WARN_STREAM("Error is too high");
  //    return false;
  //  }

  return true;
}

bool LocalizationChecker::computeLocalizationLikelihood(const tf::StampedTransform& map_pose)
{
  pcl::PointCloud<PointType>::Ptr likelihood_scan_in_robot_frame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr likelihood_scan_in_map_frame(new pcl::PointCloud<PointType>());

  // Quick and dirty hack: create a synthetic scan to compute Likelihood
  sensor_msgs::LaserScan likelihood_scan = current_scan_;
  double likelihood_distance =
      localization_distance_ < likelihood_scan.range_max ? localization_distance_ : likelihood_scan.range_max - 0.01;

  std::fill(likelihood_scan.ranges.begin(), likelihood_scan.ranges.end(), likelihood_distance);

  scanToPointCloud(odom_frame_, likelihood_scan, likelihood_scan_in_robot_frame);
  pcl_ros::transformPointCloud(*likelihood_scan_in_robot_frame, *likelihood_scan_in_map_frame, map_pose);
  // setting header should be done by pcl_ros::transformPointCloud, but it isn't...
  likelihood_scan_in_map_frame->header.frame_id = map_pose.frame_id_;
  // stamp in pcl is in microseconds, in ros in nanosecs
  likelihood_scan_in_map_frame->header.stamp = pcl_conversions::toPCL(map_pose.stamp_);

  raytracer_ = boost::make_shared<Raytracer<PointType>>();
  matcher_ = boost::make_shared<LocalizationMatcher<PointType>>();
  likelihood_matcher_ = boost::make_shared<LocalizationLikelihoodMatcher<PointType>>();

  PointType robot;
  robot.x = map_pose.getOrigin().x();
  robot.y = map_pose.getOrigin().y();

  localization_scorer_->raytrace(likelihood_scan_in_map_frame, *raytracer_, *likelihood_matcher_, robot);
  ROS_INFO_STREAM("Computing Likelihood. Score: " << localization_scorer_->score()
                                                  << " (absolute: " << localization_scorer_->absoluteScore() << ")");

  std_msgs::Float64 likelihood;
  likelihood.data = localization_scorer_->score();
  localization_likelihood_pub_.publish(likelihood);

  likelihood_test_pub_.publish(*(localization_scorer_->debug_cloud_));
  return true;
}

bool LocalizationChecker::computeLocalizationScore(const tf::StampedTransform& map_pose)
{
  pcl::PointCloud<PointType>::Ptr scan_in_robot_frame(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr scan_in_map_frame(new pcl::PointCloud<PointType>());

  scanToPointCloud(odom_frame_, current_scan_, scan_in_robot_frame);
  pcl_ros::transformPointCloud(*scan_in_robot_frame, *scan_in_map_frame, map_pose);
  // setting header should be done by pcl_ros::transformPointCloud, but it isn't...
  scan_in_map_frame->header.frame_id = map_pose.frame_id_;
  // stamp in pcl is in microseconds, in ros in nanosecs
  scan_in_map_frame->header.stamp = pcl_conversions::toPCL(map_pose.stamp_);

  raytracer_ = boost::make_shared<Raytracer<PointType>>();
  matcher_ = boost::make_shared<LocalizationMatcher<PointType>>();
  likelihood_matcher_ = boost::make_shared<LocalizationLikelihoodMatcher<PointType>>();

  PointType robot;
  robot.x = map_pose.getOrigin().x();
  robot.y = map_pose.getOrigin().y();

  // localization_scorer_->raytrace(scan_in_map_frame, *raytracer_, *likelihood_matcher_, robot);
  // ROS_INFO_STREAM("Computing Likelihood. Score: " << localization_scorer_->score()
  //                                                << " (absolute: " << localization_scorer_->absoluteScore() << ")");

  localization_scorer_->raytrace(scan_in_map_frame, *raytracer_, *matcher_, robot);
  ROS_INFO_STREAM("Computing score. Score: " << localization_scorer_->score()
                                             << " (absolute: " << localization_scorer_->absoluteScore() << ")");

  std_msgs::Float64 score;
  score.data = localization_scorer_->score();
  localization_score_pub_.publish(score);

  test_pub_.publish(*(localization_scorer_->debug_cloud_));
  return true;
}
