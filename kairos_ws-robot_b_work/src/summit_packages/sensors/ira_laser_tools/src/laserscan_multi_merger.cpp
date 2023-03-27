#include <ros/ros.h>
#include <string.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>
#include <laser_geometry/laser_geometry.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include "sensor_msgs/LaserScan.h"
#include "pcl_ros/point_cloud.h"
#include <Eigen/Dense>
#include <dynamic_reconfigure/server.h>
#include <ira_laser_tools/LaserscanMultiMergerConfig.h>
#include <sstream>

class LaserscanMerger
{
public:
  LaserscanMerger();
  void scanCallback(const sensor_msgs::LaserScan& scan, int input_number);
  void pointcloudToLaserscan(const pcl::PointCloud<pcl::PointXYZI>& merge, sensor_msgs::LaserScan& scan,
                             pcl::PointCloud<pcl::PointXYZI>& output_cloud);
  void reconfigureCallback(ira_laser_tools::LaserscanMultiMergerConfig& config, uint32_t level);

private:
  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  laser_geometry::LaserProjection projector_;
  tf::TransformListener tf_listener_;

  ros::Publisher pointcloud_publisher_;
  ros::Publisher laserscan_publisher_;
  std::vector<ros::Subscriber> scan_subscribers_;

  std::vector<bool> clouds_modified_;
  std::vector<sensor_msgs::LaserScan> input_scans_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>> output_clouds_;

  std::vector<std::string> input_topics_;

  int laserscanTopicParser();
  void subscribeToTopics();
  

  // scan merging configuration
  double angle_min_;
  double angle_max_;
  double angle_increment_;
  double time_increment_;
  double scan_time_;
  double range_min_;
  double range_max_;

  std::string destination_frame_;
  std::string fixed_frame_;
  std::string cloud_destination_topic_;
  std::string scan_destination_topic_;
  bool check_topic_type_;
  std::string laserscan_topics_;

public:

  void waitForInputTopics();
};

LaserscanMerger::LaserscanMerger() : node_(""), private_node_("~")
{
  private_node_.getParam("destination_frame", destination_frame_);
  private_node_.getParam("fixed_frame", fixed_frame_);
  private_node_.getParam("cloud_destination_topic", cloud_destination_topic_);
  private_node_.getParam("scan_destination_topic", scan_destination_topic_);

  private_node_.getParam("laserscan_topics", laserscan_topics_);
  check_topic_type_ = false;
  private_node_.getParam("check_topic_type", check_topic_type_);

  pointcloud_publisher_ = private_node_.advertise<sensor_msgs::PointCloud2>(cloud_destination_topic_, 1, false);
  laserscan_publisher_ = private_node_.advertise<sensor_msgs::LaserScan>(scan_destination_topic_, 1, false);
}

void LaserscanMerger::reconfigureCallback(ira_laser_tools::LaserscanMultiMergerConfig& config, uint32_t level)
{
  this->angle_min_ = config.angle_min;
  this->angle_max_ = config.angle_max;
  this->angle_increment_ = config.angle_increment;
  this->time_increment_ = config.time_increment;
  this->scan_time_ = config.scan_time;
  this->range_min_ = config.range_min;
  this->range_max_ = config.range_max;
}

void LaserscanMerger::waitForInputTopics(){
  ros::Rate r(1);
  bool topics_found = false;

  while (ros::ok() && (topics_found != true))
  {
    ROS_WARN_STREAM_THROTTLE(5, "LaserscanMerger::waitForInputTopics: waiting for scan topics " << laserscan_topics_ );
   
    if(laserscanTopicParser() == 0)
    {
      subscribeToTopics();
      topics_found = true;
    }

    ros::spinOnce();
    r.sleep();
     
  }


}

int LaserscanMerger::laserscanTopicParser()
{
  // LaserScan topics to subscribe
  ros::master::V_TopicInfo topics;
  ros::master::getTopics(topics);

  std::istringstream iss(laserscan_topics_);
  std::vector<std::string> tokens;
  std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
            std::back_inserter<std::vector<std::string>>(tokens));

  std::vector<std::string> tmp_input_topics_;
  

  if (check_topic_type_ == false)
  {
    tmp_input_topics_ = tokens;
  }
  else
  {
    for (int i = 0; i < tokens.size(); ++i)
    {
      for (int j = 0; j < topics.size(); ++j)
      {
        if ((topics[j].name.compare(node_.resolveName(tokens[i])) == 0) &&
            (topics[j].datatype.compare("sensor_msgs/LaserScan") == 0))
        {
          tmp_input_topics_.push_back(topics[j].name);
          ROS_INFO_STREAM("LaserscanMerger::laserscanTopicParser: found topic " << topics[j].name );
        }
      }
    }
  }

  sort(tmp_input_topics_.begin(), tmp_input_topics_.end());
  std::vector<std::string>::iterator last = std::unique(tmp_input_topics_.begin(), tmp_input_topics_.end());
  tmp_input_topics_.erase(last, tmp_input_topics_.end());
  input_topics_ = tmp_input_topics_;

  if(tokens.size() == input_topics_.size())
    return 0;
  else
    return -1;
}

void LaserscanMerger::subscribeToTopics()
{
  // Unsubscribe from previous topics
  for (int i = 0; i < scan_subscribers_.size(); ++i)
    scan_subscribers_[i].shutdown();

  if (input_topics_.size() > 0)
  {
    scan_subscribers_.resize(input_topics_.size());
    clouds_modified_.resize(input_topics_.size());
    input_scans_.resize(input_topics_.size());
    output_clouds_.resize(input_topics_.size());

    std::ostringstream ss;
    ss << "Subscribing to " << scan_subscribers_.size() << " topics:";
    for (int i = 0; i < input_topics_.size(); ++i)
    {
      // this weird node_.subscribe<MsgType, const MsgType&> is needed to bind to a callback
      // which receives a const Type&
      // otherwise, we are forced to subscribe to callback which receives a shared_ptr
      scan_subscribers_[i] = node_.subscribe<sensor_msgs::LaserScan, const sensor_msgs::LaserScan&>(
          input_topics_[i], 1, boost::bind(&LaserscanMerger::scanCallback, this, _1, i));
      clouds_modified_[i] = false;
      ss << " " << input_topics_[i];
    }
    ROS_INFO_STREAM("LaserscanMerger::subscribeToTopic: " << ss.str());
  }
  else
    ROS_ERROR_STREAM("LaserscanMerger::subscribeToTopic: Not subscribed to any topic.");
}

void LaserscanMerger::scanCallback(const sensor_msgs::LaserScan& scan, int input_number)
{
  // Verify that TF knows how to transform from the received scan to the fixed scan frame
  // as we are using a high precission projector, we need the tf at the time of the last
  // scan
  try
  {
    ros::Time end_stamp = (scan.header.stamp + ros::Duration().fromSec((scan.ranges.size() - 1) * scan.time_increment));
    if (!tf_listener_.waitForTransform(fixed_frame_, scan.header.frame_id, end_stamp, ros::Duration(10)))
      return;
  }
  catch (const tf::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }

  input_scans_[input_number] = scan;
  sensor_msgs::PointCloud2 scan_in_fixed_frame;
  try
  {
    projector_.transformLaserScanToPointCloud(fixed_frame_, scan, scan_in_fixed_frame, tf_listener_);
    // need this weird if. otherwise, in case of no intensities on input, it will keep printing at each iteration
    // a warning message about intensities missing field
    if (scan.intensities.size() != 0)
    {
      pcl::fromROSMsg(scan_in_fixed_frame, output_clouds_[input_number]);
    }
    else
    {
      pcl::PointCloud<pcl::PointXYZ> cloud_without_intensities;
      pcl::fromROSMsg(scan_in_fixed_frame, cloud_without_intensities);
      pcl::copyPointCloud(cloud_without_intensities, output_clouds_[input_number]);
    }
  }
  catch (const tf::TransformException& ex)
  {
    ROS_ERROR("%s", ex.what());
    return;
  }

  clouds_modified_[input_number] = true;

  // Count how many scans we have
  int totalClouds = 0;
  for (int i = 0; i < clouds_modified_.size(); ++i)
    if (clouds_modified_[i])
      ++totalClouds;

  // Go ahead only if all subscribed scans have arrived
  if (totalClouds == clouds_modified_.size())
  {
    ros::Time current_stamp;
    pcl_conversions::fromPCL(std::min_element(output_clouds_.begin(), output_clouds_.end(),
                                              [](pcl::PointCloud<pcl::PointXYZI> n, pcl::PointCloud<pcl::PointXYZI> m) {
                                                return n.header.stamp < m.header.stamp;
                                              })
                                 ->header.stamp,
                             current_stamp);

    int first = 0;
    for (first = 0; first < clouds_modified_.size(); first++)
      if (output_clouds_[first].points.size() != 0)
        break;

    if (first == clouds_modified_.size())
    {
      ROS_INFO_THROTTLE(5, "Input is empty (either laserscan was empty or couldn't be transformed into fixed frame)");
      return;
    }

    for (int i = first; i < clouds_modified_.size(); ++i)
    {
      clouds_modified_[i] = false;
    }

    pcl::PointCloud<pcl::PointXYZI> merge;
    merge.header.frame_id = destination_frame_;
    merge.header.stamp = pcl_conversions::toPCL(current_stamp);
    for (auto& cloud : output_clouds_)
    {
      pcl::PointCloud<pcl::PointXYZI> transformed;
      pcl_ros::transformPointCloud(destination_frame_, current_stamp, cloud, fixed_frame_, transformed, tf_listener_);
      merge += transformed;
    }

    sensor_msgs::LaserScan output_scan;
    output_scan.header.frame_id = destination_frame_;
    output_scan.header.stamp = current_stamp;
    pcl::PointCloud<pcl::PointXYZI> output_cloud;
    output_cloud.header = merge.header;
    pointcloudToLaserscan(merge, output_scan, output_cloud);

    pointcloud_publisher_.publish(output_cloud);
    laserscan_publisher_.publish(output_scan);
  }
}

void LaserscanMerger::pointcloudToLaserscan(const pcl::PointCloud<pcl::PointXYZI>& merge,
                                            sensor_msgs::LaserScan& output_scan,
                                            pcl::PointCloud<pcl::PointXYZI>& output_cloud)
{
  output_scan.angle_min = this->angle_min_;
  output_scan.angle_max = this->angle_max_;
  output_scan.angle_increment = this->angle_increment_;
  output_scan.time_increment = this->time_increment_;
  output_scan.scan_time = this->scan_time_;
  output_scan.range_min = this->range_min_;
  output_scan.range_max = this->range_max_;

  uint32_t ranges_size = std::ceil((output_scan.angle_max - output_scan.angle_min) / output_scan.angle_increment);
  output_scan.ranges.assign(ranges_size, output_scan.range_max + 1.0);
  output_scan.intensities.assign(ranges_size, 0);

  output_cloud.points.clear();
  for (auto& point : merge.points)
  {
    const float& x = point.x;
    const float& y = point.y;
    const float& z = point.z;
    const float& intensity = point.intensity;
    if (std::isnan(x) || std::isnan(y) || std::isnan(z))
    {
      ROS_DEBUG("rejected for nan in point(%f, %f, %f)\n", x, y, z);
      continue;
    }

    double range_sq = y * y + x * x;
    double range_min_sq_ = output_scan.range_min * output_scan.range_min;
    if (range_sq < range_min_sq_)
    {
      ROS_DEBUG("rejected for range %f below minimum value %f. Point: (%f, %f, %f)", range_sq, range_min_sq_, x, y, z);
      continue;
    }

    double angle = atan2(y, x);
    if (angle < output_scan.angle_min || angle > output_scan.angle_max)
    {
      ROS_DEBUG("rejected for angle %f not in range (%f, %f)\n", angle, output_scan.angle_min, output_scan.angle_max);
      continue;
    }
    int index = (angle - output_scan.angle_min) / output_scan.angle_increment;

    if (output_scan.ranges[index] * output_scan.ranges[index] > range_sq)
    {
      output_scan.ranges[index] = sqrt(range_sq);
      output_scan.intensities[index] = intensity;
      output_cloud.points.push_back(point);
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laser_multi_merger");

  LaserscanMerger laser_merger;
  
  laser_merger.waitForInputTopics();

  dynamic_reconfigure::Server<ira_laser_tools::LaserscanMultiMergerConfig> reconfigure_server;
  dynamic_reconfigure::Server<ira_laser_tools::LaserscanMultiMergerConfig>::CallbackType reconfigure_callback;

  reconfigure_callback = boost::bind(&LaserscanMerger::reconfigureCallback, &laser_merger, _1, _2);
  reconfigure_server.setCallback(reconfigure_callback);

  

  ros::spin();

  return 0;
}
