#ifndef _LOCALIZATION_CHECKER_LOCALIZATION_SCORER_
#define _LOCALIZATION_CHECKER_LOCALIZATION_SCORER_

#include <localization_checker/raytrace_matcher.h>
#include <localization_checker/raytrace_iterator.h>

#include <nav_msgs/OccupancyGrid.h>

#include <pcl/common/distances.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

template <typename PointT>
class LocalizationScorer
{
public:
  typedef boost::shared_ptr<LocalizationScorer> Ptr;
  LocalizationScorer(const nav_msgs::OccupancyGrid& grid)
  {
    grid_ = grid;
    obstacle_value_ = 100;
  }

  virtual void init()
  {
    map_cloud_.reset(new pcl::PointCloud<PointT>());
    occupancyGridToPointCloud(grid_, obstacle_value_, map_cloud_);

    map_tree_.reset(new pcl::KdTreeFLANN<PointT>());
    map_tree_->setInputCloud(map_cloud_);

    absolute_score_ = 0.0;
  }

  virtual double score()
  {
    return absolute_score_;
  }

  virtual double absoluteScore()
  {
    return absolute_score_;
  }

  void raytrace(const typename pcl::PointCloud<PointT>::Ptr& scan_cloud, Raytracer<PointT>& raytracer,
                RaytraceMatcher<PointT>& matcher, PointT robot_pose = PointT())
  {
    absolute_score_ = 0;

    scan_cloud_ = scan_cloud;
    scan_tree_.reset(new pcl::KdTreeFLANN<PointT>());
    scan_tree_->setInputCloud(scan_cloud_);

    matcher.setMapTree(map_tree_);
    matcher.setScanTree(scan_tree_);

    debug_cloud_.reset(new pcl::PointCloud<PointT>());
    *debug_cloud_ = *scan_cloud_;
    debug_cloud_->points.resize(0);
    debug_cloud_->points.push_back(robot_pose);
    for (auto& scan_point : scan_cloud_->points)
    {
      double score;
      raytracePoint(scan_point, raytracer, matcher, score, robot_pose);
      absolute_score_ += score;
      PointT to_add = scan_point;
      to_add.intensity = 0.5;
      debug_cloud_->points.push_back(to_add);
      // debug_cloud_->points.push_back(scan_point);
    }
    debug_cloud_->width = debug_cloud_->points.size();
    debug_cloud_->height = 1;
  }

  void raytracePoint(const PointT point, Raytracer<PointT>& raytracer, RaytraceMatcher<PointT>& matcher,
                     double& point_score, PointT robot_pose = PointT())
  {
    double step = 0.1;
    raytracer.init(robot_pose, point, step);
    matcher.reset();
    while (raytracer.hasNext() == true and matcher.hasFinished() == false)
    {
      matcher.match(raytracer.next(), point);
      PointT to_add = raytracer.current();
      to_add.intensity = 1.0;
      debug_cloud_->points.push_back(to_add);
      // debug_cloud_->points.push_back(raytracer.current());
    }
    point_score = matcher.score();
  }

protected:
  nav_msgs::OccupancyGrid grid_;
  typename pcl::PointCloud<PointT>::Ptr map_cloud_;
  typename pcl::KdTree<PointT>::Ptr map_tree_;
  typename pcl::PointCloud<PointT>::Ptr scan_cloud_;
  typename pcl::KdTree<PointT>::Ptr scan_tree_;

  double resolution_;
  double score_;
  double absolute_score_;
  double scoring_factor_;  // is the factor applied to the
  uint8_t obstacle_value_;

public:
  typename pcl::PointCloud<PointT>::Ptr debug_cloud_;
  // this is a helper method to convert grids to pointclouds
  void occupancyGridToPointCloud(const nav_msgs::OccupancyGrid& grid, const unsigned char& obstacle_value,
                                 typename pcl::PointCloud<PointT>::Ptr& cloud)
  {
    int init_x, init_y, end_x, end_y;

    init_x = 0;
    init_y = 0;

    end_x = grid.info.width;
    end_y = grid.info.height;

    cloud->clear();
    for (unsigned int my = init_y; my < end_y; my++)
    {
      for (unsigned int mx = init_x; mx < end_x; mx++)
      {
        unsigned int index = my * grid.info.width + mx;

        unsigned char value = grid.data[index];

        if (value == obstacle_value)  // to not use functions from costmap
        {
          double wx, wy;
          wx = grid.info.origin.position.x + (mx + 0.5) * grid.info.resolution;
          wy = grid.info.origin.position.y + (my + 0.5) * grid.info.resolution;

          PointT point;
          point.x = wx;
          point.y = wy;
          point.z = 0;
          cloud->push_back(point);
        }
      }
    }
  }
};

template <typename PointT>
class LocalizationScorerLaserScan : public LocalizationScorer<PointT>
{
public:
  LocalizationScorerLaserScan(const nav_msgs::OccupancyGrid& grid) : LocalizationScorer<PointT>(grid)
  {
  }

  virtual double score()
  {
    // std::cout << "Absolute: " << LocalizationScorer<PointT>::absolute_score_
    //          << ", factor: " << LocalizationScorer<PointT>::scan_cloud_->points.size() << ", Score: "
    //          << (LocalizationScorer<PointT>::absolute_score_ /
    //          LocalizationScorer<PointT>::scan_cloud_->points.size())
    //          << "\n";
    return LocalizationScorer<PointT>::absolute_score_ / LocalizationScorer<PointT>::scan_cloud_->points.size();
  }
};
#endif  // _LOCALIZATION_CHECKER_LOCALIZATION_SCORER_
