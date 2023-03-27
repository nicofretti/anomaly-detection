#include <iostream>

#include <localization_checker/raytrace_matcher.h>
#include <localization_checker/raytrace_iterator.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <gtest/gtest.h>

const int info_debug_level = 2;
const int error_debug_level = 1;
const int g_debug_output = error_debug_level;

typedef pcl::PointXYZ PointType;

class RaytraceMatcherTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    map_cloud.reset(new pcl::PointCloud<PointType>());
    map_cloud->points.push_back(PointType(1.0, 1.0, 0.0));
    map_cloud->points.push_back(PointType(-1.0, 1.0, 0.0));
    map_cloud->points.push_back(PointType(1.0, -1.0, 0.0));
    map_cloud->points.push_back(PointType(-1.0, -1.0, 0.0));
  }

  void createScanCloudFromMapCloud(double scale)
  {
    scan_cloud.reset(new pcl::PointCloud<PointType>());
    for (auto& point : map_cloud->points)
    {
      scan_cloud->points.push_back(PointType(point.x * scale, point.y * scale, point.z * scale));
    }
  }

  void addPointToScanCloud(PointType point)
  {
    scan_cloud->points.push_back(point);
  }
  void removePointFromScanCloud()
  {
    scan_cloud->points.pop_back();
  }

  pcl::PointCloud<PointType>::Ptr map_cloud;
  pcl::PointCloud<PointType>::Ptr scan_cloud;
};

TEST_F(RaytraceMatcherTest, localizationLikelihoodMatcherFinished)
{
  double scale = 2.0;
  createScanCloudFromMapCloud(scale);
  pcl::KdTree<PointType>::Ptr map_tree(new pcl::KdTreeFLANN<PointType>);
  map_tree->setInputCloud(map_cloud);

  pcl::KdTree<PointType>::Ptr scan_tree(new pcl::KdTreeFLANN<PointType>);
  scan_tree->setInputCloud(scan_cloud);

  PointType center = PointType(0, 0, 0);
  for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //  for (auto& scan_point : scan_cloud->points)
  {
    PointType scan_point = scan_cloud->points[i];
    PointType map_point = map_cloud->points[i];

    LocalizationLikelihoodMatcher<PointType> matcher(map_tree, scan_tree);

    double step = 0.1;
    Raytracer<PointType> raytracer(center, scan_point, step);
    int iterations = 0;
    // raytrace in x direction, from 0 to 1 and step 0.1, so points are [0.1, 0.2,..., 1.0], thats 10
    double distance = std::sqrt(map_point.x * map_point.x + map_point.y * map_point.y);
    int iterations_to_iterate = distance / step;
    while (raytracer.hasNext() == true and matcher.hasFinished() == false)
    {
      iterations++;
      matcher.match(raytracer.next(), scan_point);
      if (g_debug_output >= info_debug_level)
        std::cout << raytracer.current() << std::endl;
    }
    EXPECT_TRUE(matcher.hasFinished());
    EXPECT_EQ(iterations, iterations_to_iterate);
  }
}

TEST_F(RaytraceMatcherTest, localizationLikelihoodMatcherScoreFullMatch)
{
  double scale = 2.0;
  createScanCloudFromMapCloud(scale);

  pcl::KdTree<PointType>::Ptr map_tree(new pcl::KdTreeFLANN<PointType>);
  map_tree->setInputCloud(map_cloud);

  pcl::KdTree<PointType>::Ptr scan_tree(new pcl::KdTreeFLANN<PointType>);
  scan_tree->setInputCloud(scan_cloud);

  double total_score = 0;
  PointType center = PointType(0, 0, 0);
  for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //  for (auto& scan_point : scan_cloud->points)
  {
    PointType scan_point = scan_cloud->points[i];
    PointType map_point = map_cloud->points[i];

    LocalizationLikelihoodMatcher<PointType> matcher(map_tree, scan_tree);

    double step = 0.1;
    Raytracer<PointType> raytracer(center, scan_point, step);
    while (raytracer.hasNext() == true and matcher.hasFinished() == false)
    {
      matcher.match(raytracer.next(), scan_point);
      if (g_debug_output >= info_debug_level)
        std::cout << raytracer.current() << std::endl;
    }
    double score = matcher.score();
    EXPECT_EQ(score, 1.0);
    total_score += score;
  }
  EXPECT_EQ(total_score, 4.0);
}

TEST_F(RaytraceMatcherTest, localizationLikelihoodMatcherScoreMissesAll)
{
  double scale = 0.0;
  createScanCloudFromMapCloud(scale);

  pcl::KdTree<PointType>::Ptr map_tree(new pcl::KdTreeFLANN<PointType>);
  map_tree->setInputCloud(map_cloud);

  pcl::KdTree<PointType>::Ptr scan_tree(new pcl::KdTreeFLANN<PointType>);
  scan_tree->setInputCloud(scan_cloud);

  double total_score = 0;
  PointType center = PointType(0, 0, 0);
  for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //  for (auto& scan_point : scan_cloud->points)
  {
    PointType scan_point = scan_cloud->points[i];
    PointType map_point = map_cloud->points[i];

    LocalizationLikelihoodMatcher<PointType> matcher(map_tree, scan_tree);

    double step = 0.1;
    Raytracer<PointType> raytracer(center, scan_point, step);
    while (raytracer.hasNext() == true and matcher.hasFinished() == false)
    {
      matcher.match(raytracer.next(), scan_point);
      if (g_debug_output >= info_debug_level)
        std::cout << raytracer.current() << std::endl;
    }
    double score = matcher.score();
    if (scan_point.x != 0)
      EXPECT_EQ(score, 1.0);
    else
      EXPECT_EQ(score, 0.0);
    total_score += score;
  }
  EXPECT_EQ(total_score, 0.0);
}

TEST_F(RaytraceMatcherTest, localizationLikelihoodMatcherScoreMissesOne)
{
  double scale = 2.0;
  createScanCloudFromMapCloud(scale);
  addPointToScanCloud(PointType(0, 0, 0));

  pcl::KdTree<PointType>::Ptr map_tree(new pcl::KdTreeFLANN<PointType>);
  map_tree->setInputCloud(map_cloud);

  pcl::KdTree<PointType>::Ptr scan_tree(new pcl::KdTreeFLANN<PointType>);
  scan_tree->setInputCloud(scan_cloud);

  PointType center = PointType(0, 0, 0);
  double total_score = 0;
  int points_with_0_score = 0;
  int points_with_1_score = 0;
  for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //  for (auto& scan_point : scan_cloud->points)
  {
    PointType scan_point = scan_cloud->points[i];

    LocalizationLikelihoodMatcher<PointType> matcher(map_tree, scan_tree);

    double step = 0.1;
    Raytracer<PointType> raytracer(center, scan_point, step);
    while (raytracer.hasNext() == true and matcher.hasFinished() == false)
    {
      matcher.match(raytracer.next(), scan_point);
      if (g_debug_output >= info_debug_level)
        std::cout << raytracer.current() << std::endl;
    }
    double score = matcher.score();
    if (scan_point.x != 0)
    {
      EXPECT_EQ(score, 1.0);
      points_with_1_score++;
    }
    else
    {
      EXPECT_EQ(score, 0.0);
      points_with_0_score++;
    }
    total_score += score;
  }
  EXPECT_EQ(1, points_with_0_score);
  EXPECT_EQ(4, points_with_1_score);
  EXPECT_EQ(total_score, 4.0);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  // ros::init(argc, argv, "matcher_test");
  // ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
