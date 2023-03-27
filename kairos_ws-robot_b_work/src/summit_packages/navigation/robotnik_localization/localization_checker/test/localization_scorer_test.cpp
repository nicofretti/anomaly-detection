#include <localization_checker/localization_scorer.h>

#include <nav_msgs/OccupancyGrid.h>

#include <pcl/kdtree/kdtree.h>
#include <gtest/gtest.h>

const int info_debug_level = 2;
const int error_debug_level = 1;
const int g_debug_output = error_debug_level;

typedef pcl::PointXYZI PointType;

class LocalizationScorerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    for (double x = -edge; x <= edge; x += resolution)
    {
      PointType p;
      p.x = x;
      p.y = -edge;
      real_points.push_back(p);
    }
    for (double y = -edge + resolution; y <= edge - resolution; y += resolution)
    {
      PointType p;
      p.x = -edge;
      p.y = y;
      real_points.push_back(p);
      p.x = edge;
      p.y = y;
      real_points.push_back(p);
    }
    for (double x = -edge; x <= edge; x += resolution)
    {
      PointType p;
      p.x = x;
      p.y = edge;
      real_points.push_back(p);
    }

    // Create a grid map with points in the edges
    // grid.info.resolution = 0.05;
    // grid.info.width = 5;
    // grid.info.height = 5;
    // grid.info.origin.position.x = -0.125;
    // grid.info.origin.position.y = -0.125;
    grid.info.resolution = resolution;
    grid.info.width = ((int)(edge / resolution)) * 2 + 1;
    grid.info.height = ((int)(edge / resolution)) * 2 + 1;
    grid.info.origin.position.x = -edge - resolution / 2.0;
    grid.info.origin.position.y = -edge - resolution / 2.0;

    size_t grid_size = grid.info.width * grid.info.height;
    grid.data = std::vector<int8_t>(grid_size, 0);
    for (size_t i = 0; i < grid.info.width; i++)
    {
      size_t index;
      index = i;
      grid.data[index] = occupied;
      index = grid_size - i - 1;
      grid.data[index] = occupied;
    }
    for (size_t i = 0; i < grid.info.height; i++)
    {
      size_t index;
      index = i * grid.info.width;
      grid.data[index] = occupied;
      index = i * grid.info.width + grid.info.width - 1;
      grid.data[index] = occupied;
    }
  }

  std::vector<PointType> real_points;
  nav_msgs::OccupancyGrid grid;
  static const int8_t occupied = 100;
  static constexpr double edge = 0.20;
  static constexpr double resolution = 0.05;
};

TEST_F(LocalizationScorerTest, convertGridToPointCloud)
{
  // Real points in the edges

  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  LocalizationScorer<PointType> ls(grid);
  ls.occupancyGridToPointCloud(grid, occupied, map_cloud);

  ASSERT_EQ(real_points.size(), map_cloud->points.size());
  for (size_t i = 0; i < map_cloud->points.size(); i++)
  {
    EXPECT_NEAR(real_points[i].x, map_cloud->points[i].x, 1e-5);
  }
}

TEST_F(LocalizationScorerTest, naivePerfectMatch)
{
  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr scan_cloud(new pcl::PointCloud<PointType>());

  LocalizationScorer<PointType> ls(grid);
  ls.init();
  ls.occupancyGridToPointCloud(grid, occupied, map_cloud);
  ls.occupancyGridToPointCloud(grid, occupied, scan_cloud);

  Raytracer<PointType> raytracer;
  LocalizationLikelihoodMatcher<PointType> matcher;  //(map_cloud, scan_cloud);

  ls.raytrace(scan_cloud, raytracer, matcher);

  EXPECT_EQ(scan_cloud->points.size(), ls.absoluteScore());
  EXPECT_NEAR(32, ls.score(), 1e-5);  // 32 is the number of points in the edges according to the synthetic map created
}

TEST_F(LocalizationScorerTest, naiveHalfMatch)
{
  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr scan_cloud(new pcl::PointCloud<PointType>());

  LocalizationScorer<PointType> ls(grid);
  ls.init();
  ls.occupancyGridToPointCloud(grid, occupied, map_cloud);
  ls.occupancyGridToPointCloud(grid, occupied, scan_cloud);
  scan_cloud->points.resize(scan_cloud->points.size() / 2);

  Raytracer<PointType> raytracer;
  LocalizationLikelihoodMatcher<PointType> matcher;  //(map_cloud, scan_cloud);

  ls.raytrace(scan_cloud, raytracer, matcher);

  EXPECT_EQ(scan_cloud->points.size(), ls.absoluteScore());
  EXPECT_NEAR(scan_cloud->points.size(), ls.score(), 1e-5);
}

TEST_F(LocalizationScorerTest, scanPerfectMatch)
{
  LocalizationScorerLaserScan<PointType> ls(grid);
  ls.init();

  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  ls.occupancyGridToPointCloud(grid, occupied, map_cloud);

  pcl::PointCloud<PointType>::Ptr scan_cloud(new pcl::PointCloud<PointType>());
  ls.occupancyGridToPointCloud(grid, occupied, scan_cloud);

  Raytracer<PointType> raytracer;
  LocalizationMatcher<PointType> matcher(map_cloud, scan_cloud);

  ls.raytrace(scan_cloud, raytracer, matcher);

  EXPECT_EQ(scan_cloud->points.size(), ls.absoluteScore());
  EXPECT_NEAR(1, ls.score(), 1e-5);
}

TEST_F(LocalizationScorerTest, scanHalfMatch)
{
  LocalizationScorerLaserScan<PointType> ls(grid);
  ls.init();

  pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>());
  ls.occupancyGridToPointCloud(grid, occupied, map_cloud);

  pcl::PointCloud<PointType>::Ptr scan_cloud(new pcl::PointCloud<PointType>());
  ls.occupancyGridToPointCloud(grid, occupied, scan_cloud);
  //  std::cout << "map\n";
  //  for (size_t i = 0; i < map_cloud->points.size(); i++)
  //    std::cout << map_cloud->points[i] << "\n";
  //  std::cout << "scan\n";
  //
  //  for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //    std::cout << scan_cloud->points[i] << "\n";
  //  std::cout << "a\n";

  for (size_t i = scan_cloud->points.size() / 2; i < scan_cloud->points.size(); i++)
  {
    scan_cloud->points[i].x = 0;
    scan_cloud->points[i].y = 0;
  }

  // for (size_t i = 0; i < scan_cloud->points.size(); i++)
  //  std::cout << scan_cloud->points[i] << "\n";

  Raytracer<PointType> raytracer;
  LocalizationMatcher<PointType> matcher;  //(map_cloud, scan_cloud);

  ls.raytrace(scan_cloud, raytracer, matcher);

  EXPECT_EQ(scan_cloud->points.size() / 2, ls.absoluteScore());
  EXPECT_NEAR(0.5, ls.score(), 1e-5);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  // ros::init(argc, argv, "matcher_test");
  // ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
