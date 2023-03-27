#include <iostream>

#include <localization_checker/raytrace_iterator.h>

#include <gtest/gtest.h>

const int info_debug_level = 2;
const int error_debug_level = 1;
const int g_debug_output = error_debug_level;

typedef pcl::PointXYZ PointType;

TEST(RaytracerIterator, iterateOnOneAxisPositive)
{
  double start_x = 0;
  double end_x = 1.0;
  double step = .1;
  PointType start(start_x, 0, 0);
  PointType end(end_x, 0, 0);

  std::vector<PointType> real_points;
  for (double x = start_x + step; x <= end_x; x += step)
    real_points.push_back(PointType(x, 0, 0));

  // just to check that we create the correct array
  EXPECT_EQ(real_points.size(), 10);
  // if (real_points.size() != raytraced_points.size())
  // {
  //   if (g_debug_output >= error_debug_level)
  //   {
  //     std::cout << "sets of different sizes, real: " << real_points.size() << ", raytraced " <<
  //     raytraced_points.size()
  //               << std::endl;
  //     return false;
  //   }
  // }

  Raytracer<PointType> raytracer(start, end, step);
  std::vector<PointType> raytraced_points;

  while (raytracer.hasNext())
  {
    raytraced_points.push_back(raytracer.next());
    if (g_debug_output >= info_debug_level)
      std::cout << raytraced_points.back() << std::endl;
  }

  EXPECT_EQ(real_points.size(), raytraced_points.size());

  for (size_t i = 0; i < real_points.size(); i++)
  {
    // if (real_points[i].x != raytraced_points[i].x)
    //{
    //  if (g_debug_output >= error_debug_level)
    //    std::cout << "points at " << i << " are different: " << real_points[i] << ", raytraced " <<
    //    raytraced_points[i]
    //              << std::endl;
    //  return false;
    //}
    EXPECT_FLOAT_EQ(real_points[i].x, raytraced_points[i].x);
  }
}

TEST(RaytracerIterator, iterateOnOneAxisNegative)
{
  double start_x = 0;
  double end_x = -1.0;
  double step = .1;
  PointType start(start_x, 0, 0);
  PointType end(end_x, 0, 0);

  std::vector<PointType> real_points;
  int steps = std::abs(end_x - start_x) / step;
  int sign = (end_x - start_x) >= 0 ? 1 : -1;
  // for (double x = start_x + step; x <= end_x; x += step)
  for (int i = 1; i <= steps; i++)
    real_points.push_back(PointType(start_x + (sign * i * step), 0, 0));

  // just to check that we create the correct array
  EXPECT_EQ(real_points.size(), 10);
  // if (real_points.size() != raytraced_points.size())
  // {
  //   if (g_debug_output >= error_debug_level)
  //   {
  //     std::cout << "sets of different sizes, real: " << real_points.size() << ", raytraced " <<
  //     raytraced_points.size()
  //               << std::endl;
  //     return false;
  //   }
  // }

  Raytracer<PointType> raytracer(start, end, step);
  std::vector<PointType> raytraced_points;

  while (raytracer.hasNext())
  {
    raytraced_points.push_back(raytracer.next());
    if (g_debug_output >= info_debug_level)
      std::cout << raytraced_points.back() << std::endl;
  }

  EXPECT_EQ(real_points.size(), raytraced_points.size());

  for (size_t i = 0; i < real_points.size(); i++)
  {
    // if (real_points[i].x != raytraced_points[i].x)
    //{
    //  if (g_debug_output >= error_debug_level)
    //    std::cout << "points at " << i << " are different: " << real_points[i] << ", raytraced " <<
    //    raytraced_points[i]
    //              << std::endl;
    //  return false;
    //}
    EXPECT_FLOAT_EQ(real_points[i].x, raytraced_points[i].x);
  }
}

TEST(RaytracerIterator, addLastPointWhenIterationIsNotEven)
{
  double start_x = 0;
  double end_x = 1.1;
  double step = .2;
  PointType start(start_x, 0, 0);
  PointType end(end_x, 0, 0);

  std::vector<PointType> real_points;
  for (double x = start_x + step; x <= end_x; x += step)
    real_points.push_back(PointType(x, 0, 0));

  // this should be tested with ASSERT/EXPECT_FLOAT_NE, but gtest does not have that macro :S
  ASSERT_NE(real_points.back().x, end_x);
  real_points.push_back(PointType(end_x, 0, 0));
  for (auto& p : real_points)
    if (g_debug_output >= info_debug_level)
      std::cout << p << std::endl;

  Raytracer<PointType> raytracer(start, end, step);
  std::vector<PointType> raytraced_points;

  while (raytracer.hasNext())
  {
    raytraced_points.push_back(raytracer.next());
    if (g_debug_output >= info_debug_level)
      std::cout << raytraced_points.back() << std::endl;
  }

  EXPECT_EQ(real_points.size(), std::ceil((end_x - start_x) / step));
  EXPECT_EQ(real_points.size(), raytraced_points.size());

  for (size_t i = 0; i < real_points.size(); i++)
  {
    EXPECT_FLOAT_EQ(real_points[i].x, raytraced_points[i].x);
  }
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  // ros::init(argc, argv, "matcher_test");
  // ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}
