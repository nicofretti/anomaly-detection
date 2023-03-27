#include <gtest/gtest.h>
#include <ros/ros.h>

#include <rcomponent/rcomponent.h>

#include <log4cxx/appenderskeleton.h>
#include <log4cxx/spi/loggingevent.h>

TEST(TestProcedureComponent, shouldGetProperNamespaces)
{
  ros::NodeHandle nh("~");

  rcomponent::RComponent component(nh, "private");

  ROS_ERROR("pako");
  ROS_ERROR_NAMED("pepe", "manolo");

  EXPECT_EQ("/rosunit_procedure_component", component.getPublicNamespace());
  EXPECT_EQ("/rosunit_procedure_component/private", component.getPrivateNamespace());

  FAIL();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rosunit_component_log");

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
