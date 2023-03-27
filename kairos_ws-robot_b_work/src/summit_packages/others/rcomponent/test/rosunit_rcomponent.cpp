#include <gtest/gtest.h>
#include <ros/ros.h>

#include <rcomponent/rcomponent.h>

TEST(TestProcedureComponent, shouldGetProperNamespaces)
{
  ros::NodeHandle nh("~");

  rcomponent::RComponent component(nh, "private");

  EXPECT_EQ("/rosunit_procedure_component", component.getPublicNamespace());
  EXPECT_EQ("/rosunit_procedure_component/private", component.getPrivateNamespace());

  FAIL();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rosunit_procedure_component");

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
