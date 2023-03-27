#include <localization_checker/localization_checker.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "localization_checker");

  LocalizationChecker icp_pose;
  icp_pose.init();
  ros::spin();
}
