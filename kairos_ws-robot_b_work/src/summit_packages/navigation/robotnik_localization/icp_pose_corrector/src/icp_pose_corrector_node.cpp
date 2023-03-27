#include <icp_pose_corrector/icp_pose_corrector.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "icp_pose_corrector");

  IcpPoseCorrector icp_pose;
  icp_pose.init();
  ros::spin();
}
