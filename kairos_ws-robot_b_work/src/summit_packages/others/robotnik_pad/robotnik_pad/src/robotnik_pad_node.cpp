#include <robotnik_pad/robotnik_pad.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "robotnik_pad");
  ros::NodeHandle n;

  RobotnikPad robotnik_pad(n);
  robotnik_pad.start();
}