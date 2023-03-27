#include <?rc_package/?rc_node.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "?rc_node");
  ros::NodeHandle n;

  ?RCNode ?rc_node(n);
  ?rc_node.start();
}