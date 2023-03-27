#include <?pkg_name/?new_node.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "?new_node");
  ros::NodeHandle n;

  ?NewNode ?new_node(n);
  ?new_node.start();
}