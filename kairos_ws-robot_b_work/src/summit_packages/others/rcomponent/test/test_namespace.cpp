#include <ros/ros.h>

#include <rcomponent/rcomponent.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rosunit_procedure_component");
  ros::NodeHandle virgola_nh("~");
  ros::NodeHandle vacio_nh;

  std::cout << "vacio: " << vacio_nh.getNamespace() << "\n";
  std::cout << "virgola: " << virgola_nh.getNamespace() << "\n";

  rcomponent::RComponent component(vacio_nh);
  std::cout << "public: " << component.getPublicNamespace() << "\n";
  std::cout << "private: " << component.getPrivateNamespace() << "\n";

  component = rcomponent::RComponent(virgola_nh, "javi");
  std::cout << "public: " << component.getPublicNamespace() << "\n";
  std::cout << "private: " << component.getPrivateNamespace() << "\n";
  return 0;
}
