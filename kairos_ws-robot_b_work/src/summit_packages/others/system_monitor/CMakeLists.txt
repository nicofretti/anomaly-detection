cmake_minimum_required(VERSION 2.8.3)
project(system_monitor)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation)

add_message_files(
   FILES
   CoreTemp.msg
   CoreUsage.msg
   CPUTemperatureStatus.msg
   CPUUsageStatus.msg
   Diagnostic.msg
   DiagnosticCPUTemperature.msg
   DiagnosticCPUUsage.msg
   DiagnosticHDD.msg
   DiagnosticMEM.msg
   DiagnosticNET.msg
   Disk.msg
   HDDStatus.msg
   Interface.msg
   Memory.msg
   MEMStatus.msg
   NetStatus.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs  
   system_monitor # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS rospy message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   bin/cpu_monitor.py
   bin/hdd_monitor.py
   bin/mem_monitor.py
   bin/net_monitor.py
   bin/ntp_monitor.py
   bin/system_monitor_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(
  DIRECTORY bin config launch 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
