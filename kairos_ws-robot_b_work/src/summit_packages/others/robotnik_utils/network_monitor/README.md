Network Monitor
====================

Reads information from /ping_time_monitor/status to restart the network interfaces if there is no connection to a specified host

Configuration:
sudo is required to restart the network interfaces. To be able to run the script as sudo without being asked for the password, we have to edit the sudoers file:

sudo visudo: Under the line "%sudo   ALL=(ALL:ALL) ALL", add this:
  "USERNAME"  ALL=(ALL) NOPASSWD: "path_to_package"/network_monitor/scripts/restartNetworking.sh


Run:
roslaunch network_monitor network_monitor.launch
