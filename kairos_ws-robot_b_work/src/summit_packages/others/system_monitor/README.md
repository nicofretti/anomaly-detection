# system_monitor

## Synopsis

System monitoring tools for ROS.

**Author(s):** Willow Garage, Inc., Jerome Maye, Ralf Kaestner

**Maintainer:** Ralf Kaestner <ralf.kaestner@gmail.com>

**License:** BSD License (BSD)

**Operating system(s):** Debian-based Linux

**Package PPA:** ppa:ethz-asl/ros

## Description

This project provides system monitoring tools for ROS in the form of the
following ROS nodes:

* CPU monitor
* HDD monitor
* Memory monitor
* Network monitor
* NTP monitor

Each node publishes ROS diagnostics which can conveniently be visualized
in the runtime monitor.

## Installation

Download the repository in the src folder of your worksapce:
 
  ```
  git clone https://github.com/RobotnikAutomation/system_monitor    
  ```

Keep in mind that this package has some dependencies. If you do not have them, you will have to install them:

  ```
  sudo apt-get install sysstat ifstat ntpdate
  ```
