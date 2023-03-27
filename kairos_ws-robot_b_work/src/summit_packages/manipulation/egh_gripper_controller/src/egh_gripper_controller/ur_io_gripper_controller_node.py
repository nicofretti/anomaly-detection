#!/usr/bin/env python
import rospy

from controllers.ur_io_gripper_controller import URIOGripperController

def main():
    rospy.init_node("URIOGripperController")
    publisher = URIOGripperController()
    rospy.loginfo('%s: starting'%(rospy.get_name()))
    publisher.start()

if __name__ == "__main__":
    main()
    exit()
