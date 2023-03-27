#!/usr/bin/env python

import rospy
from xmlrpc_controller import XMLRPCGripperController


def main():

    rospy.init_node("xmlrpc_gripper_controller_node")

    rc_node = XMLRPCGripperController()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
