#!/usr/bin/env python

import rospy

from gazebo_leds_manager import LedsManager


def main():

    rospy.init_node("gazebo_leds_manager_node")

    rc_node = LedsManager()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
