#!/usr/bin/env python

import rospy

from leds_driver import LedsDriver


def main():

    rospy.init_node("leds_driver_node")

    rc_node = LedsDriver()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
