#!/usr/bin/env python

import rospy

from rcomponent import RComponent


def main():

    rospy.init_node("rcomponent_test")

    rc_node = RComponent()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
