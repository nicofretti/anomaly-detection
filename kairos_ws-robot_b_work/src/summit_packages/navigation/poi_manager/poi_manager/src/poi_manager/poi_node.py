#!/usr/bin/env python

import rospy
from poi import PoiManager


def main():

    rospy.init_node("poi_manager")

    rc_node = PoiManager()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
