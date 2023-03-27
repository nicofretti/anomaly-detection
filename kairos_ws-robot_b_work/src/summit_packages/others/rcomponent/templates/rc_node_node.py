#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from ?rc_node import ?RCNode


def main():

    rospy.init_node("?rc_node_node")

    rc_node = ?RCNode()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()