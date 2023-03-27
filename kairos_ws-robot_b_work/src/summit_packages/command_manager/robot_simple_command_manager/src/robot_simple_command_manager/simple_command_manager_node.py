#!/usr/bin/env python

import rospy
from simple_command_manager import SimpleCommandManager


def main():

    rospy.init_node("command_manager_node")

    rc_node = SimpleCommandManager()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
