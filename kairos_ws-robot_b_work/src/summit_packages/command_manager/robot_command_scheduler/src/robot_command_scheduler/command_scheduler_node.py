#!/usr/bin/env python

import rospy
from command_scheduler import CommandScheduler


def main():

    rospy.init_node("command_scheduler_node")

    rc_node = CommandScheduler()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
