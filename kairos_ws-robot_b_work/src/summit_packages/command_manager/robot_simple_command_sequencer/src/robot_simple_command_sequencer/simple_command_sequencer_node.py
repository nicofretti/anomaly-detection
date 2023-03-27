#!/usr/bin/env python

import rospy
from simple_command_sequencer import SimpleCommandSequencer


def main():

    rospy.init_node("command_sequencer_node")

    rc_node = SimpleCommandSequencer()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
