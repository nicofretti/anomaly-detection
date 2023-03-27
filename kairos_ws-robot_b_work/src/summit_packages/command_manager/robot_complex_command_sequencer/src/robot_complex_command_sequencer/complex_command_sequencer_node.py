#!/usr/bin/env python

import rospy
from complex_command_sequencer import ComplexCommandSequencer


def main():

    rospy.init_node("command_sequencer_node")

    rc_node = ComplexCommandSequencer()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
