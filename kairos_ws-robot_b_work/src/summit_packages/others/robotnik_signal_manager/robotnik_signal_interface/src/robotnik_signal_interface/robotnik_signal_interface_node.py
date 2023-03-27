#!/usr/bin/env python

import rospy

from robotnik_signal_interface import SignalInterface


def main():

    rospy.init_node("robotnik_signal_interface")

    rc_node = SignalInterface()

    rospy.loginfo('%s: starting' % (rospy.get_name()))

    rc_node.start()


if __name__ == "__main__":
    main()
