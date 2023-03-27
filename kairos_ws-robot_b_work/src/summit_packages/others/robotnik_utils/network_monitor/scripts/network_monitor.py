#!/usr/bin/env python

import rospy
from ping_time_monitor.msg import PingStatus
import os
import rospkg

args = {}

def ping_status_callback(data):
    if data.packet_loss >= 1.0:
        rospy.logwarn("Lost some packets. Packet loss: %f", data.packet_loss)
        if data.packet_loss >= args['packet_loss_threshold']:
            rospy.logerr("Lost all packets. Restarting network interface")
            os.system('sudo ' + args['script'])

def main():
    rospy.init_node('network_monitor')

    rospack = rospkg.RosPack()

    arg_defaults = {
      'packet_loss_threshold': 100.0,
      'script': rospack.get_path('network_monitor') + '/scripts/restartNetworking.sh',
    }

    for name in arg_defaults:
        try:
            if rospy.search_param(name):
                args[name] = rospy.get_param('~%s'%name)
            else:
                args[name] = arg_defaults[name]
            #print name
        except rospy.ROSException, e:
            rospy.logerror('%s: %s'%(e, _name))

    rospy.Subscriber('/ping_time_monitor/status', PingStatus, ping_status_callback)

    rospy.spin()

if __name__ == '__main__':
    main()
