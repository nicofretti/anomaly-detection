#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Robotnik Automation SLL
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Robotnik Automation SSL nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospy

# Class to check the health of a topic


class TopicHealthMonitor:
    def __init__(self, subscriber, timeout=5.0, required=True):
        '''
            @param subscriber as a rospy.Subscriber to check
            @param timeout as double. Timeout to consider that the topic is not receiving data anymore
            @param required as bool. Flag to include this topic when it checks the overall status of the topics
        '''
        self._timeout = timeout
        self._required = required
        self._last_msg_time = rospy.Time(0)
        self._subscriber = subscriber

    def tick(self):
        '''
            @brief Set the last message time
        '''
        self._last_msg_time = rospy.Time.now()

    def is_receiving(self):
        '''
            @param timeout as double. Timeout used to check if it is receiving data
            @return True if the topic is receiving data
        '''
        if self._subscriber == None or self._subscriber.get_num_connections() == 0:
            return False
        if (rospy.Time.now() - self._last_msg_time).to_sec() > self._timeout:
            return False

        return True

    def is_required(self):
        return self._required

    def get_subscriber(self):
        return self._subscriber

    def set_timeout(self, timeout):
        self._timeout = timeout

    def get_timeout(self):
        return self._timeout
