#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2015, Robotnik Automation SLL
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

import time, threading, commands

from robotnik_msgs.msg import State
from ping_time_monitor.msg import PingStatus

DEFAULT_FREQ = 1.0
MIN_TIMEOUT = 0.1
MIN_COUNT = 1
MAX_FREQ = 100.0

    
# Class Template of Robotnik component for Pyhton
class PingTimeMonitor:
    
    def __init__(self, args):
        
        self.node_name = rospy.get_name().replace('/','')
        self.host_ = args['host']
        self.timeout_ = args['timeout']
        self.count_ = args['count']
        # desired freq has to be equal to the max time to get the ping result
        self.desired_freq = DEFAULT_FREQ
        
        
        if self.timeout_ < 0.0:
            rospy.loginfo('%s::init: timeout (%f) is not possible. Setting timeout to %f'%(self.node_name,self.timeout_, MIN_TIMEOUT))
            self.timeout_ = MIN_TIMEOUT
        if self.count_ <= 0:
            rospy.loginfo('%s::init: count (%f) is not possible. Setting count to %f'%(self.node_name,self.count_, MIN_COUNT))
            self.count_ = MIN_COUNT
        # Checks value of freq
        if self.desired_freq <= 0.0 or self.desired_freq > MAX_FREQ:
            rospy.loginfo('%s::init: Desired freq (%f) is not possible. Setting desired_freq to %f'%(self.node_name,self.desired_freq, MAX_FREQ))
            self.desired_freq = MAX_FREQ
    
        # desired freq has to be equal to the max time to get the ping result
        self.desired_freq = max(0.00001, 1.0/(self.timeout_ * self.count_))
        self.real_freq = 0.0
        
        # Saves the state of the component
        self.state = State.INIT_STATE
        # Saves the previous state
        self.previous_state = State.INIT_STATE
        # flag to control the initialization of the component
        self.initialized = False
        # flag to control the initialization of ROS stuff
        self.ros_initialized = False
        # flag to control that the control loop is running
        self.running = False
        # Variable used to control the loop frequency
        self.time_sleep = 1.0 / self.desired_freq
        # State msg to publish
        self.msg_state = State()
        # Timer to publish state
        self.publish_state_timer = 1
        
        self.ping_status = PingStatus()
        self.ping_status.count = self.count_
        self.ping_status.timeout = self.timeout_
        self.ping_status.host = self.host_
        
        
        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)
        
            
    def setup(self):
        '''
            Initializes de hand
            @return: True if OK, False otherwise
        '''
        self.initialized = True
        
        return 0
        
        
    def rosSetup(self):
        '''
            Creates and inits ROS components
        '''
        if self.ros_initialized:
            return 0
        
        self._state_publisher = rospy.Publisher('~state', State, queue_size=10)
        self._ping_status_publisher = rospy.Publisher('~status', PingStatus, queue_size=10)
    
    
        self.ros_initialized = True
        
        self.publishROSstate()
        
        return 0
        
        
    def shutdown(self):
        '''
            Shutdowns device
            @return: 0 if it's performed successfully, -1 if there's any problem or the component is running
        '''
        if self.running or not self.initialized:
            return -1
        rospy.loginfo('%s::shutdown'%self.node_name)
        
        # Cancels current timers
        self.t_publish_state.cancel()
        
        self._state_publisher.unregister()
        
        self.initialized = False
        
        return 0
    
    
    def rosShutdown(self):
        '''
            Shutdows all ROS components
            @return: 0 if it's performed successfully, -1 if there's any problem or the component is running
        '''
        if self.running or not self.ros_initialized:
            return -1
        
        # Performs ROS topics & services shutdown
        self._state_publisher.unregister()
        
        self.ros_initialized = False
        
        return 0
            
    
    def stop(self):
        '''
            Creates and inits ROS components
        '''
        self.running = False
        
        return 0
    
    
    def start(self):
        '''
            Runs ROS configuration and the main control loop
            @return: 0 if OK
        '''
        self.rosSetup()
        
        if self.running:
            return 0
            
        self.running = True
        
        self.controlLoop()
        
        return 0
    
    
    def controlLoop(self):
        '''
            Main loop of the component
            Manages actions by state
        '''
        
        while self.running and not rospy.is_shutdown():
            t1 = time.time()
            
            if self.state == State.INIT_STATE:
                self.initState()
                
            elif self.state == State.READY_STATE:
                self.readyState()
                
            elif self.state == State.SHUTDOWN_STATE:
                self.shutdownState()
                
            self.allState()
            
            t2 = time.time()
            tdiff = (t2 - t1)
            
            
            t_sleep = self.time_sleep - tdiff
            
            if t_sleep > 0.0:
    			try:
    				rospy.sleep(t_sleep)
    			except rospy.exceptions.ROSInterruptException:
    				rospy.loginfo('%s::controlLoop: ROS interrupt exception'%self.node_name)
    				self._running = False
            
            t3= time.time()
            self.real_freq = 1.0/(t3 - t1)
        
        self.running = False
        # Performs component shutdown
        self.shutdownState()
        # Performs ROS shutdown
        self.rosShutdown()
        rospy.loginfo('%s::controlLoop: exit control loop'%self.node_name)
        
        return 0
        
        
    def rosPublish(self):
        '''
            Publish topics at standard frequency
        '''
        
        
        
        self._ping_status_publisher.publish(self.ping_status)
        
        return 0
        
    
    def initState(self):
        '''
            Actions performed in init state
        '''
        
        if not self.initialized:
            self.setup()
            
        else:         
            self.switchToState(State.STANDBY_STATE)
        
        
        return
    
    
    
    def readyState(self):
        '''
            Actions performed in ready state
        '''
 
            
        
        return
        
    
    def shutdownState(self):
        '''
            Actions performed in shutdown state 
        '''
        if self.shutdown() == 0:
            self.switchToState(State.INIT_STATE)
        
        return
    
    
    def switchToState(self, new_state):
        '''
            Performs the change of state
        '''
        if self.state != new_state:
            self.previous_state = self.state
            self.state = new_state
            rospy.loginfo('%s::switchToState: %s'%(self.node_name, self.stateToString(self.state)))
        
        return
    
        
    def allState(self):
        '''
            Actions performed in all states
        '''
        try:
            status, output = commands.getstatusoutput("ping -c %d -W %f %s"%(self.count_, self.timeout_, self.host_))
        except Exception,e:
            rospy.logerr('%s::readyState: error calling ping: %s',self.node_name, e)
            self.switchToState(State.FAILURE_STATE);
            return
            
        lines = output.splitlines()
        
        counter = 0
        total_time = 0.0
        
        self.ping_status.time = -1
        self.ping_status.header.stamp = rospy.Time.now()
        self.ping_status.t_min = -1
        self.ping_status.t_max = -1
        self.ping_status.t_avg = -1
        self.ping_status.t_mdev = -1
        self.ping_status.packet_loss = 100.0
            
        if status == 0:
            for l in lines:
                # finding total time
                ret = l.rfind('time=')
                if ret!= -1:
                    t = l[ret:]
                    t2 = float(t.rsplit('=')[1].strip(' ms'))
                    counter += 1
                    total_time += t2
                else:
                    # finding time stats
                    ret = l.rfind('rtt')
                    if ret!= -1:
                        
                        # 1.822/2.342/3.097/0.482 ms
                        t_stats = l.split(' = ')[1].split('/')
                        if len(t_stats) == 4:
                            self.ping_status.t_min = float(t_stats[0])
                            self.ping_status.t_avg = float(t_stats[1])
                            self.ping_status.t_max = float(t_stats[2])
                            self.ping_status.t_mdev = float(t_stats[3].split(' ms', 1)[0])
                        
                    else:
                        ret = l.rfind('packet loss')    
                        if ret!= -1:
                            packet_loss_line_splitted = l.split(',')
                            for p in packet_loss_line_splitted:
                                if p.rfind('packet')!= -1:
                                    self.ping_status.packet_loss = float(p.strip(' ').split(' ')[0].rstrip('%'))
                            
                    
            if counter >0:
                # total_time/=counter
                #print 'Time average = %.2lf'%total_time
                self.ping_status.time = total_time
                self.ping_status.header.stamp = rospy.Time.now()
                self.switchToState(State.READY_STATE);
            else:
                #print 'Error on ping response format'
                self.ping_status.time = -1
                self.ping_status.header.stamp = rospy.Time.now()
                self.switchToState(State.EMERGENCY_STATE);
        else:
			self.switchToState(State.EMERGENCY_STATE);
			
        self.rosPublish()
        
        return
    
    
    def stateToString(self, state):
        '''
            @param state: state to convert
            @type state: State
            @returns the equivalent string of the state
        '''
        if state == State.INIT_STATE:
            return 'INIT_STATE'
                
        elif self.state == State.STANDBY_STATE:
            return 'STANDBY_STATE'
            
        elif self.state == State.READY_STATE:
            return 'READY_STATE'
            
        elif self.state == State.EMERGENCY_STATE:
            return 'EMERGENCY_STATE'
            
        elif self.state == State.FAILURE_STATE:
            return 'FAILURE_STATE'
            
        elif self.state == State.SHUTDOWN_STATE:
            return 'SHUTDOWN_STATE'
        else:
            return 'UNKNOWN_STATE'
    
        
    def publishROSstate(self):
        '''
            Publish the State of the component at the desired frequency
        '''
        self.msg_state.state = self.state
        self.msg_state.state_description = self.stateToString(self.state)
        self.msg_state.desired_freq = self.desired_freq
        self.msg_state.real_freq = self.real_freq
        self._state_publisher.publish(self.msg_state)
        
        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)
        self.t_publish_state.start()
    
        
        
def main():

    rospy.init_node("ping_time_monitor_node")
    
    
    _name = rospy.get_name().replace('/','')
    
    arg_defaults = {
      'topic_state': 'state',
      'host': '127.0.0.1',
      'count': 4,
      'timeout': 1
    }
    
    args = {}
    
    for name in arg_defaults:
        try:
            if rospy.search_param(name): 
                args[name] = rospy.get_param('~%s'%name)
            else:
                args[name] = arg_defaults[name]
            #print name
        except rospy.ROSException, e:
            rospy.logerror('%s: %s'%(e, _name))
            
    
    rc_node = PingTimeMonitor(args)
    
    rospy.loginfo('%s: starting'%(_name))

    rc_node.start()


if __name__ == "__main__":
    main()
