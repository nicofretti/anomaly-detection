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

import rospy,roslib
import rospkg

import time, threading
import os
import os.path
import glob
import signal
import exceptions
import errno
import datetime
import commands
import re

import psutil

from robotnik_msgs.msg import State
from rosbag_manager_msgs.srv import Record, RecordRequest, RecordResponse
from rosbag_manager_msgs.msg import RosbagManagerStatus


DEFAULT_FREQ = 10.0
MAX_FREQ = 20.0

"""
Manage the execution of a new rosbag process
"""
class RosbagProcessManager():

    def __init__(self, params):

        self.rosbag_record_waitpid = -1
        self.buffer_size_ = params['buffer_size']
        self.chunk_size_ =  params['chunk_size']
        self.topics_ = params['topics']
        self.expanded_topics_ = list(self.topics_)
        self.split_ = params['split']
        self.split_size_ = params['split_size']
        self.compression_ = params['compression']
        self.regex_ = params['regex']
        self.autostart_ = params['autostart']
        self.max_hdd_percent_ = params['max_hdd_percent']

        # output_path should be folder_path, as it is not the file name
        self.output_folder_ = params['output_folder']
        self.output_filename_ = ''
        self.output_path_ = self.output_folder_

    def initialize(self):
        """
            Initialize
        """

        recordpath = roslib.packages.find_node('rosbag', 'record')

        #print recordpath

        if not recordpath:
            rospy.logerr("rosbag package not found")
            return -1

        self.command = recordpath[0]

        rospy.loginfo('RosbagProcessManager:initialize: command = %s'%(self.command))

        return 0

    def setOutputPath(self, folder, filename):
        """
            Saves the output path
        """
        self.output_filename_ = filename
        self.output_folder_ = folder
        if not self.output_folder_.endswith('/'):
            self.output_folder_ += '/'

        self.output_path_ = self.output_folder_ + self.output_filename_
        self.setCommandArguments()

    def getOutputPath(self):
        return self.output_path_

    def getOutputFolder(self):
        return self.output_folder_

    def setCommandArguments(self):
        """
            Sets the arguments used for rosbag
        """
        self.args = ()
        self.args = self.args + ('--buffsize', str(self.buffer_size_))
        self.args = self.args + ('--chunksize', str(self.chunk_size_))

        if self.split_:
            self.args = self.args + ('--split', )
            self.args = self.args + ('--size=%d'%self.split_size_,)
        if self.compression_:
            self.args = self.args + ('--bz2', )
        if self.regex_:
            self.args = self.args + ('--regex', )

        self.args = self.args + ('-O',  self.output_path_)

        self.expandTopics()

        for topic in self.expanded_topics_:
            self.args = self.args + (topic, )

    def expandTopics(self):
        """
            Expands topic names according to what is reported by the ROS Master
        """

        # add namespace if topics are not absolute
        qualified_topics = map(lambda topic : topic if topic.startswith('/') else rospy.get_namespace() + topic, self.topics_)

        # previously regex where expanded here, but rosbag record is able to handle them, so let him to do it
        self.expanded_topics_ = qualified_topics

    def runCommand(self):
        """
            Executes the command
        """
        #arsgtxt=''
        #for i in self.args:
        #    arsgtxt = arsgtxt + '|' + i
        #rospy.loginfo('RosbagProcessManager: runCommand: %s'%(self.command + ' '+ arsgtxt))
        self.rosbag_record_waitpid = os.fork()

        if self.rosbag_record_waitpid:

            rospy.loginfo('RosbagProcessManager: runCommand: parent: pid = %d'%self.rosbag_record_waitpid)
            return 0
            #os.waitpid(self.rosbag_record_waitpid, 0)
        else:
            #rospy.loginfo('RosbagProcessManager: runCommand : %s %s', self.command, self.args)
            os.execvp(self.command, (self.command,) + self.args)


        return -1


    def stopCommand(self):
        """
            Stops the process
        """
        if self.rosbag_record_waitpid:
            rospy.loginfo('RosbagProcessManager::stopCommand: Killing process %d'%self.rosbag_record_waitpid)
            os.kill(self.rosbag_record_waitpid, signal.SIGINT)
            #ret,status = os.waitpid(self.rosbag_record_waitpid, os.WNOHANG)
            #rospy.loginfo('RosbagProcessManager::stopCommand: ret = %d,  status = %d'%(ret, status))
            return 0
        else:
            rospy.loginfo('RosbagProcessManager::stopCommand: No process to kill')
            return -1



"""
Ros component to manage the execution of a rosbag process
"""
class RosBagManager:

    def __init__(self, args):

        self.node_name = rospy.get_name()#.replace('/','')
        self.desired_freq = args['desired_freq']
        # Checks value of freq
        if self.desired_freq <= 0.0 or self.desired_freq > MAX_FREQ:
            rospy.loginfo('%s::init: Desired freq (%f) is not possible. Setting desired_freq to %f'%(self.node_name,self.desired_freq, DEFAULT_FREQ))
            self.desired_freq = DEFAULT_FREQ

        self.process_manager = RosbagProcessManager(args)

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
        self.msg_status = RosbagManagerStatus()
        # Timer to publish state
        self.publish_status_timer = 1
        # Flag active when recording
        self.is_recording = False
        # Saves the moment of the initialization
        self.init_record_time = rospy.Time.now()
        # Saves the name of the bag file
        self.bag_name = ''
        # Saves the name of the bag path
        self.bag_path = args['output_folder']
        # if autostart == true start rosbat automatically
        self.autostart = args['autostart']
        # if % hdd used > max_hdd_percent stop rosbag 
        # if max_hdd_percent = 0 no check
        self.max_hdd_percent = args['max_hdd_percent']

        self.t_publish_status = threading.Timer(self.publish_status_timer, self.publishROSstate)


    def setup(self):
        '''
            Initializes de hand
            @return: True if OK, False otherwise
        '''
        if self.process_manager.initialize() == 0:
            self.initialized = True
        else:
            return -1

        return 0


    def rosSetup(self):
        '''
            Creates and inits ROS components
        '''
        if self.ros_initialized:
            return 0

        self._status_publisher = rospy.Publisher('~status', RosbagManagerStatus, queue_size=1)
        self._state_publisher = rospy.Publisher('~state', State, queue_size=1)

        # Services
        self._set_recording_service = rospy.Service('~set_recording', Record, self.setRecordingServiceCb)


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

        if self.is_recording:
            self.process_manager.stopCommand()
            self.is_recording = False

        self.rosShutdown()

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
        self.t_publish_status.cancel()
        self._status_publisher.unregister()
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

            elif self.state == State.STANDBY_STATE:
                self.standbyState()

            elif self.state == State.READY_STATE:
                self.readyState()

            elif self.state == State.EMERGENCY_STATE:
                self.emergencyState()

            elif self.state == State.FAILURE_STATE:
                self.failureState()

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
                    self.running = False

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

        return 0


    def initState(self):
        '''
            Actions performed in init state
        '''
        if not self.initialized:
            self.setup()

        else:
            self.switchToState(State.STANDBY_STATE)

        if (self.autostart):
            hdd = psutil.disk_usage('/')
            if self.max_hdd_percent>0:
                rospy.loginfo('RosbagManager: hdd used:%d percent, this node will store rosbags while the hdd used < %s percent '%(hdd.percent,self.max_hdd_percent))
                
                if hdd.percent>self.max_hdd_percent:
                    rospy.logerr('RosbagManager: hdd used:%d > max_hdd_percent defined:%s'%(hdd.percent,self.max_hdd_percent))
                    self.switchToState(State.SHUTDOWN_STATE)
                    return
            else:
                rospy.loginfo('RosbagManager: hdd used:%d this node will store rosbag with no limits'%(hdd.percent))
                
            req=RecordRequest()
            req.action='start'
            req.name='rosbag_%s'%self.createBagName();
            print (self.bag_path)
            req.path=self.bag_path+'/autorosbag_%s'%self.createBagName();
            self.setRecordingServiceCb(req)
            self.switchToState(State.READY_STATE)


        return


    def standbyState(self):
        '''
            Actions performed in standby state
        '''
        if self.is_recording:
            self.switchToState(State.READY_STATE)

        return


    def readyState(self):
        '''
            Actions performed in ready state
        '''
        if not self.is_recording:
            self.switchToState(State.STANDBY_STATE)

        if self.max_hdd_percent>0:
            hdd = psutil.disk_usage('/')
            if hdd.percent>self.max_hdd_percent:
                rospy.logerr('RosbagManager: hdd used:%d > max_hdd_percent defined:%s'%(hdd.percent,self.max_hdd_percent))
                self.switchToState(State.SHUTDOWN_STATE)
        return


    def shutdownState(self):
        '''
            Actions performed in shutdown state
        '''

        if (self.autostart):
            req=RecordRequest()
            req.action='stop'
            #req.name='rosbag_%s'%self.createBagName();
            #req.path='/tmp/autorosbag_%s'%self.createBagName();
            self.setRecordingServiceCb(req)
            self.switchToState(State.EMERGENCY_STATE)
            self.running = False
            return

        if self.shutdown() == 0:
            self.switchToState(State.INIT_STATE)

        return


    def emergencyState(self):
        '''
            Actions performed in emergency state
        '''

        return


    def failureState(self):
        '''
            Actions performed in failure state
        '''


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
        self.msg_status.header.stamp = rospy.Time.now()
        self.msg_status.state.state = self.state
        self.msg_status.state.state_description = self.stateToString(self.state)
        self.msg_status.state.desired_freq = self.desired_freq
        self.msg_status.state.real_freq = self.real_freq
        self.msg_status.recording = self.is_recording
        if self.is_recording:
            self.msg_status.time_recording = (rospy.Time.now() - self.init_record_time).to_sec()
            self.msg_status.stored_size = self.getBagSize(self.process_manager.output_path_) / (1024*1024) # convert to MiB

        self._status_publisher.publish(self.msg_status)
        self._state_publisher.publish(self.msg_status.state)

        self.t_publish_status = threading.Timer(self.publish_status_timer, self.publishROSstate)
        self.t_publish_status.start()

    def setRecordingServiceCb(self, req):
        '''
        ROS service to start/stop the rosbag recording
        @param req: Required action
        @type req: inelfe_msgs/Record.srv
        '''
        rospy.loginfo('RosbagManager:setRecorgingServiceCb: action %s, name = %s, path = %s'%(req.action, req.name, req.path))

        # Initializes rosbag process
        if not self.is_recording and req.action == RecordRequest.START:
            folder_path = req.path
            if folder_path == '':
                folder_path = self.process_manager.getOutputFolder()
            if not folder_path.endswith('/'):
                folder_path = folder_path + '/'

            folder_path = os.path.expanduser(os.path.expandvars(folder_path))
            if self.createDirectoryTree(folder_path)!= 0:
                rospy.logerr('RosbagManager:setRecorgingServiceCb: Error creating directory tree at %s'%folder_path)
                return False, "Error creating directory tree at %s"%(folder_path)
            self.bag_name = req.name
            if self.bag_name == '':
                self.bag_name = self.createBagName()
            #self.bag_name = "data" # Generic bag name
            self.bag_path = folder_path
            self.process_manager.setOutputPath(folder_path, self.bag_name)
            # Opens the file to save the info of the bag
            self.info_filename = folder_path+self.bag_name+'_info.yaml'
            self.createInfoFile(self.info_filename)

            if self.process_manager.runCommand() != 0:
                rospy.logerr('RosbagManager:setRecorgingServiceCb: Error executing the command')
                return False, 'Error running the rosbag process'

            self.init_record_time = rospy.Time.now()
            self.is_recording = True
            self.msg_status.path = self.bag_path
            self.msg_status.compression = self.process_manager.compression_
            self.msg_status.regex = self.process_manager.regex_

            self.msg_status.bag_name = self.bag_name
            self.info_file.writelines('ros_init_time: [%d,%d]\n'%(self.init_record_time.secs, self.init_record_time.nsecs))
            return True,"Recording initialized"

        elif self.is_recording and req.action in [RecordRequest.STOP, RecordRequest.DISCARD]:
            self.process_manager.stopCommand()
            time.sleep(1)
            end_time = rospy.Time.now()
            # Saves information in log file
            self.info_file.writelines(['ros_end_time: [%d,%d]\n'%(end_time.secs, end_time.nsecs), 'time_recording: %d\n'%(end_time - self.init_record_time).to_sec(),
            'size: %.4lf\n'%self.msg_status.stored_size, 'buffsize: %d\n'%self.process_manager.buffer_size_, 'chunksize: %d\n'%self.process_manager.chunk_size_,
            'topics: %s\n'%self.process_manager.topics_, 'path: %s\n'%self.bag_path, 'bag_files: %s\n'%self.getBagFiles(self.process_manager.output_path_),
            'compression: %s\n'%self.process_manager.compression_, 'regex: %s\n'%self.process_manager.regex_, 'split: %s\n'%self.process_manager.split_])
            if self.process_manager.split_:
                self.info_file.write('split_size: %d\n'%self.process_manager.split_size_)
            self.closeInfoFile()
            if req.action == RecordRequest.DISCARD:
                self.deleteFiles()
            self.is_recording = False
            return True, "Stop recording"

        elif self.is_recording and req.action == RecordRequest.START:
            return True, "Already recording"
        elif not self.is_recording and req.action in [RecordRequest.STOP, RecordRequest.DISCARD]:
            return True, "Nothing to do"

        return False,"Should never reach here"


    def createDirectoryTree(self, path):
        """
            Creates the directory tree to save the rosbag
            @param path as string
        """
        try:
            os.makedirs(path)
            return 0
        except exceptions.OSError, e:
            if e.errno != errno.EEXIST:
                rospy.logerr('RosBagManager:createDirectoryTree: %s'%(e))
                return -1
            else:
                return 0

    def deleteFiles(self):
        # this may fail if bag is still "active" when called (maybe when it is too big)"
        try:
            os.remove(self.info_filename)
            for bag_filename in self.getBagFilepaths(self.process_manager.output_path_):
                os.remove(bag_filename)
            return 0
        except exceptions.OSError, e:
            rospy.logerr('RosBagManager:deleteFiles: %s'%(e))
            return -1

    def createBagName(self):
        """
            Returns a bag filename based on datetime (2015-06-18_22_53_42)
        """
        d = str(datetime.datetime.now())
        d1_s = d.replace(' ', '_')
        d2_s = d1_s.replace(':', '_')
        d3_s = d2_s.split('.')[0]

        rospy.logwarn('createbagname %s'%d3_s)
        return d3_s

    def getBagSize(self, bag_file):
        """
            Returns the size of the current size in MB
            Returns -1 if error
        """
        files = self.getBagFilepaths(bag_file)
        return sum(map(os.path.getsize, files))

    def createInfoFile(self, path):
        """
            Creates the file to save the rosbag info
        """

        self.info_file = open(path, 'w')

        return 0

    def closeInfoFile(self):
        """
            Closes the info file
        """
        if hasattr(self, 'info_file'):
            self.info_file.close()
            return 0
        else:
            return -1

    def getBagFilepaths(self, bag_file):
        """
            Returns a list with all the bag files created, with full path
            Returns -1 if error
        """
        return glob.glob(bag_file + '*.bag*')

    def getBagFiles(self, bag_file):
        """
            Returns a list with all the bag files created
            Returns -1 if error
        """
        actual_files = map(os.path.basename, glob.glob(bag_file + '*.bag*'))
        noactive_files = map(lambda f : f[:-len('.active')] if f.endswith('.active') else f, actual_files)
        unique_files = list(set(noactive_files))
        return unique_files

def main():

    rospy.init_node("rosbag_node")


    _name = rospy.get_name().replace('/','')

    arg_defaults = {
      'topic_state': 'state',
      'topic_status': 'status',
      'desired_freq': DEFAULT_FREQ,
      'topics': '',
      'output_folder': rospkg.RosPack().get_path('rosbag_manager'),
      'buffer_size': 256,
      'chunk_size': 768,
      'split': False,   # Split the rosbag?
      'split_size': 1024, # size of splitted bags
      'compression': True, # Compress the bag file?
      'regex': True, # use regular expressions to subscribe to topics
      'autostart': False,
      'max_hdd_percent': 80, # If hdd % used > max_hdd_percent stop rosbag, if 0 no check
    }

    args = {}

    for name in arg_defaults:
        try:
            if rospy.search_param(name):
                args[name] = rospy.get_param('~%s'%(name)) # Adding the name of the node, because the para has the namespace of the node
            else:
                args[name] = arg_defaults[name]
                rospy.set_param('~%s'%(name), arg_defaults[name])

            #print name
        except rospy.ROSException, e:
            rospy.logerr('%s: %s'%(e, _name))


    rc_node = RosBagManager(args)

    rospy.loginfo('%s: starting'%(_name))

    rc_node.start()


if __name__ == "__main__":
    main()
