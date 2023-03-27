#!/usr/bin/env python

# Software License Agreement (BSD License)
#
# Copyright (c) 2017, Robotnik Automation SLL
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

import time
import threading

from robotnik_msgs.msg import State
from ar_track_alvar_msgs.msg import AlvarMarkers, AlvarMarker
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import PoseArray, TransformStamped, PoseWithCovarianceStamped, Pose2D
from marker_mapping.srv import InitPoseFromFrame, InitPoseFromFrameRequest, InitPoseFromFrameResponse, SaveFrame, SetFrameId
from marker_mapping.msg import FrameMappingStatus, FrameStatus

from tf import TransformListener, Exception as tfException, ConnectivityException, LookupException, ExtrapolationException
import tf
from tf2_ros import TransformBroadcaster
import numpy
from tf import transformations

import rospkg
import os
import exceptions
import yaml
import math


DEFAULT_FREQ = 10.0
MAX_FREQ = 50.0
DEFAULT_FRAME_ID = 'map'
DEFAULT_BASE_FRAME_ID = 'base_link'
DEFAULT_MAX_FRAME_DISTANCE = 1.25  # max allowed distance to save markers and init pose
# max allowed orientation difference between the frame and the base_frame to save markers and init pose
DEFAULT_MAX_FRAME_ORIENTATION_DIFF = 0.790
DEFAULT_POSE_COVARIANCE = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]

# Class Template of Robotnik component for Pyhton


class FrameMapping:

    def __init__(self, args):

        self.node_name = rospy.get_name()  # .replace('/','')
        self.desired_freq = args['desired_freq']
        self._frame_id = args['frame_id']
        self._base_frame_id = args['base_frame_id']
        self._publish_saved_frames_tf = args['publish_saved_frames_tf']
        self._folder_path = args['folder_path']
        self._config_filename = args['config_filename']
        self._load_frames_on_init = args['load_frames_on_init']
        self._max_frame_distance = args['max_frame_distance']
        self._max_frame_orientation_diff = args['max_frame_orientation_diff']
        self._published_frame_prefix = args['published_frame_prefix']
        # The covariance set whenever setting the global pose
        self._default_pose_covariance = args['default_pose_covariance']
        self._init_pose_from_frame_service_name = '~'+args['init_pose_from_frame_service_name']

        self._max_tf_diff_time = 1.0

        # Checks value of freq
        if self.desired_freq <= 0.0 or self.desired_freq > MAX_FREQ:
            rospy.loginfo('%s::init: Desired freq (%f) is not possible. Setting desired_freq to %f' %
                          (self.node_name, self.desired_freq, DEFAULT_FREQ))
            self.desired_freq = DEFAULT_FREQ

        if self._max_frame_distance <= 0.0:
            rospy.loginfo('%s::init: max_frame_distance (%f) is not possible. Setting to default value %f' %
                          (self.node_name, self._max_frame_distance, DEFAULT_MAX_FRAME_DISTANCE))
            self._max_frame_distance = DEFAULT_MAX_FRAME_DISTANCE

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
        self._msg_status = FrameMappingStatus()
        self._msg_status.ids_detected = []
        self._msg_status.ids_recorded = []
        # Timer to publish state
        self.publish_state_timer = 1
        # Saves the time with
        self.last_marker_time = rospy.Time.now()
        # Saves the received markers
        self._marker_msg = AlvarMarkers()
        # Dict to save the markers
        self._frames_dict = {}
        # The covariance set whenever setting the global pose

        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)

    def setup(self):
        '''
            Initializes de hand
            @return: True if OK, False otherwise
        '''

        if self._load_frames_on_init:
            if self.loadFramesFromFile(self._folder_path, self._config_filename) != 0:
                rospy.loginfo("%s::setup: Error loading config file", self.node_name)

        # Create local folder if it does not exist
        if self.createDirectoryTree(self._folder_path) == 0:
            rospy.loginfo("%s::setup: Created config folder %s", self.node_name, self._folder_path)
            self.initialized = True
            return 0
        else:
            return -1

    def rosSetup(self):
        '''
            Creates and inits ROS components
        '''
        if self.ros_initialized:
            return 0

        self._transform_listener = TransformListener()
        self._transform_broadcaster = TransformBroadcaster()

        # Publishers
        self._status_publisher = rospy.Publisher('~status', FrameMappingStatus, queue_size=10)
        self._state_publisher = rospy.Publisher('~state', State, queue_size=10)
        self._initialpose_publisher = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
        self._pose_from_frame_array_publisher = rospy.Publisher('~pose_from_frames', PoseArray, queue_size=10)
        # Subscribers
        # topic_name, msg type, callback, queue_size
        #self.markers_subscriber = rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.markersCb, queue_size = 10)

        # Service Servers
        self.save_marker_service_server = rospy.Service('~save_frame', SaveFrame, self.saveFrameServiceCb)
        self.init_pose_from_marker_service_server = rospy.Service(
            self._init_pose_from_frame_service_name, InitPoseFromFrame, self.initPoseFromFrameServiceCb)
        self.init_pose_service_server = rospy.Service('~init_pose', Trigger, self.initPoseServiceCb)
        self.set_frame_id = rospy.Service('~set_frame_id', SetFrameId, self.setFrameIdServiceCb)
        # Service Clients
        # self.service_client = rospy.ServiceProxy('service_name', ServiceMsg)
        # ret = self.service_client.call(ServiceMsg)

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
        rospy.loginfo('%s::shutdown' % self.node_name)

        # Cancels current timers
        self.t_publish_state.cancel()

        self._status_publisher.unregister()
        self._state_publisher.unregister()
        self._initialpose_publisher.unregister()

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
        self._status_publisher.unregister()
        self._state_publisher.unregister()
        self._initialpose_publisher.unregister()

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
                    rospy.loginfo('%s::controlLoop: ROS interrupt exception' % self.node_name)
                    self.running = False

            t3 = time.time()
            self.real_freq = 1.0/(t3 - t1)

        self.running = False
        # Performs component shutdown
        self.shutdownState()
        # Performs ROS shutdown
        self.rosShutdown()
        rospy.loginfo('%s::controlLoop: exit control loop' % self.node_name)

        return 0

    def rosPublish(self):
        '''
            Publish topics at standard frequency
        '''
        t_now = rospy.Time.now()

        self._status_publisher.publish(self._msg_status)

        # Publish array of poses for visualization purposes
        if len(self._msg_status.ids_detected) > 0:
            msg = PoseArray()
            for frame_status in self._msg_status.ids_detected:
                msg.poses.append(frame_status.pose_from_frame)
            msg.header.stamp = t_now
            msg.header.frame_id = self._frame_id
            self._pose_from_frame_array_publisher.publish(msg)

        if self._publish_saved_frames_tf:

            for frame in self._frames_dict:
                # it only publishes tf if the frame was saved related to the same frame_id is set right know. It allows markers with different frames
                if self._frame_id == self._frames_dict[frame]['frame']:

                    transform = TransformStamped()
                    transform.header.stamp = t_now
                    transform.header.frame_id = self._frame_id
                    transform.child_frame_id = '%s%s' % (self._published_frame_prefix, self._frames_dict[frame]['id'])
                    transform.transform.translation.x = self._frames_dict[frame]['position'][0]
                    transform.transform.translation.y = self._frames_dict[frame]['position'][1]
                    transform.transform.translation.z = self._frames_dict[frame]['position'][2]
                    transform.transform.rotation.x = self._frames_dict[frame]['orientation'][0]
                    transform.transform.rotation.y = self._frames_dict[frame]['orientation'][1]
                    transform.transform.rotation.z = self._frames_dict[frame]['orientation'][2]
                    transform.transform.rotation.w = self._frames_dict[frame]['orientation'][3]
                    self._transform_broadcaster.sendTransform(transform)

                else:
                    rospy.logwarn_throttle(5, 'frame %s related to %s is not able to be published in the current frame %s' % (
                        self._frames_dict[frame]['id'], self._frames_dict[frame]['frame'], self._frame_id))

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

    def standbyState(self):
        '''
            Actions performed in standby state
        '''
        t_now = rospy.Time.now()

        # is detecting saved frames?
        if self.isDetectingRecordedFrames():
            self.switchToState(State.READY_STATE)

        return

    def readyState(self):
        '''
            Actions performed in ready state
        '''
        if self.isDetectingRecordedFrames() == False:
            self.switchToState(State.STANDBY_STATE)
        else:
            self.updateDetectedFramesStatus()

        return

    def shutdownState(self):
        '''
            Actions performed in shutdown state
        '''
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
            rospy.loginfo('%s::switchToState: %s' % (self.node_name, self.stateToString(self.state)))

        return

    def allState(self):
        '''
            Actions performed in all states
        '''
        self.rosPublish()

        return

    def stateToString(self, state):
        '''
            @param state: state to set
            @type state: State
            @returns the equivalent string of the state
        '''
        if state == State.INIT_STATE:
            return 'INIT_STATE'

        elif state == State.STANDBY_STATE:
            return 'STANDBY_STATE'

        elif state == State.READY_STATE:
            return 'READY_STATE'

        elif state == State.EMERGENCY_STATE:
            return 'EMERGENCY_STATE'

        elif state == State.FAILURE_STATE:
            return 'FAILURE_STATE'

        elif state == State.SHUTDOWN_STATE:
            return 'SHUTDOWN_STATE'
        else:
            return 'UNKNOWN_STATE'

    def publishROSstate(self):
        '''
            Publish the State of the component at the desired frequency
        '''
        msg_state = State()
        msg_state.state = self.state
        msg_state.state_description = self.stateToString(self.state)
        msg_state.desired_freq = self.desired_freq
        msg_state.real_freq = self.real_freq

        self._state_publisher.publish(msg_state)

        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)
        self.t_publish_state.start()

    def saveFrameServiceCb(self, req):
        '''
            ROS service server
            @param req: Required action
            @type req: marker_mapping/SaveFrame
        '''

        # 1- get the
        frame_id = req.frame_id

        if frame_id == '':
            return False, 'The param frame_id has to be specified'

        try:
            position, quaternion = self._transform_listener.lookupTransform(frame_id, self._base_frame_id, rospy.Time())

        except tf.Exception, e:
            rospy.logerr('%s::saveFrameServiceCb: %s' % (self.node_name, e))
            return False, str(e)
        except ConnectivityException, e:
            rospy.logerr('%s::saveFrameServiceCb: %s' % (self.node_name, e))
            return False, str(e)
        except LookupException, e:
            rospy.logerr('%s::saveFrameServiceCb: %s' % (self.node_name, e))
            return False, str(e)
        except ExtrapolationException, e:
            rospy.logerr('%s::saveFrameServiceCb: %s' % (self.node_name, e))
            return False, str(e)

        dist_to_frame = self.distanceToXY(position[0], position[1])
        #dist_to_frame = 0.0
        # TODO: calculate rotation difference

        if dist_to_frame > self._max_frame_distance:
            rospy.logerr('%s:saveFrameServiceCb: the distance (%.3lf) to the frame %s is greater than the maximum allowed %.3lf',
                         self.node_name, dist_to_frame, frame_id, self._max_frame_distance)
            return False, 'The distance to the frame %s is greater than the maximum allowed (%.3lf)' % (frame_id, self._max_frame_distance)

        if frame_id in self._frames_dict:
            rospy.logwarn('%s:saveFrameServiceCb: overwritting frame %s', self.node_name, frame_id)
        else:
            rospy.loginfo('%s:saveFrameServiceCb: saving NEW frame %s', self.node_name, frame_id)

        # get the transform from global_frame_id to the one we are going to save
        ret, position, quaternion = self.transformFrame(frame_id, self._frame_id)

        if ret == 0:
            self._frames_dict[frame_id] = {'id': frame_id, 'frame': self._frame_id, 'position': position, 'orientation': quaternion,
                                           'max_distance': self._max_frame_distance, 'pose_covariance': self._default_pose_covariance}
        else:
            rospy.logerr('%s:saveFrameServiceCb: error getting the transform %s -> %s',
                         self.node_name, self._frame_id, frame_id)
            return False, 'error getting the transform %s -> %s' % (self._frame_id, frame_id)

        print self._frames_dict

        # Saving to file
        self.saveFramesToFile(self._folder_path, self._config_filename)
        return True, 'ok'

    def initPoseServiceCb(self, req):
        '''
            ROS service server. Relocates the robot based on the reading frames and compared with saved ones
            @param req: Required action
            @type req: std_srv/Trigger
        '''
        request = InitPoseFromFrameRequest()
        #response,info = self.initPoseFromFrameServiceCb(request)

        return self.initPoseFromFrameServiceCb(request)

    def initPoseFromFrameServiceCb(self, req):
        '''
            ROS service server. Relocates the robot based on the detected frames and compared with saved ones
            @param req: Required action
            @type req: marker_mapping/InitPoseFromFrame
        '''
        #
        # only in ready
        if self.state == State.READY_STATE:
            selected_frame = None
            frames_to_compare_by_distance = []
            frame_out_of_range = False

            if req.frame_id != '':
                if self.isDetectingFrame(req.frame_id):
                    for frame_status in self._msg_status.ids_detected:
                        if frame_status.id == req.frame_id:

                            dist_to_frame = frame_status.current_distance

                            if dist_to_frame > self._frames_dict[frame_status.id]['max_distance']:
                                frame_out_of_range = True
                                rospy.logwarn('%s:initPoseFromFrameServiceCb: the distance (%.3lf) to frame %s is greater than the maximum allowed %.3lf',
                                              self.node_name, dist_to_frame, frame_status.id, self._frames_dict[frame_status.id]['max_distance'])
                            else:
                                selected_frame = frame_status
                else:
                    return False, "Requested frame (%s) is no detected" % req.frame_id
            else:
                # array of [frame, distance_to_base] to save all the detected ones

                # look for the closest detected markers
                for frame_status in self._msg_status.ids_detected:
                    # if the marker is in our list
                    dist_to_frame = frame_status.current_distance

                    if dist_to_frame > self._frames_dict[frame_status.id]['max_distance']:
                        frame_out_of_range = True
                        rospy.logwarn('%s:initPoseFromFrameServiceCb: the distance (%.3lf) to frame %s is greater than the maximum allowed %.3lf',
                                      self.node_name, dist_to_frame, frame_status.id, self._frames_dict[frame_status.id]['max_distance'])
                    else:
                        #rospy.logwarn('%s:initPoseFromFrameServiceCb: the distance (%.3lf) to frame %s is lower than the maximum allowed %.3lf',self.node_name,dist_to_frame, frame_status.id, self._frames_dict[frame_status.id]['max_distance'])

                        frames_to_compare_by_distance.append(frame_status)

            #################################
            min_distance = 999999999

            for i in frames_to_compare_by_distance:
                if i.current_distance < min_distance:
                    selected_frame = i
                    min_distance = i.current_distance

            if selected_frame == None:
                if frame_out_of_range == True:
                    return False, "Detected frames are out of allowed range"
                else:
                    return False, "No valid frames were found to initialize the pose"
            else:
                rospy.logwarn('%s:initPoseFromFrameServiceCb: using frame %s to set the position',
                              self.node_name, selected_frame.id)

                # Set the new pose to AMCL
                msg = PoseWithCovarianceStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self._frame_id
                msg.pose.covariance = self._frames_dict[selected_frame.id]['pose_covariance']
                msg.pose.pose.position.x = selected_frame.pose_from_frame.position.x
                msg.pose.pose.position.y = selected_frame.pose_from_frame.position.y
                msg.pose.pose.position.z = selected_frame.pose_from_frame.position.z
                msg.pose.pose.orientation.x = selected_frame.pose_from_frame.orientation.x
                msg.pose.pose.orientation.y = selected_frame.pose_from_frame.orientation.y
                msg.pose.pose.orientation.z = selected_frame.pose_from_frame.orientation.z
                msg.pose.pose.orientation.w = selected_frame.pose_from_frame.orientation.w
                self._initialpose_publisher.publish(msg)

                return True, "Pose initialized based on frame %s" % selected_frame.id

        else:
            rospy.logerr('%s::initPoseFromFrameServiceCb: not in ready state', self.node_name)
            return False, "Not in READY state"

    def transformFrame(self, source_frame, target_frame):
        '''
            Transform a frame into the desired one
            @param source_frame as string, frame_id to transform
            @param target_frame as string, new parent frame
            @return 0, position, quaternion if ok
            @retunr -1, [], [] if error
        '''
        # sets the header to pose attribute (since it's not present)
        # marker.pose.header = marker.header
        try:
            position, quaternion = self._transform_listener.lookupTransform(target_frame, source_frame, rospy.Time())
            rospy.loginfo('%s::transformFrame: transform from %s to %s [%s - %s]',
                          self.node_name, source_frame, target_frame, str(position), str(quaternion))

            return 0, position, quaternion
        except tfException, e:
            rospy.logerr('%s::transformFrame: %s' % (self.node_name, e))
            return -1, [], []
        except ConnectivityException, e:
            rospy.logerr('%s::transformFrame: %s' % (self.node_name, e))
            return -1, [], []
        except LookupException, e:
            rospy.logerr('%s::transformFrame: %s' % (self.node_name, e))
            return -1, [], []
        except ExtrapolationException, e:
            rospy.logerr('%s::transformFrame: %s' % (self.node_name, e))
            return -1, [], []

        return -1, [], []

    def createDirectoryTree(self, path):
        """
            Creates the directory tree to save the logs
            @param path as string
        """
        try:
            os.makedirs(path)
            return 0
        except exceptions.OSError, e:
            if e.errno != 17:
                rospy.logerr('%s:createDirectoryTree: %s' % (self.node_name, e))
                return -1
            else:
                return 0

    def saveFramesToFile(self, folder, filename):
        """
            Creates the file markers config
        """
        file_path = folder+'/'+filename
        try:
            rospy.loginfo('%s::saveFramesToFile: opening file %s', self.node_name, file_path)
            config_file = file(file_path, 'w')
        except exceptions.OSError, e:
            rospy.logerr('%s:saveFramesToFile: %s' % (self.node_name, e))
            return -1

        self._msg_status.ids_recorded = []

        for frame_id in self._frames_dict:
            rospy.loginfo('%s::saveFramesToFile: Saving frame %s', self.node_name, frame_id)
            config_file.writelines(['- id: %s\n' % (self._frames_dict[frame_id]['id']),
                                    '  frame: %s\n' % (self._frames_dict[frame_id]['frame'])])
            config_file.writelines(['  position: %s\n' % (str(self._frames_dict[frame_id]['position'])),
                                    '  orientation: %s\n' % (str(self._frames_dict[frame_id]['orientation'])),
                                    '  max_distance: %s\n' % (str(self._frames_dict[frame_id]['max_distance'])),
                                    '  pose_covariance: %s\n' % (str(self._frames_dict[frame_id]['pose_covariance']))])
            self._msg_status.ids_recorded.append(str(frame_id))
        config_file.close()

        return 0

    def loadFramesFromFile(self, folder, filename):
        """
            Load markers from the config files
        """
        file_path = folder+'/'+filename
        try:
            with open(file_path, 'r') as stream:

                rospy.loginfo('%s::loadFramesFromFile: opening file %s', self.node_name, file_path)

                config_yaml = yaml.load(stream)
                # clear the current list
                self._frames_dict.clear()
                #print config_yaml

                for frame_in_config in config_yaml:
                    # TODO: check that the yaml format is ok
                    # inserting the new marker
                    self._frames_dict[str(frame_in_config['id'])] = frame_in_config
                    self._msg_status.ids_recorded.append(str(frame_in_config['id']))

        except yaml.YAMLError, e:
            rospy.logerr('%s:loadFramesFromFile: error parsing yaml file %s. %s' % (self.node_name,  file_path, e))
            return -1
        except IOError, e:
            rospy.logerr('%s:loadFramesFromFile: error parsing yaml file %s. %s' % (self.node_name,  file_path, e))
            return -1
        except KeyError, e:
            self._frames_dict.clear()
            rospy.logerr('%s:loadFramesFromFile: error parsing yaml file %s. %s' % (self.node_name,  file_path, e))
            return -1
        except IndexError, e:
            self._frames_dict.clear()
            rospy.logerr('%s:loadFramesFromFile: error parsing yaml file %s. %s' % (self.node_name,  file_path, e))
            return -1
        except TypeError, e:
            self._frames_dict.clear()
            rospy.logerr('%s:loadFramesFromFile: error parsing yaml file %s. %s' % (self.node_name,  file_path, e))
            return -1

        rospy.loginfo('%s::loadFramesFromFile: yaml loaded successfully', self.node_name)

        #print self._frames_dict

        return 0

    def distanceToXY(self, x, y):
        '''
            Calculates the distance to a point x,y
        '''
        return math.sqrt(x*x + y*y)

    def setFrameIdServiceCb(self, req):
        '''
           Callback to set the frame_id param
        '''

        rospy.loginfo('%s::setFrameIdServiceCb: setting frame from %s to %s',
                      self.node_name, self._frame_id, req.frame_id)

        self._frame_id = req.frame_id

        return True, "Frame %s set correctly" % req.frame_id

    def isDetectingRecordedFrames(self):
        '''
            Returns true if it's detecting those frames -> the transform exists
        '''
        self._msg_status.ids_detected = []
        for frame in self._frames_dict:
            if self._frames_dict[frame]['frame'] == self._frame_id:
                try:
                    t = self._transform_listener.getLatestCommonTime(self._base_frame_id, frame)
                    if (rospy.Time.now() - t).to_sec() <= self._max_tf_diff_time:
                        frame_status = FrameStatus()
                        frame_status.id = str(frame)
                        self._msg_status.ids_detected.append(frame_status)
                except tfException:
                    pass
        if len(self._msg_status.ids_detected) > 0:
            return True
        return False

    def isDetectingFrame(self, frame):
        '''
            Returns true if is detecting the frame passed
            @param frame as string
        '''
        for frame_status in self._msg_status.ids_detected:
            if frame_status.id == frame:
                return True

        return False

    def updateDetectedFramesStatus(self):
        '''
            Updates the status of the detected frames
            @returns 0 if ok, -1 otherwise

        '''
        for frame_status in self._msg_status.ids_detected:
            # if the marker is in our list
            try:
                position, quaternion = self._transform_listener.lookupTransform(
                    self._base_frame_id, frame_status.id, rospy.Time())

                dist_to_frame = self.distanceToXY(position[0], position[1])
                frame_status.current_distance = dist_to_frame
                frame_status.max_distance = self._frames_dict[frame_status.id]['max_distance']

                # Creates a transform matrix from base_frame_id to the current marker
                translation_base_to_marker = position
                rotation_base_to_marker = quaternion
                t_mat44_base_to_marker = self._transform_listener.fromTranslationRotation(
                    translation_base_to_marker, rotation_base_to_marker)

                t_mat44_marker_to_base = numpy.linalg.inv(t_mat44_base_to_marker)
                translation_marker_to_base = tuple(transformations.translation_from_matrix(t_mat44_marker_to_base))[:3]
                rotation_marker_to_base = tuple(transformations.quaternion_from_matrix(t_mat44_marker_to_base))

                # Creates a transform matrix from frame_id (map) to the previously saved marker
                saved_frame = self._frames_dict[frame_status.id]
                translation_map_to_saved_marker = saved_frame['position']
                rotation_map_to_saved_marker = saved_frame['orientation']
                t_mat44_map_to_saved_marker = self._transform_listener.fromTranslationRotation(
                    translation_map_to_saved_marker, rotation_map_to_saved_marker)

                # Calculates the transform from map to base based on the marker
                t_mat44_map_to_base = numpy.dot(t_mat44_map_to_saved_marker, t_mat44_marker_to_base)
                translation_map_to_base = tuple(transformations.translation_from_matrix(t_mat44_map_to_base))[:3]
                rotation_map_to_base = tuple(transformations.quaternion_from_matrix(t_mat44_map_to_base))

                # Set the new pose to AMCL
                frame_status.pose_from_frame.position.x = translation_map_to_base[0]
                frame_status.pose_from_frame.position.y = translation_map_to_base[1]
                frame_status.pose_from_frame.position.z = translation_map_to_base[2]
                frame_status.pose_from_frame.orientation.x = rotation_map_to_base[0]
                frame_status.pose_from_frame.orientation.y = rotation_map_to_base[1]
                frame_status.pose_from_frame.orientation.z = rotation_map_to_base[2]
                frame_status.pose_from_frame.orientation.w = rotation_map_to_base[3]

            except tf.Exception, e:
                rospy.logerr('%s::updateDetectedFramesStatus: %s' % (self.node_name, e))
            except ConnectivityException, e:
                rospy.logerr('%s::updateDetectedFramesStatus: %s' % (self.node_name, e))
            except LookupException, e:
                rospy.logerr('%s::updateDetectedFramesStatus: %s' % (self.node_name, e))
            except ExtrapolationException, e:
                rospy.logerr('%s::updateDetectedFramesStatus: %s' % (self.node_name, e))

        self._msg_status.header.stamp = rospy.Time.now()
        self._msg_status.header.frame_id = self._frame_id

        return 0


def main():

    rospy.init_node("frame_mapping_node")

    _name = rospy.get_name().replace('/', '')

    rp = rospkg.RosPack()

    arg_defaults = {
        'topic_state': 'state',
        'desired_freq': DEFAULT_FREQ,
        'frame_id': DEFAULT_FRAME_ID,
        'base_frame_id': DEFAULT_BASE_FRAME_ID,
        'publish_saved_frames_tf': True,
        'folder_path': os.path.join(rp.get_path('marker_mapping'), 'config'),
        'config_filename': 'frames.yaml',
        'load_frames_on_init': True,
        'max_frame_distance': DEFAULT_MAX_FRAME_DISTANCE,    # used to filter by distance to marker
        'published_frame_prefix': 'mm_',
        # in radians. Max difference allowed to save and init poses
        'max_frame_orientation_diff': DEFAULT_MAX_FRAME_ORIENTATION_DIFF,
        'default_pose_covariance': DEFAULT_POSE_COVARIANCE,  # in radians. Max difference allowed to save and init poses
        'init_pose_from_frame_service_name': 'init_pose_from_frame'
    }

    args = {}

    for name in arg_defaults:
        try:
            if rospy.search_param(name):
                # Adding the name of the node, because the para has the namespace of the node
                args[name] = rospy.get_param('~%s' % (name))
            else:
                args[name] = arg_defaults[name]
            #print name
        except rospy.ROSException, e:
            rospy.logerr('%s: %s' % (e, _name))

    rc_node = FrameMapping(args)

    rospy.loginfo('%s: starting' % (_name))

    rc_node.start()


if __name__ == "__main__":
    main()
