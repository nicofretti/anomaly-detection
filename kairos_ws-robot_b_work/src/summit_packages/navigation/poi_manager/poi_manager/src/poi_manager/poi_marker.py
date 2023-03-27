#!/usr/bin/env python

"""
Copyright (c) 2017, Robotnik Automation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import roslib; roslib.load_manifest("interactive_markers")
import rospy, rospkg
import copy
import os
import math

from interactive_markers.interactive_marker_server import *
from visualization_msgs.msg import InteractiveMarker, Marker, InteractiveMarkerControl
from interactive_markers.menu_handler import *
from move_base_msgs.msg import *

import actionlib
from actionlib_msgs.msg import GoalStatus, GoalID
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped, Pose
from std_srvs.srv import Empty, Trigger, SetBool,SetBoolRequest,SetBoolResponse
from poi_manager_msgs.srv import *
from poi_manager_msgs.msg import *
from std_msgs.msg import Header
from poi_manager_msgs.msg import PoiState
from robotnik_msgs.msg import State
from robot_local_control_msgs.msg import LocalizationStatus
from tf import TransformListener


#frame_id = 'map'

# Client based on ActionServer to send goals to the purepursuit node
class MoveBaseClient():


	def __init__(self, planner_name):
		self.planner_name = planner_name
		# Creates the SimpleActionClient, passing the type of the action
		# (GoTo) to the constructor.
		self.client = actionlib.SimpleActionClient(planner_name, MoveBaseAction)

	## @brief Sends the goal to
	## @param goal_pose as geometry_msgs/PoseStamped
	## @return 0 if OK, -1 if no server, -2 if it's tracking a goal at the moment
	def goTo(self, goal_pose):
		# Waits until the action server has started up and started
		# listening for goals.
		if self.client.wait_for_server(timeout = rospy.Duration(3.0) ):
			goal = MoveBaseGoal()
			#set goal
			goal.target_pose = goal_pose
			self.client.send_goal(goal)
			return 0
		else:
			rospy.logerr('%s::MoveBaseClient:goTo: Error waiting for server %s', rospy.get_name(), self.planner_name)
			return -1

	## @brief cancel the current goal
	def cancel(self):
		rospy.logwarn('%s::MoveBaseClient:cancel: cancelling the goal', rospy.get_name())
		self.client.cancel_goal()

	## @brief Get the state information for this goal
    ##
    ## Possible States Are: PENDING, ACTIVE, RECALLED, REJECTED,
    ## PREEMPTED, ABORTED, SUCCEEDED, LOST.
    ##
    ## @return The goal's state. Returns LOST if this
    ## SimpleActionClient isn't tracking a goal.
	def getState(self):
		return self.client.get_state()

	## @brief Returns ret if OK, otherwise -1
	def getResult(self):
		ret = self.client.get_result()
		if not ret:
			return -1

		else:
			return ret
	def wait(self):
		return self.client.wait_for_result()

# Client based on ActionServer to send goals to the purepursuit node
class InitPoseClient():

	def __init__(self, topic_name):
		self.topic_name = topic_name
		# Creates a ROS publisher
		self.client = rospy.Publisher(topic_name, PoseWithCovarianceStamped, queue_size=10)

	## @brief Sends the pose
	## @param goal_pose as geometry_msgs/PoseStamped
	## @return 0 if OK, -1 if no server, -2 if it's tracking a goal at the moment
	def setPose(self, pose):
		rospy.loginfo('%s::InitPoseClient:setPose: setting pose', rospy.get_name())
		self.client.publish(pose)

		return
		self.client.publish(pose)

		return

class PointPath(InteractiveMarker):

    def __init__(self, name, description, frame_id, is_manager = False, is_editable = False):
        InteractiveMarker.__init__(self)

        marker_scale_x = rospy.get_param('~marker_scale_x', 0.5)
        marker_scale_y = rospy.get_param('~marker_scale_y', 0.15)
        marker_scale_z = rospy.get_param('~marker_scale_z', 0.15)

        self.header.frame_id = frame_id
        self.name = name
        self.description = 'POI: %s'%description
        #marker arrow
        self.marker = Marker()

        self.marker.type = Marker.ARROW
        self.marker.scale.x = marker_scale_x
        self.marker.scale.y = marker_scale_y
        self.marker.scale.z = marker_scale_z
        #self.marker.pose.position.z = 0.05

        ##Text of markers
        control = InteractiveMarkerControl()
        control.always_visible = True
        marker_name = Marker()
        marker_name.type = Marker.TEXT_VIEW_FACING

        if is_manager:
            #arrow color
            self.marker.color.r = 0.8
            self.marker.color.g = 0.0
            self.marker.color.b = 0.0
            self.marker.color.a = 0.5
            self.marker.scale.x = 1
            self.marker.scale.y = 0.2
            self.marker.scale.z = 0.2
            #interactive marker move_plane
            self.marker_move_control = InteractiveMarkerControl()
            self.marker_move_control.always_visible = True
            self.marker_move_control.orientation.w = 1
            self.marker_move_control.orientation.x = 0
            self.marker_move_control.orientation.y = 1
            self.marker_move_control.orientation.z = 0
            self.marker_move_control.name = "move_plane"
            self.marker_move_control.markers.append( self.marker )
            self.marker_move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            self.controls.append( self.marker_move_control )

            #interactive marker rotate_z
            self.marker_rotate_control = InteractiveMarkerControl()
            self.marker_rotate_control.orientation.w = 1
            self.marker_rotate_control.orientation.x = 0
            self.marker_rotate_control.orientation.y = 1
            self.marker_rotate_control.orientation.z = 0
            self.marker_rotate_control.name = "rotate_z"
            self.marker_rotate_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS

            self.controls.append( self.marker_rotate_control )

            #control of text marker
            control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
        elif is_editable:
    		#arrow color
    		self.marker.color.r = 0.8
    		self.marker.color.g = 0.8
    		self.marker.color.b = 0.0
    		self.marker.color.a = 0.75
    		self.marker.scale.x = 1
    		self.marker.scale.y = 0.2
    		self.marker.scale.z = 0.2
    		#interactive marker move_plane
    		self.marker_move_control = InteractiveMarkerControl()
    		self.marker_move_control.always_visible = True
    		self.marker_move_control.orientation.w = 1
    		self.marker_move_control.orientation.x = 0
    		self.marker_move_control.orientation.y = 1
    		self.marker_move_control.orientation.z = 0
    		self.marker_move_control.name = "move_plane"
    		self.marker_move_control.markers.append( self.marker )
    		self.marker_move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE

    		self.controls.append( self.marker_move_control )
    		#interactive marker rotate_z
    		self.marker_rotate_control = InteractiveMarkerControl()
    		self.marker_rotate_control.orientation.w = 1
    		self.marker_rotate_control.orientation.x = 0
    		self.marker_rotate_control.orientation.y = 1
    		self.marker_rotate_control.orientation.z = 0
    		self.marker_rotate_control.name = "rotate_z"
    		self.marker_rotate_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS

    		self.controls.append( self.marker_rotate_control )

    		#control of text marker
    		control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
    		control.orientation.w = 1
    		control.orientation.x = 0
    		control.orientation.y = 1
    		control.orientation.z = 0
        else:
    		self.marker.color.r = 0.0
    		self.marker.color.g = 0.7
    		self.marker.color.b = 0.0
    		self.marker.color.a = 0.5
    		control.interaction_mode = InteractiveMarkerControl.FIXED

        self.marker_control = InteractiveMarkerControl()
        self.marker_control.name = 'menu'
        self.marker_control.always_visible = True
        self.marker_control.orientation.w = 1
        self.marker_control.orientation.x = 0
        self.marker_control.orientation.y = 1
        self.marker_control.orientation.z = 0
        self.marker_control.markers.append( self.marker )
        self.marker_control.interaction_mode = InteractiveMarkerControl.MENU
        self.controls.append( self.marker_control )

        #color of text
        marker_name.text = name
        marker_name.color.a = 0.95
        marker_name.scale.x = 0.35
        marker_name.scale.y = 0.1
        marker_name.scale.z = 0.1
        marker_name.pose.position.x	=0
        marker_name.pose.position.y	=0
        marker_name.pose.position.z	=0.2
        control.markers.append(marker_name)
        self.controls.append(control)

    ## @brief method called every time that an interaction is received
    def processFeedback(self, feedback):
        self.pose = feedback.pose

## @brief Manages the creation of waypoints and how to send them to Purepursuit
class PointPathManager(InteractiveMarkerServer):

  def __init__(self, name, args):
    InteractiveMarkerServer.__init__(self, 'poi_interactive_marker')
    self.list_of_points = []
    self.frame_id = args['frame_id']
    self.base_frame_id = args['base_frame_id']
    self.counter_points = 0
    self.init_pose_topic_name = args['init_pose_topic_name']
    self.goto_planner_action_name = args['goto_planner']
    self.load_pois_service_name = args['load_pois_service_name']
    self.add_poi_service_name = args['add_poi_service_name']
    self.delete_poi_service_name = args['delete_poi_service_name']
    self.save_pois_service_name = args['save_pois_service_name']
    self.delete_all_pois_service_name = args['delete_all_pois_service_name']
    self.rlc_localization_status_topic_name = args['rlc_localization_status_topic_name']
    self.node_name = rospy.get_name()

    self.robot_environment = ""

    # Menu handler to create a menu
    self.initMenuHandlers()

    self.rosSetup()


  def initMenuHandlers(self):
    self.menu_handler = MenuHandler()
    h_first_entry = self.menu_handler.insert( "POI" )
    self.entry_new = self.menu_handler.insert( "New", parent=h_first_entry, callback=self.createNewPOI )
    self.entry_new_from_robot_pose = self.menu_handler.insert( "Current pose", parent=h_first_entry, callback=self.createNewPOIFromRobotPose )
    self.entry_edit = self.menu_handler.insert( "Edit", parent=h_first_entry, callback=self.editPOI ) #IF this order changes the SAVE Callback must be changed
    self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
    self.entry_delete = self.menu_handler.insert( "Delete", parent=h_first_entry, callback=self.deletePOI )

    h_second_entry = self.menu_handler.insert( "Navigation" )
    entry = self.menu_handler.insert( "Go", parent=h_second_entry, callback=self.gotoPOI)	# Send the path from the first point to the last one
    entry = self.menu_handler.insert( "Stop", parent=h_second_entry, callback=self.stop)	# Stops the current path
    self.h_loc_entry = self.menu_handler.insert( "Localization" )
    entry = self.menu_handler.insert( "Init Pose", parent=self.h_loc_entry, callback=self.setInitialPose)


    # The covariance set whenever setting the global pose
    self._default_pose_covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]

    self.h_fourth_entry = self.menu_handler.insert( "ALL POIs" )
    entry = self.menu_handler.insert( "Delete", parent=self.h_fourth_entry, callback=self.deleteAllPOIs )
    # entry = self.menu_handler.insert( "Load from yaml", parent=self.h_fourth_entry, callback=self.loadPOIs )
    # entry = self.menu_handler.insert( "Save to yaml", parent=self.h_fourth_entry, callback=self.savePOIs )

    # Creates the first point (manager)
    self.initial_point = PointPath('POIManager', 'POIManager', frame_id = self.frame_id, is_manager=True)
    self.insert(self.initial_point, self.initial_point.processFeedback)

    ##if is_manager menu Edit and remove have not any sense
    self.menu_handler.setVisible(self.entry_edit, False)
    self.menu_handler.setVisible(self.entry_delete, False)
    self.menu_handler.apply( self, self.initial_point.name )
    self.applyChanges()

  def newPOIfromPose2D(self, elem, name, is_editable=False):
    new_point = PointPath(name, name, frame_id = self.frame_id)
    new_point.pose.position.x = elem.pose.x
    new_point.pose.position.y = elem.pose.y
    new_point.pose.position.z = 0.2
    new_point.pose.orientation.x = 0
    new_point.pose.orientation.y = 0
    new_point.pose.orientation.z = math.sin(elem.pose.theta*0.5)
    new_point.pose.orientation.w = math.cos(elem.pose.theta*0.5)
    self.list_of_points.append(new_point)
    self.insert(new_point, new_point.processFeedback)
    self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
    self.menu_handler.setVisible(self.entry_new, False)
    self.menu_handler.setVisible(self.entry_edit, True)
    self.menu_handler.setVisible(self.entry_delete, True)
    self.menu_handler.setVisible(self.h_fourth_entry, False)
    self.menu_handler.setVisible(self.entry_new_from_robot_pose, False)
    self.menu_handler.apply( self, new_point.name )
    self.counter_points = self.counter_points + 1

  def newPOIfromPose3D(self, elem, name, is_editable=False):
    new_point = PointPath(name, name, frame_id = self.frame_id)
    new_point.pose.position.x = elem.position.x
    new_point.pose.position.y = elem.position.y
    #new_point.pose.position.z = elem.position.z
    new_point.pose.position.z = 0.2
    new_point.pose.orientation.x = elem.orientation.x
    new_point.pose.orientation.y = elem.orientation.y
    new_point.pose.orientation.z = elem.orientation.z
    new_point.pose.orientation.w = elem.orientation.w
    self.list_of_points.append(new_point)
    self.insert(new_point, new_point.processFeedback)
    self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
    self.menu_handler.setVisible(self.entry_new, False)
    self.menu_handler.setVisible(self.entry_edit, True)
    self.menu_handler.setVisible(self.entry_delete, True)
    self.menu_handler.setVisible(self.h_fourth_entry, False)
    self.menu_handler.setVisible(self.entry_new_from_robot_pose, False)
    self.menu_handler.apply( self, new_point.name )
    self.counter_points = self.counter_points + 1


  def newPOIfromPose(self, pose, name, is_editable=True):
    new_point = PointPath(name, name, frame_id = self.frame_id, is_manager = False, is_editable = is_editable)
    rospy.loginfo("%s::newPOIfromPose: adding %s",rospy.get_name(),name)
    #rospy.loginfo("%s::newPOIfromPose: adding: %s",rospy.get_name(),str(pose))
    new_point.pose.position.x = pose.position.x
    new_point.pose.position.y = pose.position.y
    #new_point.pose.position.z = pose.position.z
    new_point.pose.position.z = 0.2
    new_point.pose.orientation.x = pose.orientation.x
    new_point.pose.orientation.y = pose.orientation.y
    new_point.pose.orientation.z = pose.orientation.z
    new_point.pose.orientation.w = pose.orientation.w
    #rospy.loginfo("%s::newPOIfromPose: new point: %s",rospy.get_name(),str(new_point))

    self.list_of_points.append(new_point)
    self.insert(new_point, new_point.processFeedback)
    if not is_editable:
      self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
    self.menu_handler.setVisible(self.entry_new, False)
    self.menu_handler.setVisible(self.entry_edit, True)
    self.menu_handler.setVisible(self.entry_delete, True)
    self.menu_handler.setVisible(self.h_fourth_entry, False)
    self.menu_handler.setVisible(self.entry_new_from_robot_pose, False)
    self.menu_handler.apply( self, new_point.name )
    self.counter_points = self.counter_points + 1
    return new_point

  def save_poi_service(self,name,frame,pose):
    if self.robot_environment == "":
      return False," No environment selected"
    try:
      resp = rospy.ServiceProxy(self.add_poi_service_name , AddPOI)
      p = LabeledPose()
      p.name = name
      p.environment = self.robot_environment
      p.frame_id = frame
      p.pose = pose

      res = resp(p)

      if (res.success):
        return True,res.message
      else:
        return False,res.message
    except rospy.ServiceException, e:
      msg = "Service call to %s failed: %s" % (self.add_poi_service_name,e)
      rospy.logerr('%s::save_poi_service: %s',rospy.get_name(), msg)
      return False,msg


  ## @brief Callback called to create new POI from Menu
  def createNewPOI(self, feedback):
    new_point = self.newPOIfromPose(feedback.pose, 'p%d'%(self.counter_points), is_editable=False)


    rospy.loginfo("%s::addPOI: %s, environment: %s" ,self.node_name, self.add_poi_service_name, self.robot_environment)

    success,msg=self.save_poi_service(new_point.name,new_point.header.frame_id,new_point.pose)
    if success == False:
        rospy.logerr('%s::createNewPOI: Error calling save_poi_service -> %s', rospy.get_name(), msg)

    self.applyChanges()

  def createNewPOIFromRobotPose(self, feedback):
    current_pose = self.getCurrentPose()
    robot_pose = Pose()
    if(not current_pose[0]):
      rospy.logerror("%s::addPOI: %s ,environment: Error: %s" ,self.node_name, self.add_poi_service_name, current_pose[1])
      return
    robot_pose.position.x = current_pose[2][0]
    robot_pose.position.y = current_pose[2][1]
    robot_pose.position.z = 0.2
    robot_pose.orientation.x = current_pose[3][0]
    robot_pose.orientation.y = current_pose[3][1]
    robot_pose.orientation.z = current_pose[3][2]
    robot_pose.orientation.w = current_pose[3][3]
    new_point = self.newPOIfromPose(robot_pose, 'p%d'%(self.counter_points), is_editable=False)
    rospy.loginfo("%s::addPOI: %s ,environment: %s" ,self.node_name, self.add_poi_service_name,self.robot_environment)

    success,msg=self.save_poi_service(new_point.name,new_point.header.frame_id,new_point.pose)

    #TODO: error feedback
    self.applyChanges()

  ## @brief Callback called to reload from yaml
  def loadPOIs(self, feedback):
    rospy.loginfo("%s::loadPOIs: Before load there are %d pois", rospy.get_name(), len(self.list_of_points))
    # Deletes all the loaded points
    if len(self.list_of_points) >= 0:
      self.deleteAllPOIs(feedback)
    try:
      resp = rospy.ServiceProxy(self.load_pois_service_name, ReadPOIs)
      poi_list = resp()
      rospy.loginfo(str(poi_list))
    except rospy.ServiceException, e:
      rospy.logerr("%s::loadPOIs: Service call failed: %s",rospy.get_name(),e)
      return
    #create	the POis
    for elem in poi_list.pose_list:
      self.newPOIfromPose2D(elem, elem.label, frame_id)
    self.applyChanges()

    rospy.loginfo("%s::loadPOIs: After load there are %d pois", rospy.get_name(), len(self.list_of_points))
    self.counter_points = self.counter_points + 1

  ## @brief Callback called to delete all
  def deleteAllPOIs(self, feedback):
    rospy.loginfo("%s::deleteAllPOIs %d",rospy.get_name(), len(self.list_of_points))
    for i in range(0,len(self.list_of_points)):
      p=self.list_of_points.pop()
      self.erase(p.name)
      self.counter_points = self.counter_points - 1
      self.applyChanges()

  ## @brief Callback called to delete point
  def deletePOI(self, feedback):
    rospy.loginfo("%s::deletePOI",rospy.get_name())
    if len(self.list_of_points) > 0:
      for i in self.list_of_points:
        if i.name==feedback.marker_name:
          self.list_of_points.remove(i)
          self.erase(i.name)
          #self.counter_points = self.counter_points - 1
          self.delete_poi_from_poi_manager(i.name)
          break
      self.applyChanges()
      #~ #call update_pois service
      #~ self.update_pois()

  #save POIs to yaml file
  def savePOIs(self, feedback):
    rospy.loginfo("%s::savePOIs: saving %d points", rospy.get_name(), len(self.list_of_points))
    ##if this marker is editable, make it no-editable
    if(self.menu_handler.getCheckState( 2 ) == MenuHandler.CHECKED): #2 is the menu_entry_id of Edit
      self.menu_handler.setCheckState(2, MenuHandler.UNCHECKED )
      #delete the editable POI
      for i in self.list_of_points:
        if i.name==feedback.marker_name:
          self.deletePOI(feedback)
          print("found!")
          break
      #create Noeditable POI
      self.newPOIfromPose(i.pose, i.name, is_editable=False)
      self.applyChanges()

    try:
      update_pois = rospy.ServiceProxy(self.save_pois_service_name, UpdatePOIs)
    except rospy.ServiceException, e:
      rospy.logerr("%s::savePOIs: Service call failed: %s",rospy.get_name(),e)
      return

    newPOIs = []
    for i in self.list_of_points:
      pose = Pose2D(i.pose.position.x, i.pose.position.y, 2*math.asin(i.pose.orientation.z))
      newPOIs.append(LabeledPose(i.name,pose))
    try:
      resp = update_pois(newPOIs)
    except rospy.ServiceException as exc:
      rospy.logerr("%s::savePOIs: Error: %s", rospy.get_name(), str(exc))

  def editPOI(self, feedback):
    rospy.loginfo("%s::editPOI: %s menu:%s"%(rospy.get_name(),feedback.marker_name,feedback.menu_entry_id))
    if self.counter_points > 0:
      handle = feedback.menu_entry_id
      state = self.menu_handler.getCheckState( handle )
      #check if is already editing
      if state == MenuHandler.CHECKED:
        self.menu_handler.setCheckState( handle, MenuHandler.UNCHECKED )
        #delete the editable POI
        for i in self.list_of_points:
          if i.name==feedback.marker_name:
            self.deletePOI(feedback)
            break
        #create Noeditable POI
        success,msg=self.save_poi_service(i.name,i.header.frame_id,i.pose)
        self.newPOIfromPose(i.pose, i.name, is_editable=False)


        self.applyChanges()
      else:
        self.menu_handler.setCheckState( handle, MenuHandler.CHECKED )
        self.pose = feedback.pose
        #delete the POI
        for i in self.list_of_points:
          if i.name==feedback.marker_name:
            self.deletePOI(feedback)
            break
        #create the POI now editable
        self.newPOIfromPose(i.pose, i.name, is_editable=True)
        self.applyChanges()

  ## @brief Starts the route
  def gotoPOI(self, feedback):

    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id =  self.frame_id
    goal.pose.position.x = feedback.pose.position.x
    goal.pose.position.y = feedback.pose.position.y

    goal.pose.orientation.x= feedback.pose.orientation.x
    goal.pose.orientation.y = feedback.pose.orientation.y
    goal.pose.orientation.z = feedback.pose.orientation.z
    goal.pose.orientation.w = feedback.pose.orientation.w

    self.planner_client.goTo(goal)
    rospy.loginfo('%s::gotoPOI: Sending goal',rospy.get_name())

    self.switchToAction(PoiState.GOTO)

    return

  ## @brief Starts the route
  def setInitialPose(self, feedback):
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = self.frame_id
    msg.pose.covariance = self._default_pose_covariance
    msg.pose.pose.position.x = feedback.pose.position.x
    msg.pose.pose.position.y = feedback.pose.position.y
    msg.pose.pose.position.z = 0.2
    msg.pose.pose.orientation.x = feedback.pose.orientation.x
    msg.pose.pose.orientation.y = feedback.pose.orientation.y
    msg.pose.pose.orientation.z = feedback.pose.orientation.z
    msg.pose.pose.orientation.w = feedback.pose.orientation.w

    self.init_pose_client.setPose(msg)
    rospy.loginfo('%s::setPoseCB: setting pose',rospy.get_name())
    return

  ## @brief Stops the current route if it's started
  def stop(self, feedback):
    self.planner_client.cancel()
    return

  def rlcLocalizationStatusCb(self, msg):
    if msg.state == "READY":
        if self.robot_environment != msg.environment:
          rospy.loginfo('%s::rlcLocalizationStatusCb: The current environment %s has changed %s. Frame %s -> %s',rospy.get_name(),self.robot_environment, msg.environment, self.frame_id, msg.pose.header.frame_id)
          self.robot_environment = msg.environment

          self.frame_id = msg.pose.header.frame_id

          #rospy.loginfo('%s::rlcLocalizationStatusCb: Initializing the master POI',rospy.get_name())
          # Removes and creates the manager
          self.erase(self.initial_point.name)
          self.initial_point = PointPath('POIManager', 'POIManager', frame_id = self.frame_id, is_manager=True)
          self.insert(self.initial_point, self.initial_point.processFeedback)
          self.menu_handler.setVisible(self.entry_new, True)
          self.menu_handler.setVisible(self.entry_edit, False)
          self.menu_handler.setVisible(self.entry_delete, False)
          self.menu_handler.setVisible(self.entry_new_from_robot_pose, True)
          self.menu_handler.apply( self, self.initial_point.name )
          self.applyChanges()

          req = SetBoolRequest()
          req.data=True
          resp = self.serviceLoadPOIs(req)
          if resp[0] == False:
            rospy.logerr("%s::rlcLocalizationStatusCb: %s", self.node_name, resp[1])
    else:
        if self.robot_environment != '':
            rospy.logwarn('%s::rlcLocalizationStatusCb: localization status (%s) is not ready, unsetting current enviroment',rospy.get_name(), msg.state)
            self.robot_environment = ''
            req = SetBoolRequest()
            req.data=True
            resp = self.serviceLoadPOIs(req)
            if resp[0] == False:
              rospy.logerr("%s::rlcLocalizationStatusCb: %s", self.node_name, resp[1])

  def saveRobotPoseServiceCb(self, msg):
    current_pose = self.getCurrentPose()
    robot_pose = Pose()
    if(not current_pose[0]):
      rospy.logerror("%s::saveRobotPoseServiceCb: Error getting current pose: %s" ,self.node_name, current_pose[1])
      return
    robot_pose.position.x = current_pose[2][0]
    robot_pose.position.y = current_pose[2][1]
    robot_pose.position.z = 0.2
    robot_pose.orientation.x = current_pose[3][0]
    robot_pose.orientation.y = current_pose[3][1]
    robot_pose.orientation.z = current_pose[3][2]
    robot_pose.orientation.w = current_pose[3][3]

    rospy.loginfo('%s::saveRobotPoseServiceCb',rospy.get_name())
    new_point = self.newPOIfromPose(robot_pose, 'p%d'%(self.counter_points), is_editable=False)

    success,msg=self.save_poi_service(new_point.name,new_point.header.frame_id,new_point.pose)

    self.applyChanges()

    return True, "OK"



  def rosSetup(self):
    '''
      ROS components setup
    '''
    # publishers
    self.state_publisher = rospy.Publisher('~state', PoiState, queue_size=10)
    # subscribers
    self.rlc_localization_status_subscriber = rospy.Subscriber(self.rlc_localization_status_topic_name, LocalizationStatus, self.rlcLocalizationStatusCb)

    # service servers
    self.load_poi_tag_service_server = rospy.Service('~load_pois', SetBool, self.serviceLoadPOIs)
    self.delete_all_poi_tag_service_server = rospy.Service('~delete_all_pois', SetBool, self.serviceDeleteAllPOIs)

    self.service_get_current_pose_service = rospy.Service('~get_current_pose', GetPoseTrigger, self.getCurrentPoseCB)

    self.service_get_current_pose_service = rospy.Service('~delete_poi', DeletePOI, self.deletePoiCB)

    #self.add_pois_2d_tag_service_server = rospy.Service('~add_pois_2d', UpdatePOIs, self.serviceAddPOIsfromPose2D)
    #self.init_pose_2d_tag_service_server = rospy.Service('~inti_pose_2d', UpdatePOIs, self.ServiceInitialPose2D)
    self.stop_tag_service_server = rospy.Service('~stop_goto', SetBool, self.serviceStop)
    self.service_server = rospy.Service('~save_robot_pose', Trigger, self.saveRobotPoseServiceCb)

    self.tf_transform_listener = TransformListener()

    # Action clients
    self.planner_client = MoveBaseClient(planner_name=self.goto_planner_action_name)

    self.init_pose_client = InitPoseClient(self.init_pose_topic_name)
    self._state = PoiState()

    self._pose_received_time = rospy.Time(0)

    return


  def controlLoop(self):
    '''
      Control loop of the component
    '''
    self.state_publisher.publish(self._state)

    if self._state.action == PoiState.GOTO:
      if self.planner_client.getState() != GoalStatus.ACTIVE:
        self.switchToAction(PoiState.IDLE)



    return

  def switchToState(self, new_state):
    '''
      Performs the change of state
    '''
    if self._state.state.state != new_state:

      self._state.state.state = new_state
      self._state.state.state_description = self.stateToString(self._state.state.state)
      rospy.loginfo('%s::switchToState: %s',self.node_name,self._state.state.state_description)

    return


  def switchToAction(self, new_action):
    '''
      Performs the change of action
    '''
    if self._state.action != new_action:

      self._state.action = new_action

      rospy.loginfo('%s::switchToAction: %s',self.node_name, self._state.action)

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

  def getCurrentPoseCB(self, req):
    resp = GetPoseTriggerResponse()
    current_pose = self.getCurrentPose()
    resp.success = current_pose[0]
    resp.message = current_pose[1]
    if(resp.success):
        print(str(current_pose[2]))
        print(str(current_pose[3]))
        resp.pose.position.x = current_pose[2][0]
        resp.pose.position.y = current_pose[2][1]
        resp.pose.position.z = 0.2
        resp.pose.orientation.x = current_pose[3][0]
        resp.pose.orientation.y = current_pose[3][1]
        resp.pose.orientation.z = current_pose[3][2]
        resp.pose.orientation.w = current_pose[3][3]
    return resp



  def getCurrentPose(self):

    try:
        self.tf_transform_listener.waitForTransform(self.frame_id, self.base_frame_id, rospy.Time(), rospy.Duration(1.0))
        position, quaternion = self.tf_transform_listener.lookupTransform(self.frame_id, self.base_frame_id, rospy.Time(0))
        #print(str(position))
        #print(str(quaternion))
        return True, ("Transform between " + self.base_frame_id + " -> " + self.frame_id), position, quaternion
    except Exception as e:
        return False, ("Error to transform between " + self.base_frame_id + " -> " + self.frame_id + " : " + str(e)), None, None



  def goToTagserviceCb(self, req):
    '''
      callback for the ros service go to tag
    '''
    if self._state.action != PoiState.GOTO:

      for poi in self.list_of_points:

        if req.name == poi.name:
          goal = PoseStamped()
          goal.header.stamp = rospy.Time.now()
          goal.header.frame_id = self.frame_id
          goal.pose.position.x = poi.pose.position.x
          goal.pose.position.y = poi.pose.position.y

          goal.pose.orientation.x= poi.pose.orientation.x
          goal.pose.orientation.y = poi.pose.orientation.y
          goal.pose.orientation.z = poi.pose.orientation.z
          goal.pose.orientation.w = poi.pose.orientation.w
          self.planner_client.goTo(goal)
          rospy.loginfo('%s::goToTagserviceCb: Sending pose', self.node_name)
          self.switchToAction(PoiState.GOTO)

          return True,'OK'

      return False,'Tag %s not found'%req.name

    else:
      error_message = 'Robot doing another action.'
      error_message = error_message + "State in "  + str(self._state.action)
      return False, error_message

  # @brief Stops the current route if it's started. If req.data == true it try stop and return OK message or not else return false a msg.
  # @param req: Srv type SetBool, request- bool
  # @return Srv type SetBool, response- bool sucess, message: string
  def serviceStop (self, req):
    if(req.data==True):
      try:
        self.planner_client.cancel()
        return True,'OK'
      except:
        return False,'Exception'
    else:
      return False,'OK'

  def delete_environment_from_poi_manager(self):
    try:
      resp = rospy.ServiceProxy(self.delete_all_pois_service_name , DeleteEnvironment)
      res = resp(self.robot_environment)
      if (res.success):
        return True,res.message
      else:
        return False,res.message
    except rospy.ServiceException, e:
      msg = "%s::DeleteEnvironment: Service call failed: %s" % (rospy.get_name(),e)
      rospy.logerr(msg)
      return False,msg

  def deletePoiCB(self, req):
    rospy.loginfo("%s::deletePOI: %s",rospy.get_name(), req.name)
    resp = DeletePOIResponse()
    if len(self.list_of_points) > 0:
      for i in self.list_of_points:
        if i.name==req.name:
          if (self.delete_poi_from_poi_manager(i.name)[0]):
            self.list_of_points.remove(i)
            self.erase(i.name)
            #self.counter_points = self.counter_points - 1
            resp.success = True
            resp.message = "Deleted"
            break
    if(not resp.success):
      resp.message = "The point " + req.name +" doesn't exist"
    self.applyChanges()
    return resp

  def delete_poi_from_poi_manager(self,name):
    try:
      resp = rospy.ServiceProxy(self.delete_poi_service_name , DeletePOI)
      res = resp(name,self.robot_environment)
      if (res.success):
        return True,res.message
      else:
        return False,res.message
    except rospy.ServiceException, e:
      msg = "%s::DeletePoi: Service call failed: %s" % (rospy.get_name(),e)
      rospy.logerr(msg)
      return False,msg

  def serviceDeleteAllPOIs(self, req):
    if(req.data==True):
      try:
        rospy.loginfo("%s::deleteAllPOIs %d",self.node_name,len(self.list_of_points))
        for i in range(0,len(self.list_of_points)):
          p=self.list_of_points.pop()
          self.erase(p.name)
          self.counter_points = self.counter_points - 1
          self.applyChanges()
        self.delete_environment_from_poi_manager()
        return True,'OK'
      except:
        return False, 'Exception'
    else:
      return False,'OK'

  #def serviceAddPoint (self, req):


  # @brief Load the pois. If req.data == true it try charge the pois and return OK or not else return false a msg.
  # @param req: Srv type SetBool, request- bool
  # @return Srv type SetBool, response- bool sucess, message: string
  def serviceLoadPOIs(self, req):
    if(req.data == True):
      rospy.loginfo("%s::serviceLoadPOIs: there were %d pois before loading",self.node_name,len(self.list_of_points))
      rospy.loginfo("%s::serviceLoadPOIs: %s, environment: %s" ,self.node_name, self.load_pois_service_name,self.robot_environment)
      poi_list=[]

      if self.robot_environment != '':
          # call read_pois service
          try:
            rospy.wait_for_service(self.load_pois_service_name , timeout = 2)
          except rospy.ROSException, e:
            rospy.logerr("%s::serviceLoadPOIs: %s", self.node_name,e)
            return False,'Exception'
          try:
            resp = rospy.ServiceProxy(self.load_pois_service_name , GetPOIs)
            res = resp(self.robot_environment)
            if (res.success or len(res.p_list)==0):
              poi_list=res.p_list
              #print (poi_list,type(poi_list))
            else:
              #self.serviceDeleteAllPOIs(req)
              return False,res.message
          except rospy.ServiceException, e:
            rospy.logerr("%s::serviceLoadPOIs: Service call failed: %s",self.node_name,e)
            return False,'Exception'

      # Deletes all the visualized points
      for i in range(0,len(self.list_of_points)):
        p=self.list_of_points.pop()
        self.erase(p.name)
        self.counter_points = self.counter_points - 1
        self.applyChanges()
      #create	the POis
      for elem in poi_list:
        self.newPOIfromPose3D(elem.pose, elem.name, elem.frame_id)
      self.applyChanges()
      rospy.loginfo("%s::serviceLoadPOIs: there are %d pois after loading",self.node_name,len(self.list_of_points))
      return True,'OK'
    else:
      return False,'Input argument was set as false'

  def serviceSavePOIs(self, req):
    if(req.data == True):
      rospy.loginfo("%s::savePOIs", self.node_name)
      ##if this marker is editable, make it no-editable
      if(self.menu_handler.getCheckState( 2 ) == MenuHandler.CHECKED): #2 is the menu_entry_id of Edit
        self.menu_handler.setCheckState(2, MenuHandler.UNCHECKED )
        #create Noeditable POI
        self.newPOIfromPose(i, i.name, is_editable=False)
        self.applyChanges()
      ##call service update_pois
      try:
        rospy.wait_for_service(self.save_pois_service_name, timeout = 2)
      except rospy.ROSException, e:
        rospy.logerr("%s::savePOIs: %s", self.node_name,e)
        return False,'Exception'

      try:
        update_pois = rospy.ServiceProxy(self.save_pois_service_name, UpdatePOIs)
      except rospy.ServiceException, e:
        rospy.logerr("%s::savePOIs: Service call failed: %s",self.node_name,e)
        return False,'Exception'

      newPOIs = []
      for i in self.list_of_points:
        pose = Pose2D(i.pose.position.x, i.pose.position.y, 2*math.asin(i.pose.orientation.z))
        newPOIs.append(LabeledPose(i.name,pose))
      try:
        resp = update_pois(newPOIs)
      except rospy.ServiceException as exc:
        rospy.logerr("%s::savePOIs:: Service did not process request: %s",self.node_name,str(exc))
        return False,'Exception'
      return True,'OK'
    else:
      return False,'OK'


  def serviceAddPOIsfromPose2D(self, req):
    for poi in req.pose_list:
      name = poi.label
      new_point = PointPath(name, name, frame_id = self.frame_id)
      new_point.pose.position.x = poi.pose.x
      new_point.pose.position.y = poi.pose.y
      new_point.pose.orientation.x = 0
      new_point.pose.orientation.y = 0
      new_point.pose.orientation.z = math.sin(poi.pose.theta*0.5)
      new_point.pose.orientation.w = math.cos(poi.pose.theta*0.5)
      self.list_of_points.append(new_point)
      self.insert(new_point, new_point.processFeedback)
      self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
      self.menu_handler.setVisible(self.entry_new, False)
      self.menu_handler.setVisible(self.entry_edit, True)
      self.menu_handler.setVisible(self.entry_delete, True)
      self.menu_handler.setVisible(self.h_fourth_entry, False)
      self.menu_handler.apply( self, new_point.name )
      self.counter_points = self.counter_points + 1
    load = SetBool()
    load.data = True
    self.serviceSavePOIs (load)
    self.serviceLoadPOIs (load)
    return Empty()

  def ServiceInitialPose2D(self, req):
    poi = req.pose_list[0]
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = self.frame_id
    msg.pose.covariance = self._default_pose_covariance
    msg.pose.pose.position.x = poi.pose.x
    msg.pose.pose.position.y = poi.pose.y
    msg.pose.pose.position.z = 0.2
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = math.sin(poi.pose.theta*0.5)
    msg.pose.pose.orientation.w = math.cos(poi.pose.theta*0.5)


if __name__=="__main__":
	rospy.init_node("poi_markers")

	_name = rospy.get_name().replace('/','')

	arg_defaults = {
      'base_frame_id': 'robot_base_footprint',
	  'frame_id': 'robot_map',
	  'goto_planner': 'mb_avoidance/move_base',
	  'init_pose_topic_name': 'initialpose',
	  'load_pois_service_name': 'poi_manager/get_poi_list',
      'add_poi_service_name': 'poi_manager/add_poi',
      'delete_poi_service_name': 'poi_manager/delete_poi',
	  'save_pois_service_name': 'poi_manager/update_pois',
      'delete_all_pois_service_name': 'poi_manager/delete_environment',
      'rlc_localization_status_topic_name' : 'robot_local_control/LocalizationComponent/status'
	}

	args = {}



	for name in arg_defaults:
		try:
			if rospy.search_param(name):
				args[name] = rospy.get_param('~'+name) # Adding the name of the node, because the para has the namespace of the node
			else:
				args[name] = arg_defaults[name]
			#print name
		except rospy.ROSException, e:
			rospy.logerror('%s: %s'%(e, _name))
	#frame_id = args['frame_id']
	#TODO: the object should get the args dict and set them in the init, not this way
	server = PointPathManager(_name, args)
	t_sleep = 0.5
	running = True

	while not rospy.is_shutdown() and running:

		try:
			rospy.sleep(t_sleep)
			server.controlLoop()
		except rospy.exceptions.ROSInterruptException:
			rospy.loginfo('Main: ROS interrupt exception')
			running = False
