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
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from std_srvs.srv import Empty
from poi_manager.srv import *
from poi_manager.msg import *

frame_id = 'map'

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
			#if self.getState() != GoalStatus.LOST:
			#	rospy.loginfo('PurepursuitClient: goto_planner is tracking a goal')
			#	return -2
				
			goal = MoveBaseGoal()
				
			#set goal
			goal.target_pose = goal_pose
			
			self.client.send_goal(goal)
			return 0
		else:
			rospy.logerr('MoveBaseClient: Error waiting for server')
			return -1
	
	## @brief cancel the current goal
	def cancel(self):		
		rospy.loginfo('MoveBaseClient: cancelling the goal')
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
		
		self.client.publish(pose)
		
		return

class PointPath(InteractiveMarker):
	
	def __init__(self, name, description, is_manager = False, is_editable = False):
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
		#print("interaction!")
		

## @brief Manages the creation of waypoints and how to send them to Purepursuit
class PointPathManager(InteractiveMarkerServer):
	
	def __init__(self, name, goto_planner, init_pose_topic_name):
		InteractiveMarkerServer.__init__(self, 'poi_interactive_marker')
		self.list_of_points = []
		self.frame_id = frame_id
		self.counter_points = 0

		# Menu handler to create a menu
		self.menu_handler = MenuHandler()
		h_first_entry = self.menu_handler.insert( "POI" )		
		self.entry_new = self.menu_handler.insert( "New", parent=h_first_entry, callback=self.createNewPOI )
		self.entry_edit = self.menu_handler.insert( "Edit", parent=h_first_entry, callback=self.editPOI ) #IF this order changes the SAVE Callback must be changed
		self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
		self.entry_delete = self.menu_handler.insert( "Delete", parent=h_first_entry, callback=self.deletePOI )
		
		h_second_entry = self.menu_handler.insert( "Navigation" )
		entry = self.menu_handler.insert( "Go", parent=h_second_entry, callback=self.gotoPOI)	# Send the path from the first point to the last one
		entry = self.menu_handler.insert( "Stop", parent=h_second_entry, callback=self.stop)	# Stops the current path 
		self.h_loc_entry = self.menu_handler.insert( "Localization" )
		entry = self.menu_handler.insert( "Init Pose", parent=self.h_loc_entry, callback=self.setInitialPose)
		
		#topics of goto_planner and initpose
		self.planner_client = MoveBaseClient(goto_planner)
		self.init_pose_client = InitPoseClient(init_pose_topic_name)
		# The covariance set whenever setting the global pose
		self._default_pose_covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]		
		
		self.h_fourth_entry = self.menu_handler.insert( "ALL POIs" )		
		entry = self.menu_handler.insert( "Delete", parent=self.h_fourth_entry, callback=self.deleteAllPOIs )
		entry = self.menu_handler.insert( "Load from yaml", parent=self.h_fourth_entry, callback=self.loadPOIs )
		entry = self.menu_handler.insert( "Save to yaml", parent=self.h_fourth_entry, callback=self.savePOIs )		
		
		# Creates the first point (manager)		
		self.initial_point = PointPath('POIManager', 'POIManager', is_manager=True)
		self.insert(self.initial_point, self.initial_point.processFeedback)						
		
		##if is_manager menu Edit and remove have not any sense			
		self.menu_handler.setVisible(self.entry_edit, False)
		self.menu_handler.setVisible(self.entry_delete, False)
		self.menu_handler.apply( self, self.initial_point.name )
		self.applyChanges()					
	
	def newPOIfromPose2D(self, elem, name, is_editable=False):
		new_point = PointPath(name, name)
		new_point.pose.position.x = elem.pose.x
		new_point.pose.position.y = elem.pose.y			
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
		self.menu_handler.apply( self, new_point.name )			
		self.counter_points = self.counter_points + 1			

	def newPOIfromPose(self, elem, name, is_editable=True):
		new_point = PointPath(name, name, False, is_editable)
		new_point.pose.position.x = elem.pose.position.x
		new_point.pose.position.y = elem.pose.position.y			
		new_point.pose.orientation.x = elem.pose.orientation.x
		new_point.pose.orientation.y = elem.pose.orientation.y
		new_point.pose.orientation.z = elem.pose.orientation.z
		new_point.pose.orientation.w = elem.pose.orientation.w
		self.list_of_points.append(new_point)		
		self.insert(new_point, new_point.processFeedback)		
		if not is_editable:
			self.menu_handler.setCheckState( self.entry_edit, MenuHandler.UNCHECKED )
		self.menu_handler.setVisible(self.entry_new, False)
		self.menu_handler.setVisible(self.entry_edit, True)
		self.menu_handler.setVisible(self.entry_delete, True)
		self.menu_handler.setVisible(self.h_fourth_entry, False)
		self.menu_handler.apply( self, new_point.name )			
		self.counter_points = self.counter_points + 1			
	
	## @brief Callback called to create new POI from Menu
	def createNewPOI(self, feedback):
		new_point = self.newPOIfromPose(feedback, 'p%d'%(self.counter_points), is_editable=False)
		self.applyChanges()
	
	## @brief Callback called to reload from yaml
	def loadPOIs(self, feedback):
		rospy.loginfo("Before load there are %d pois"%(len(self.list_of_points)))		
		# Deletes all the loaded points
		if len(self.list_of_points) > 0:
			self.deleteAllPOIs(feedback)				
		#~ ##call read_pois service
		rospy.wait_for_service('read_pois')
		try:
			resp = rospy.ServiceProxy('read_pois', ReadPOIs)
			poi_list = resp()		
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e
		#create	the POis	
		for elem in poi_list.pose_list:
			self.newPOIfromPose2D(elem, elem.label, frame_id)		
		self.applyChanges()			
		
		rospy.loginfo("After load there are %d pois"%(len(self.list_of_points)))					

	## @brief Callback called to delete all	
	def deleteAllPOIs(self, feedback):
		rospy.loginfo("DELETEALL %d"%(len(self.list_of_points)))		
		for i in range(0,len(self.list_of_points)):							
			p=self.list_of_points.pop()					
			self.erase(p.name)
			self.counter_points = self.counter_points - 1
			self.applyChanges()

	## @brief Callback called to delete point	
	def deletePOI(self, feedback):
		rospy.loginfo("DELETE")
		if self.counter_points > 0:						
			for i in self.list_of_points:				
				if i.name==feedback.marker_name:					
					self.list_of_points.remove(i)					
					self.erase(i.name)
					#self.counter_points = self.counter_points - 1					
					break
			self.applyChanges()
			#~ #call update_pois service
			#~ self.update_pois()
	
	#save POIs to yaml file
	def savePOIs(self, feedback):
		rospy.loginfo("SAVE")
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
			self.newPOIfromPose(i, i.name, is_editable=False)
			self.applyChanges()	
		##call service update_pois
		rospy.wait_for_service('update_pois')
		update_pois = rospy.ServiceProxy('update_pois', UpdatePOIs)
		newPOIs = []
		for i in self.list_of_points:	
			pose = Pose2D(i.pose.position.x, i.pose.position.y, 2*math.asin(i.pose.orientation.z))    
			newPOIs.append(LabeledPose(i.name,pose))
		try:
			resp = update_pois(newPOIs)
		except rospy.ServiceException as exc:
			print("Service did not process request: " + str(exc))	
	
	def editPOI(self, feedback):
		rospy.loginfo("EDIT %s menu:%s"%(feedback.marker_name,feedback.menu_entry_id))		
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
				self.newPOIfromPose(i, i.name, is_editable=False)
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
				self.newPOIfromPose(i, i.name, is_editable=True)
				self.applyChanges()		

	## @brief Starts the route
	def gotoPOI(self, feedback):

		goal = PoseStamped()
		goal.header.stamp = rospy.Time.now()
		goal.header.frame_id =  frame_id
		goal.pose.position.x = feedback.pose.position.x
		goal.pose.position.y = feedback.pose.position.y
	
		goal.pose.orientation.x= feedback.pose.orientation.x
		goal.pose.orientation.y = feedback.pose.orientation.y
		goal.pose.orientation.z = feedback.pose.orientation.z
		goal.pose.orientation.w = feedback.pose.orientation.w
		
		self.planner_client.goTo(goal)
		rospy.loginfo('startRouteCB: Sending pose')
		return
	
	## @brief Starts the route
	def setInitialPose(self, feedback):		
		msg = PoseWithCovarianceStamped()
		msg.header.stamp = rospy.Time.now()
		msg.header.frame_id = frame_id
		msg.pose.covariance = self._default_pose_covariance
		msg.pose.pose.position.x = feedback.pose.position.x
		msg.pose.pose.position.y = feedback.pose.position.y
		msg.pose.pose.position.z = 0.0
		msg.pose.pose.orientation.x = feedback.pose.orientation.x
		msg.pose.pose.orientation.y = feedback.pose.orientation.y
		msg.pose.pose.orientation.z = feedback.pose.orientation.z
		msg.pose.pose.orientation.w = feedback.pose.orientation.w
		
		self.init_pose_client.setPose(msg)
		rospy.loginfo('setPoseCB: setting pose')
		return
	
	## @brief Stops the current route if it's started
	def stop(self, feedback):
		self.planner_client.cancel()
		return
				 
if __name__=="__main__":
	rospy.init_node("poi_markers")
	
	_name = rospy.get_name().replace('/','')
	
	arg_defaults = {
	  'frame_id': 'map',
	  'goto_planner': 'move_base',
	  'init_pose_topic_name': 'initialpose'	  
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
	
	frame_id = args['frame_id']
	
	server = PointPathManager(_name, goto_planner = args['goto_planner'], init_pose_topic_name = args['init_pose_topic_name'])
	
	t_sleep = 0.5
	running = True
	
	while not rospy.is_shutdown() and running:
		
		try:
			rospy.sleep(t_sleep)
		except rospy.exceptions.ROSInterruptException:
			rospy.loginfo('Main: ROS interrupt exception')
			running = False
			
		

