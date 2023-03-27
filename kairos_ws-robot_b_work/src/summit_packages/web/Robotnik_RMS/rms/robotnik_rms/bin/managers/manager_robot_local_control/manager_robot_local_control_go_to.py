#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robot_local_control_msgs.srv import GoToPetition
from robot_local_control_msgs.srv import GoToPetitionRequest
from procedures_msgs.srv import ProcedureQuery
from procedures_msgs.srv import ProcedureQueryRequest
from poi_manager.srv import ReadPOIs
from poi_manager.srv import ReadPOIsRequest
from robot_local_control_msgs.msg import Pose2DStamped
from robot_local_control_msgs.msg import Twist2D

class ManagerRobotLocalControlGoTo(ManagerAstract.ManagerAbstractClass):
    def __init__(self, service_go_to_petition_add = "/robot/robot_local_control/NavigationComponent/GoToComponent/add", service_go_to_petition_query_state = "/robot/robot_local_control/NavigationComponent/GoToComponent/query_state", service_go_to_petition_cancel = "/robot/robot_local_control/NavigationComponent/GoToComponent/cancel" ,poi_manager_read_pois = "/robot/poi_manager/read_pois" , go_to_frame_id = "robot_map", max_velocity_linear_x = 0.7, max_velocity_linear_y = 0.7, max_velocity_angular_z = 0.5):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global GoToPetition
        global GoToPetitionRequest
        global ProcedureQuery
        global ProcedureQueryRequest
        global ReadPOIs
        global ReadPOIsRequest
	global Pose2DStamped
	global Twist2D
        self.service_go_to_petition_add = service_go_to_petition_add
        self.service_go_to_petition_query_state = service_go_to_petition_query_state
	self.service_go_to_petition_cancel = service_go_to_petition_cancel
        self.poi_manager_read_pois = poi_manager_read_pois
        self.go_to_frame_id = go_to_frame_id
        self.finish_move = False
        self.id_go_to = -1
	self.max_velocities = Twist2D()
	self.max_velocities.linear_x = max_velocity_linear_x
	self.max_velocities.linear_y = max_velocity_linear_y
	self.max_velocities.angular_z = max_velocity_angular_z
	self.goal = Pose2DStamped()
	self.goal.header.frame_id = go_to_frame_id
	self.finalized_with_error = ""
		
    def getPoint(self):
	pose = None
        #find the target recived
        rospy.wait_for_service(self.poi_manager_read_pois)
	try:
		service_poi_manager_read_pois = rospy.ServiceProxy(self.poi_manager_read_pois, ReadPOIs)
		read_pois_request = ReadPOIsRequest()
		response_read_pois = service_poi_manager_read_pois(read_pois_request)
		for poi_foreach in response_read_pois.pose_list:
			if (poi_foreach.label == self.current_action_msg_):
				pose = poi_foreach.pose
        except rospy.ServiceException, e:
		print "Service ReadPOIs in " + poi_manager_read_pois + " didn't call." 
		return False
	#Exist the point?
	if pose == None:
		print "The point " + self.current_action_msg_ + " is not in " + poi_manager_read_pois
		return False
	self.goal.pose = pose
	return True

    def callGoToPetitionAddService(self):
	self.finish_move = False
	self.id_go_to = -1
        #add in robot local control go to
	rospy.wait_for_service(self.service_go_to_petition_add)
	try:
		service_robot_local_control_go_to_add_petition = rospy.ServiceProxy(self.service_go_to_petition_add, GoToPetition)
		robot_local_control_go_to_petition = GoToPetitionRequest()
		robot_local_control_go_to_petition.procedure.goals.append(self.goal)
		robot_local_control_go_to_petition.procedure.max_velocities.append(self.max_velocities)
		response = service_robot_local_control_go_to_add_petition(robot_local_control_go_to_petition)
		if(response.result.result == "ok"):
			self.id_go_to = response.state.header.id
			print "Added go to id=" + str(self.id_go_to)
			return True
		else:
			self.id_go_to = -1
			print "Not added go to"
			return False
	except rospy.ServiceException, e:
		self.id_go_to = -1
		print "Service add go to call failed " + str(e)
		return False

    def callProcedureQueryIsThePetitionFinish(self):
		rospy.wait_for_service(self.service_go_to_petition_query_state)
		try:
			service_go_to_petition_query_state_petition = rospy.ServiceProxy(self.service_go_to_petition_query_state, ProcedureQuery)
			robot_local_control_query_petition = ProcedureQueryRequest()
			robot_local_control_query_petition.header.id = self.id_go_to
			response = service_go_to_petition_query_state_petition(robot_local_control_query_petition)
			if(response.state.last_event == response.state.FINISH):
				print "Finish go to id=" + str(self.id_go_to)
				self.finish_move = True
				return True
			elif(response.state.last_event == response.state.ABORT):
				print "Fail: Abort go to id=" + str(self.id_go_to)
				self.finish_move = True
				return False
			elif(response.state.last_event == response.state.CANCEL):
				print "Fail: Cancel go to id=" + str(self.id_go_to)
				self.finish_move = True
				return False
			else:
				print "Wait for finish go to id=" + str(self.id_go_to) + " state: " 
				return False
		except rospy.ServiceException, e:
			self.finalized_with_error = "Exception - " + str(e)
			self.finish_move = True
			print "Service get go to state call failed"
			return False

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	self.finalized_with_error = ""
	print ("ManagerRobotLocalControlGoTo-> Recive a action: " + str(self.current_action_msg_))
	if (self.getPoint() == False):
		self.changeState(self.STATES_FAIL, "Fail when try get the point, the manager go to is in fail")

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callGoToPetitionAddService()):
		print ("ManagerRobotLocalControlGoTo-->Executing ")
		rospy.sleep(1.)
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FAIL, "Fail when try send the point, the manager go to is in fail")


    def stateExecuting(self):
	rospy.sleep(1.)
	query_finish = self.callProcedureQueryIsThePetitionFinish()
	if (query_finish and self.finish_move):
		self.changeState(self.STATES_FINALIZED, "Finalized")
	elif (not query_finish and self.finish_move):
		self.changeState(self.STATES_FAIL, "Fail in status : "+self.finalized_with_error+"  .the manager go to is in fail")
	
    def stateError(self):
        print("Error")
	#self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
        print("Manager go_to--->Finsihed.")

    def cancel(self):
	rospy.wait_for_service(self.service_go_to_petition_cancel)
	if(self.id_go_to == -1):
		return True
	try:
		service_go_to_petition_query_state_petition = rospy.ServiceProxy(self.service_go_to_petition_cancel, ProcedureQuery)
		robot_local_control_query_petition = ProcedureQueryRequest()
		robot_local_control_query_petition.header.id = self.id_go_to
		response = service_go_to_petition_query_state_petition(robot_local_control_query_petition)
		if(response.state.last_event == response.state.CANCEL):
			print "Cancel go to id=" + str(self.id_go_to)
			self.finish_move = True
			return True
		else:
			print "Wait for cancel go to id=" + str(self.id_go_to)
			return False
	except rospy.ServiceException, e:
		self.finalized_with_error = "Exception - " + str(e)
		self.finish_move = True
		print "Service get go to cancel call failed"
		return False
	

