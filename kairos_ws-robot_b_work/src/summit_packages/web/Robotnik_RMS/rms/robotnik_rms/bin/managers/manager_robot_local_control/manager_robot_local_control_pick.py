#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robot_local_control_msgs.srv import PickPetition
from robot_local_control_msgs.srv import PickPetitionRequest
from procedures_msgs.srv import ProcedureQuery
from procedures_msgs.srv import ProcedureQueryRequest
from poi_manager.srv import ReadPOIs
from poi_manager.srv import ReadPOIsRequest

class ManagerRobotLocalControlPick(ManagerAstract.ManagerAbstractClass):
    def __init__(self, service_Pick_petition_add = "/robot/robot_local_control/NavigationComponent/PickComponent/add", service_Pick_petition_query_state = "/robot/robot_local_control/NavigationComponent/PickComponent/query_state", service_Pick_petition_cancel = "/robot/robot_local_control/NavigationComponent/PickComponent/cancel"):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global PickPetition
        global PickPetitionRequest
        global ProcedureQuery
        global ProcedureQueryRequest
        global ReadPOIs
        global ReadPOIsRequest
        self.service_Pick_petition_add = service_Pick_petition_add
        self.service_Pick_petition_query_state = service_Pick_petition_query_state
	self.service_Pick_petition_cancel = service_Pick_petition_cancel
        self.finish_action = False
        self.id_Pick = -1
	self.finalized_with_error = ""

    def callPickPetitionAddService(self):
	self.finish_action = False
	self.id_Pick = -1
        #add in robot local control Pick
	rospy.wait_for_service(self.service_Pick_petition_add)
	try:
		service_robot_local_control_Pick_add_petition = rospy.ServiceProxy(self.service_Pick_petition_add, PickPetition)
		robot_local_control_Pick_petition = PickPetitionRequest()
		response = service_robot_local_control_Pick_add_petition(robot_local_control_Pick_petition)
		if(response.result.result == "ok"):
			self.id_Pick = response.state.header.id
			print "Added Pick id=" + str(self.id_Pick)
			return True
		else:
			self.id_Pick = -1
			print "Not added Pick"
			return False
	except rospy.ServiceException, e:
		self.id_Pick = -1
		print "Service add Pick call failed " + str(e)
		return False

    def callProcedureQueryIsThePetitionFinish(self):
		rospy.wait_for_service(self.service_Pick_petition_query_state)
		try:
			service_Pick_petition_query_state_petition = rospy.ServiceProxy(self.service_Pick_petition_query_state, ProcedureQuery)
			robot_local_control_query_petition = ProcedureQueryRequest()
			robot_local_control_query_petition.header.id = self.id_Pick
			response = service_Pick_petition_query_state_petition(robot_local_control_query_petition)
			if(response.state.last_event == response.state.FINISH):
				print "Finish Pick id=" + str(self.id_Pick)
				self.finish_action = True
				return True
			elif(response.state.last_event == response.state.ABORT):
				print "Fail: Abort Pick id=" + str(self.id_Pick)
				self.finish_action = True
				return False

			else:
				print "Wait for finish Pick id=" + str(self.id_Pick)
				return False
		except rospy.ServiceException, e:
			self.finalized_with_error = "Exception - " + str(e)
			self.finish_action = True
			print "Service get Pick state call failed"
			return False

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	self.finalized_with_error = ""
	print ("ManagerRobotLocalControlPick-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callPickPetitionAddService()):
		print ("ManagerRobotLocalControlPick-->Executing ")
		rospy.sleep(1.)
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FAIL, "Fail when try send the point, the manager Pick is in fail")


    def stateExecuting(self):
	rospy.sleep(1.)
	query_finish = self.callProcedureQueryIsThePetitionFinish()
	if (query_finish and self.finish_action):
		self.changeState(self.STATES_FINALIZED, "Finalized")
	elif (not query_finish and self.finish_action):
		self.changeState(self.STATES_FAIL, "Fail in status : "+self.finalized_with_error+"  .the manager Pick is in fail")
	
    def stateError(self):
        print("Error")
	#self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
        print("Manager Pick--->Finsihed.")

    def cancel(self):
	rospy.wait_for_service(self.service_Pick_petition_cancel)
	if(self.id_Pick == -1):
		return True
	try:
		service_Pick_petition_query_state_petition = rospy.ServiceProxy(self.service_Pick_petition_cancel, ProcedureQuery)
		robot_local_control_query_petition = ProcedureQueryRequest()
		robot_local_control_query_petition.header.id = self.id_Pick
		response = service_Pick_petition_query_state_petition(robot_local_control_query_petition)
		if(response.state.last_event == response.state.CANCEL):
			print "Cancel Pick id=" + str(self.id_Pick)
			self.finish_action = True
			return True
		else:
			print "Wait for cancel Pick id=" + str(self.id_Pick)
			return False
	except rospy.ServiceException, e:
		self.finalized_with_error = "Exception - " + str(e)
		self.finish_action = True
		print "Service get Pick cancel call failed"
		return False
	

