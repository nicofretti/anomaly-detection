#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robot_local_control_msgs.srv import PlacePetition
from robot_local_control_msgs.srv import PlacePetitionRequest
from procedures_msgs.srv import ProcedureQuery
from procedures_msgs.srv import ProcedureQueryRequest

class ManagerRobotLocalControlPlace(ManagerAstract.ManagerAbstractClass):
    def __init__(self, service_place_petition_add = "/robot/robot_local_control/NavigationComponent/PlaceComponent/add", service_place_petition_query_state = "/robot/robot_local_control/NavigationComponent/PlaceComponent/query_state", place_frame_id = ""):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        global PlacePetition
        global PlacePetitionRequest
        global ProcedureQuery
        global ProcedureQueryRequest
        self.service_place_petition_add = service_place_petition_add
        self.service_place_petition_query_state = service_place_petition_query_state
        self.place_frame_id = place_frame_id
        self.finish_move = False
        self.id_place = -1

    def callPlacePetitionAddService(self):
	self.finish_move = False
	self.id_place = -1
	rospy.wait_for_service(self.service_place_petition_add)
	try:
		service_robot_local_control_place_add_petition = rospy.ServiceProxy(self.service_place_petition_add, PlacePetition)
		robot_local_control_place_petition = PlacePetitionRequest()
		response = service_robot_local_control_place_add_petition(robot_local_control_place_petition)
		if(response.result.result == "ok"):
			self.id_place = response.state.header.id
			print "Added place id=" + str(self.id_place)
			return True
		else:
			self.id_place = -1
			print "Not added place"
			return False
	except rospy.ServiceException, e:
		self.id_place = -1
		print "Service call failed"
		return False

    def callProcedureQueryIsThePetitionFinish(self):
		rospy.wait_for_service(self.service_place_petition_query_state)
		try:
			service_place_petition_query_state_petition = rospy.ServiceProxy(self.service_place_petition_query_state, ProcedureQuery)
			robot_local_control_query_petition = ProcedureQueryRequest()
			robot_local_control_query_petition.header.id = self.id_place
			response = service_place_petition_query_state_petition(robot_local_control_query_petition)
			if(response.state.current_state == response.state.FINISHED):
				print "Finish place id=" + str(self.id_place)
				self.finish_move = True
				return True
			else:
				print "Wait for finish place id=" + str(self.id_place)
				return False
		except rospy.ServiceException, e:
			print "Service call failed"
			return False

    def statePreidle(self):
	self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
	print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callPlacePetitionAddService()):
		print ("ManagerPlace-->Executing ")
		rospy.sleep(1.)
		self.changeState(self.STATES_EXECUTING, "Executing")
	else:
        	self.changeState(self.STATES_FINALIZED, "Finalize, the action not exist")


    #NECESSARY A DEFINITION
    def stateExecuting(self):
	rospy.sleep(1.)
	if self.callProcedureQueryIsThePetitionFinish():
		self.changeState(self.STATES_FINALIZED, "Finalized")

    def stateFail(self):
        self.changeState(self.STATES_FINALIZED, "Fail")

    def stateError(self):
        print("Error")
	self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
        print("Manager place--->Finsihed.")

