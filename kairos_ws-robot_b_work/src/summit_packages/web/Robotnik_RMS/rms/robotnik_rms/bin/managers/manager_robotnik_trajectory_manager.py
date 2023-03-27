#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from robotnik_trajectory_planner.msg import State
from robotnik_trajectory_manager.srv import LoadState
from robotnik_trajectory_manager.srv import LoadStateRequest

class ManagerRobotnikTrajectoryManager(ManagerAstract.ManagerAbstractClass):
    def __init__(self):
        ManagerAstract.ManagerAbstractClass.__init__(self)
        
        global State
        global LoadState
        global LoadStateRequest

        self.state_topic = "/rt_traj_planner/state"
        self.send_state_service = "/rt_traj_manager/load_state"
        self.rt_traj_planner_state_feedback_message = None
	self.finish_move = False
	self.move_try = 5
	self.cont_try = 0
	self.max_temp = 30
	self.cont_time = 0
        
        rospy.Subscriber(self.state_topic, State, self.state_callback)
        
    
    
    
    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def recivedAction(self):
        print ("ManagerExample-> Recive a action: " + str(self.current_action_msg_))

    def stateIdle(self):
        #print("Idle")
        None

    def stateExecute(self):
	if (self.callPoiManagerService(self.current_action_msg_)):
            print ("ManagerExample-->Executing: " + str(self.current_action_msg_))
            self.changeState(self.STATES_EXECUTING, "Executing")
        else:
            self.changeState(self.STATES_FINALIZED, "Finalize, the action not exist")


    #NECESSARY A DEFINITION
    def stateExecuting(self):
	rate = rospy.Rate(4) # 10hz
	if(self.rt_traj_planner_state_feedback_message == None or self.finish_move == False):
		print("Waiting to move_arm finish")
		return
        if(self.rt_traj_planner_state_feedback_message.goal_state=="IDLE"):
		self.changeState(self.STATES_FINALIZED, "Finalize")
        if(self.cont_time >= self.max_temp*4):
		if (self.cont_try > self.move_try):
			print ("Faillllll!!!")			
			self.changeState(self.STATES_FINALIZED, "Fail")
		else:
			print ("Other try : " + str(self.cont_try) + "/" + str(self.move_try))
			self.changeState(self.STATES_IDLE, "Other try")
	self.cont_time = self.cont_time +1
	print (str(self.cont_time))
	rate.sleep()
        
    def stateFail(self):
        self.changeState(self.STATES_FINALIZED, "Fail")

    def stateError(self):
        print("Error")
	self.changeState(self.STATES_FINALIZED, "Fail")

    def stateFinalized(self):
	self.cont_try = 0
        print ("ManagerExample--->Finsihed: " + str(self.current_action_msg_))
        self.changeState(self.STATES_IDLE, "Finalized")


    def state_callback(self, data):
	if(self.rt_traj_planner_state_feedback_message == None):
		self.rt_traj_planner_state_feedback_message = data
	if (self.rt_traj_planner_state_feedback_message.goal_state=="ACTIVE" and data.goal_state =="IDLE"):
		self.finish_move = True
        self.rt_traj_planner_state_feedback_message = data


    def callPoiManagerService(self, id):
	self.finish_move = False
	self.cont_time = 0
	self.cont_try = self.cont_try +1
	
        rospy.wait_for_service(self.send_state_service)
        try:
            service_robotnik_trajectory_manajer = rospy.ServiceProxy(self.send_state_service, LoadState)
            request_robotnik_trajectory_manajer = LoadStateRequest()
            request_robotnik_trajectory_manajer.id = id
            response = service_robotnik_trajectory_manajer(request_robotnik_trajectory_manajer)
            if(response.ret == False):
		print "False from trajectory"
                return False
            else:
                print("Sended move arm")
                return True

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            return False
