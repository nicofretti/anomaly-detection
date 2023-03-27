#!/usr/bin/env python
import rospy
import abc
import manager_astract_base_class as ManagerAstract


from map_nav_manager.msg import PoiState
from map_nav_manager.srv import GoToTag
from map_nav_manager.srv import GoToTagRequest


class ManagerPoiGoTo(ManagerAstract.ManagerAbstractClass):
    def __init__(self, goto_tag_poi_service="/poi_interactive_marker/goto_tag", state_poi_topic="/poi_interactive_marker/state"):
        ManagerAstract.ManagerAbstractClass.__init__(self)

        global PoiState
        global GoToTag
        global GoToTagRequest

        self.state_poi_topic = state_poi_topic
        self.send_poi_service = goto_tag_poi_service
        self.state_feedback_message = None
        self.first_change_feedback = False
	self.check_idle_state = 0
	self.check_times_idle_state = 5

        rospy.Subscriber(self.state_poi_topic, PoiState,
                         self.state_poi_callback)

    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    def stateIdle(self):
        # print("Idle")
        None

    # NECESSARY A DEFINITION
    def stateExecute(self):
	self.check_idle_state = 0
        print("ManagerPoiGoTo-->Executing: " + str(self.current_action_msg_))
        if (self.callPoiManagerService(self.current_action_msg_)):
            print("ManagerPoiGoTo-->Executing: " +
                  str(self.current_action_msg_))
            self.first_change_feedback = False
            self.changeState(self.STATES_EXECUTING,
                             "Executing task " + str(self.current_action_msg_))
        else:
            self.changeState(self.STATES_FAIL, "Fail to send a point")
        #rate = rospy.Rate(2)

    def stateExecuting(self):
        if(self.first_change_feedback):
            if(self.state_feedback_message.action != PoiState.IDLE):
		self.check_idle_state = 0
                print("waiting for finish action")
            else:
		if(self.check_idle_state > self.check_times_idle_state):
                	self.changeState(self.STATES_FINALIZED, "Finalized the task")
		else:
			self.check_idle_state = self.check_idle_state +1

    def stateFail(self):
        self.changeState(self.STATES_IDLE, "Fail")
        # None

    def stateError(self):
        print("Error")
        # None

    def stateFinalized(self):
        print("ManagerPoiGoTo--->Finsihed: " + str(self.current_action_msg_))

    def state_poi_callback(self, data):
        self.state_feedback_message = data
        self.first_change_feedback = True

    def callPoiManagerService(self, id):
        rospy.wait_for_service(self.send_poi_service)
        try:
            service_go_to_tag = rospy.ServiceProxy(
                self.send_poi_service, GoToTag)
            request_pose_list = GoToTagRequest()
            request_pose_list.name = id
            response = service_go_to_tag(request_pose_list)
            print(request_pose_list)
            if(response.ret == False):
                return False
            else:
                return True
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
            return False
