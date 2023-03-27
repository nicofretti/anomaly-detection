#!/usr/bin/env python
import rospy
import abc


class ManagerAbstractClass(object):
    '''
    Abstract class to create a manager. This class has a state machine to control the execution of a action. The states can be:
    - (-1) Error - If the action has a error when it is executing this state be use to indicate is necessary a human intervention 
    - (0) Idle - The manager is wating to the new action
    - (1) PreIdle - State before Idle state. This state is to check or connect with the action node
    - (2) Execute - The manager execute the action
    - (3) Executing - The manager is executing the action
    - (4) Finalized - The manager finalize the action
    - (5) Fail - If the action has a error but it can be relaunched automatically
    - (6) Stop - If the action is stop the action

    Attributes
    ----------
    state_description_ : dictionary
        This attribute has the state description in a string
    STATES_ERROR : int
        Value of state Error
    STATES_PREIDLE : int
        Value of state PreIdle
    STATES_IDLE : int
        Value of state Idle
    STATES_EXECUTE : int
        Value of state Execute
    STATES_EXECUTING : int
        Value of state Executing
    STATES_FINALIZED : int
        Value of state Finalized
    STATES_FAIL : int
        Value of state Fail
    current_state_ : int
        The current machine status
    message_ : str
        Text with addictional information
    current_action_msg_ : str
        Text with current the action message

    Methods
    -------
    getState()
        Return current_state_
    getCurrentActionMessage()
        Return current_action_msg_
    getMessage()
        Return message_
    getDescriptionCurrentState()
        Return a string with the current state
    changeState(new_state, message="")
        Change the state machine and save information aditional using message
    startAction(action_msg)
        Start the action if the state machine is in Idle. Return True if it can start else return False
    recivedAction() : @abstractmethod
        Method called when a new action add.If it doesn't implemented it doesn't nothing.
    stateMachineLogic(event)
        Method uses to execute the current state defined by the methods: statePreidle, stateIdle, stateExecuting, stateFail, stateError, stateFinalized
    statePreidle() : @abstractmethod
        Method executes when the state machine is in PreIdle. This start only one time before to change to Idle. If it doesn't implemented, the state machine change to Idle.
    stateIdle() : @abstractmethod
        Method executes when the state machine is in Idle. The machine state waits for a action. If it doesn't implemented, doesn't do nothing while not recive a id_action. 
    stateExecute() : @abstractmethod
        Method executes when the state machine is in Execute. This method is need to be implemented
    stateExecuting() : @abstractmethod
        Method executes when the state machine is in Executing.
    stateFail() : @abstractmethod
        Method executes when the state machine is in Fail. If it doesn't implemented, the state machine change to Idle.
    stateError() : @abstractmethod
        Method executes when the state machine is in Error. If it doesn't implemented, doesn't do nothing and it will be in this state.
    stateFinalized() : @abstractmethod
        Method executes when the state machine is in Finalized. If it doesn't implemented, the state machine change to Idle.
    confirmationFinished()
	Method to change from Finished to Idle. It will call from the RMS when the manager is in state Finished

    '''
    def __init__(self):
        self.states_description_ = {
            -1: "Error",
            0: "PreIdle",
            1: "Idle",
            2: "Execute",
            3: "Executing",
            4: "Finalized",
            5: "Fail",
            6: "Stop"
        }
        self.STATES_ERROR = -1
        self.STATES_PREIDLE = 0
        self.STATES_IDLE = 1
        self.STATES_EXECUTE = 2
        self.STATES_EXECUTING = 3
        self.STATES_FINALIZED = 4
        self.STATES_FAIL = 5
        self.STATES_STOP = 6
        self.current_state_ = self.STATES_PREIDLE
        self.message_ = ""
        self.current_action_msg_ = None
        rospy.Timer(rospy.Duration(0.5), self.stateMachineLogic)

    '''
    Return current_state_

    Returns
    -------
    int
	Id of current state machine
    '''
    def getState(self):
        return self.current_state_
    '''
    Return current_action_msg_

    Returns
    -------
    str
	Id of current action
    '''
    def getStringState(self):
	return self.states_description_[self.current_state_]
    '''
    Return message_

    Returns
    -------
    str
	Text in attribute message_
    '''
    def getStateMessage(self):
        return self.message_
    '''
    Return a string with the current state

    Returns
    -------
    str
	A String with the state machine
    '''
    def getDescriptionCurrentState(self):
        return self.states_description_.get(self.current_state_, "Unknown " + str(self.current_state_))

    '''
    Change the state machine and save information aditional using message

    Parameters
    ----------
    new_state : int
	Id of new state in state machine
    message : str, optional
	Additional information in form of text

    Returns
    -------
    str
	A String with the state machine
    '''
    def changeState(self, new_state, message=""):
	if (new_state == self.STATES_FINALIZED or new_state == self.STATES_FAIL or new_state == self.STATES_ERROR):
		self.current_action_msg_ = None
        self.current_state_ = new_state
        self.message_ = message
        return self.getDescriptionCurrentState()

    '''
    Start the action if the state machine is in Idle. Return True if it can start else return False

    Parameters
    ----------
    action_msg : str
        Id of current action
    
    Returns
    -------
    boolean
       Return True if it can start else return False
    '''
    def startAction(self, action_msg):
	print ("Recived in the class "+ self.__class__.__name__ + " the action : " + action_msg + ". The machine is in " + self.states_description_[self.current_state_])
        if (self.current_state_ == self.STATES_FINALIZED):
            rate = rospy.Rate(10)
            while(self.current_state_ != self.STATES_IDLE):
                rate.sleep()
        if(self.current_state_ == self.STATES_IDLE):
            self.current_action_msg_ = action_msg
            self.recivedAction()
            return True
        else:
            print("The manager is not in IDLE state, current state: " + self.current_state_)
            return False

    '''
    Method uses to execute the current state defined by the methods: statePreidle, stateIdle, stateExecuting, stateFail, stateError, stateFinalized

    Parameters
    ----------
    event : rospy.TimerEvent
        Called with a rospy timer event in this class.
    '''
    def stateMachineLogic(self, event):
	if(self.current_state_ == self.STATES_PREIDLE):
            self.statePreidle()
        elif (self.current_state_ == self.STATES_IDLE):
            self.stateIdle()
            if(self.current_action_msg_ != None):
                self.changeState(self.STATES_EXECUTE, "Execute")
        elif (self.current_state_ == self.STATES_EXECUTE):
            self.stateExecute()
        elif (self.current_state_ == self.STATES_EXECUTING):
            self.stateExecuting()
        elif (self.current_state_ == self.STATES_FINALIZED):
            self.current_action_msg_ = None
	    self.stateFinalized()
        elif (self.current_state_ == self.STATES_FAIL):
            self.current_action_msg_ = None
	    self.stateFail()
        elif (self.current_state_ == self.STATES_ERROR):
            self.current_action_msg_ = None
	    self.stateError()
            
	
	#print (self.__class__.__name__ + " state: " + self.states_description_[self.current_state_])
    
    '''
    Method called when a new action add.If it doesn't implemented it doesn't nothing.
    '''
    @abc.abstractmethod
    def recivedAction(self):
        None

    '''
    Method executes when the state machine is in PreIdle. This start only one time before to change to Idle. If it doesn't implemented, the state machine change to Idle.
    '''
    @abc.abstractmethod
    def statePreidle(self):
        self.changeState(self.STATES_IDLE, "Preidle")

    '''
    Method executes when the state machine is in Idle. The machine state waits for a action. If it doesn't implemented, doesn't do nothing while not recive a id_action. 
    '''
    @abc.abstractmethod
    def stateIdle(self):
        None
    
    '''
    Method executes when the state machine is in Execute. This method is need to be implemented
    '''
    @abc.abstractmethod
    def stateExecute(self):
        self.changeState(self.STATES_ERROR,
                         "Not defined stateExecute metothod")
        print "Not defined stateExecute metothod"

    '''
    Method executes when the state machine is in Executing.
    '''
    @abc.abstractmethod
    def stateExecuting(self):
        self.changeState(self.STATES_IDLE, "Executing")

    '''
    Method executes when the state machine is in Fail. If it doesn't implemented, the state machine change to Idle.
    '''
    @abc.abstractmethod
    def stateFail(self):
	rospy.sleep(3.)
	self.changeState(self.STATES_IDLE, "From fail")

    '''
    Method executes when the state machine is in Error. If it doesn't implemented, doesn't do nothing and it will be in this state.
    '''
    @abc.abstractmethod
    def stateError(self):
        rospy.sleep(3.)
	self.changeState(self.STATES_IDLE, "From Error")
    
    '''
    Method executes when the state machine is in Finalized. If it doesn't implemented, it doesn't nothing. Not use changeState in that state.
    '''
    @abc.abstractmethod
    def stateFinalized(self):
        None

    '''
    Will be call when is neccessary stop the manager.
    '''
    @abc.abstractmethod
    def cancel(self):
        return True

    def callCancel(self):
	rate = rospy.Rate(10)	
	self.current_action_msg_ = None
	if(self.current_state_ == self.STATES_PREIDLE or self.current_state_ == self.STATES_IDLE):
		return True
	elif (self.current_state_ == self.STATES_EXECUTE or self.current_state_ == self.STATES_EXECUTING):
		
		while (not self.cancel()):
			rate.sleep()
			self.changeState(self.STATES_IDLE, "From Stop")
		return True
	elif (self.current_state_ == STATES_FINALIZED or self.current_state_ == self.STATES_FAIL or self.current_state_ == self.STATES_ERROR):
		return True

    '''
    Change the state from Finalized to IDLE.

    Returns:
    boolean
       It is true if in Finalized state and change to Idle, else return false
    '''
    def confirmationFinished(self):
	if (self.current_state_ == self.STATES_FINALIZED):
		print (self.changeState(self.STATES_IDLE, "Finalized"))		
		return True
	else:
		return False
