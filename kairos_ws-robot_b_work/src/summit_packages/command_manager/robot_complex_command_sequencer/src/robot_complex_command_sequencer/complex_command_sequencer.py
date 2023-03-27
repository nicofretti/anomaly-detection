#!/usr/bin/env python
import re

import rospy
import actionlib

from std_msgs.msg import String, Empty , Bool
from std_srvs.srv import Trigger, TriggerResponse

from robot_simple_command_manager_msgs.msg import *
from robot_simple_command_manager_msgs.srv import SetCommandString, SetCommandStringResponse
from robot_simple_command_manager_msgs.srv import ManageCommandManager, ManageCommandManagerResponse
from robot_simple_command_manager_msgs.srv import GetCommandManagerList, GetCommandManagerListResponse
from robot_local_control_msgs.msg import Status

from actionlib_msgs.msg import GoalStatus
from rcomponent.rcomponent import *

import yaml

import Queue
import datetime
"""
    QUEUE of SequenceList
        SequenceList:
                SEQ1 SEQ2 ... SQn [LOOP]
                    SEQx -> [CMD1 CMD2 .. CMDn]
"""



class SequenceList():
    """
        Manages the list of sequences
    """

    def __init__(self, sequences):
        """
            Args:
                - sequences: string[] with all the sequences in the list
        """
        # Array of sequences
        self.sequences_string = sequences
        self.sequences = []
        self.current_sequence_index = 0
        self.id = ' '.join([str(elem) for elem in self.sequences_string])

        #
        self.loop = False
        self.loop_cycles = 0
        #
        self._creation_datetime = datetime.datetime.now()
        self._creation_rostime = rospy.Time.now()
        self._id = self._creation_datetime.strftime("SL-%Y%m%d-%H%M%S")

    def get_id(self):
        """
            Returns object id as string
        """

        return self.id

    def set_loop(self, value):
        """
            Sets the command loop option
        """
        self.loop = value

    def is_loop(self):
        """
            Returns if loop is enabled for this command
        """
        return self.loop

    def get_loop_cycles(self):
        """
            Returns the number of completed loops
        """
        return self.loop_cycles

    def append(self, sequence_id, commands):
        """
            Adds a new sequence

            Args:
                sequence_id: string as the sequence id
                commands: string [] as the commands that contains the sequence
        """
        self.sequences.append(Sequence(sequence_id, commands))


    def reset(self):
        """
            Resets the list of sequences
        """
        for sequence in self.sequences:
            sequence.reset()
        self.current_sequence_index = 0
        if self.loop:
            self.loop_cycles += 1

    def _get_current_sequence(self):
        """
            Gets the current sequence
        """
        if self.current_sequence_index >= len(self.sequences):
            return None
        else:
            return self.sequences[self.current_sequence_index]

    def get_next_command(self):
        """
            Gets the next command of the current sequence
            Returns:
                - The next command as string
                - None if there are no more command in the sequence list
        """
        while True:
            current_sequence = self._get_current_sequence()
            # No more sequences nor commands
            if current_sequence == None:
                return None

            command = current_sequence.get_next_command()

            if command == None:  # No more commands in the sequence
                self.current_sequence_index += 1  # increase the sequence
            else:
                return command

    def get_current_sequence_id(self):
        """
            Returns the current sequence id as string
            Returns None if the sequences finished
        """
        current_sequence = self._get_current_sequence()
        # No more sequences nor commands
        if current_sequence == None:
            return None
        else:
            return current_sequence.get_id()

    def get_count_remaining_commands(self):
        """
            Gets the size of the sequence
        """
        return (self._get_current_sequence().get_size() - self._get_current_sequence().get_index())

class Sequence():
    """
        A sequence is a list of simple commands
    """

    def __init__(self, id, commands):
        """
            Args:
                id: String
                commands: String[]
        """
        self.id = id
        self.commands = commands
        self.current_command_index = 0
        #
        self._creation_datetime = datetime.datetime.now()
        self._creation_rostime = rospy.Time.now()
        self._id = self._creation_datetime.strftime("S-%Y%m%d-%H%M%S")

    def reset(self):
        """
        Resets the sequence of commands, starting from the begining
        """
        self.current_command_index = 0

    def get_next_command(self):
        """
            Gets the next command in the list
        """
        if self.current_command_index >= len(self.commands):
            return None
        else:
            command = self.commands[self.current_command_index]
            self.current_command_index += 1
            return command

    def get_id(self):
        """
            Gets the id of the sequence
        """
        return self.id

    def get_size(self):
        """
            Gets the id of the sequence
        """
        return len(self.commands)

    def get_index(self):
        """
            Gets the id of the sequence
        """
        return self.current_command_index


class ComplexCommandSequencer(RComponent):
    """
    A class used to load simple command handlers

    ...


    """

    def __init__(self):
        self.feedback_msg = CommandStringFeedback()
        self.feedback_action_msg = RobotSimpleCommandFeedback()
        self.action_client_feedback_msg = None
        # Available sequences of commands
        self.sequences_dict = {}
        # sequence list of commands in execution
        self.current_sequence_list = None
        self.current_sequence_list_id = ''
        # last sequence list executed
        self.last_sequence_list_id = ''
        self.current_sequence_list_remaining = 0
        # current command sent/in execution
        self.current_command = ''
        # Saves the time of the feedback msg
        self.last_time_received_action_client_feedback = rospy.Time(0)
        # Timeout to control the lack of communication from the action server
        self.communication_timeout = rospy.Duration.from_sec(10)
        self.maxsize_queue = 100
        # Queue of sequence list to execute
        self.sequences_list_queue = Queue.Queue(maxsize=self.maxsize_queue)
        # Flag to enable/disable adding sequences into the queue
        self.allow_queuing = False
        # Contains the words cannot be used as sequence ids
        self._sequence_reserved_keywords = ['LOOP']
        # Contains the string to publish in a topic the current sequence status
        self.current_sequence_string = '-'
        # Flag to enable/disable execute error sequences
        self.allow_error_sequence = False
        # Flag to enable/disable execute charge battery sequences
        self.allow_charge_battery_sequence = False
        # Flag of charge battery received
        self.charge_battery_received = False

        # Flag of file sequences read
        self.sequences_already_read = False

        self.error_detected = False

        self.emergency_detected = False

        # log data
        self.log = CommandLog()

        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)
        self.sequences_file = rospy.get_param('~sequences_file', 'sequences_default.yaml')
        self.action_client_namespace = rospy.get_param('~action_client_namespace', 'command_sequencer/action')
        self.topic_robot_status = rospy.get_param('~topic_robot_status', 'robot_local_control/state')
        self.topic_safety_emergency_stop = rospy.get_param('~topic_safety_emergency_stop', 'safety_module/emergency_stop')
        rospy.loginfo('%s:: Usig topic: %s' % (self._node_name, self.topic_safety_emergency_stop))
        self.allow_queuing = rospy.get_param('~allow_queuing', False)
        self.allow_error_sequence = rospy.get_param('~allow_error_sequence', True)
        self.allow_charge_battery_sequence = rospy.get_param('~allow_charge_battery_sequence', False)
        self.use_robot_status_emergency_stop = rospy.get_param('~use_robot_status_emergency_stop', False)
        self.use_safety_emergency_stop = rospy.get_param('~use_safety_emergency_stop', True)
        rospy.loginfo('%s:: Using safety: %d' % (self._node_name, self.use_safety_emergency_stop))


        self.add_sequence_name = rospy.get_param('~add_sequence', '~add_sequence')
        rospy.loginfo('%s:: Using add sequence service: %s' % (self._node_name, self.add_sequence_name))
        self.remove_sequence_name = rospy.get_param('~remove_sequence', '~remove_sequence')
        rospy.loginfo('%s:: Using remove sequence service: %s' % (self._node_name, self.remove_sequence_name))
        self.get_sequences_name = rospy.get_param('~get_sequences', '~get_sequences')
        rospy.loginfo('%s:: Using get sequence service: %s' % (self._node_name, self.get_sequences_name))

        self.validate_sequence_keywords()

    def ros_setup(self):
        """Creates and inits ROS components"""

        RComponent.ros_setup(self)

        self.command_sub = rospy.Subscriber('~command', CommandString, self.topic_cmd_cb)
        self.command_cancel_sub = rospy.Subscriber('~cancel', Empty, self.topic_cmd_cancel_cb)

        self.command_status = rospy.Subscriber(self.topic_robot_status, Status, self.topic_robot_status_cb)
        self.safety_status = rospy.Subscriber(self.topic_safety_emergency_stop, Bool, self.topic_safety_emergency_stop_cb)

        self.command_feedback_pub = rospy.Publisher(
            '~feedback', CommandStringFeedback, queue_size=1)

        self.command_log_publisher = rospy.Publisher(
            'command_log', CommandLog, queue_size=1)


        self.command_service = rospy.Service('~command', SetCommandString, self.service_cmd_cb)
        self.command_cancel_service = rospy.Service('~cancel', Trigger, self.service_cmd_cancel_cb)
        self.command_cancel_service = rospy.Service('~charge_battery', Trigger, self.service_cmd_charge_cb)

        self.command_action_server = actionlib.SimpleActionServer('~action', RobotSimpleCommandAction, None, False)
        self.command_action_server.register_goal_callback(self.action_goal_cb)
        self.command_action_server.register_preempt_callback(self.action_preempt_cb)
        self.command_action_server.start()
        # simple action client to call the local action server
        self.command_action_server_client = actionlib.SimpleActionClient('~action', RobotSimpleCommandAction)

        self.action_client = actionlib.SimpleActionClient(self.action_client_namespace, RobotSimpleCommandAction)

        self.get_sequences_service = rospy.Service(self.get_sequences_name,
                                                  GetCommandManagerList,
                                                  self.get_sequences)

        self.add_sequence_service = rospy.Service(self.add_sequence_name,
                                                  ManageCommandManager,
                                                  self.add_sequence)

        self.remove_sequence_service = rospy.Service(self.remove_sequence_name,
                                                  ManageCommandManager,
                                                  self.remove_sequence)

        return 0

    def init_state(self):
        """ Actions perfomed in init state"""

        if not(self.sequences_already_read):
            self.load_sequences(self.sequences_file)
            self.sequences_already_read = True

        if self.action_client.wait_for_server(rospy.Duration.from_sec(2)) == True:
            self.switch_to_state(State.READY_STATE)
        else:
            rospy.loginfo_throttle(30, "%s::init_state: waiting for command server %s..." %
                                   (self._node_name, self.action_client_namespace))

    def load_sequences(self,file):
        try:
            self.sequences_dict = self.load_yaml_file(file)
            if self.sequences_dict == None:
                self.sequences_dict = {}
        except Exception as e:
            self.sequences_dict = {}
            rospy.logerr('%s::Unable to load sequences from %s: %s' % (self._node_name, self.sequences_file, e))
            return False
        return True


    def save_yaml_file(self,data_dict,file):
        with open(file, 'w+') as f:
            yaml.dump(data_dict, f,default_flow_style=False)

    def load_yaml_file(self,file):
        with open(file) as f:
            # use safe_load instead load
            data = yaml.safe_load(f)
            return data

    def add_sequence(self, msg):
        ret = ManageCommandManagerResponse()

        ret.ret.success=True
        ret.ret.message='OK. Adding sequence %s' % (msg.Command.id)
        try:
            #print (self.sequences_dict)
            #print (self.split_command_semicolon(msg.Command.command))
            self.sequences_dict[msg.Command.id]= self.split_command_semicolon(msg.Command.command)
            #print (self.sequences_dict)
            self.save_yaml_file(self.sequences_dict,self.sequences_file)
        except Exception as e:
            txt = '%s::Unable to add sequence. %s' % (self._node_name, e)
            rospy.logerr("%s",txt)
            ret.ret.success= False
            ret.ret.message = txt

        rospy.loginfo("%s",ret.ret.message)
        return ret


    def remove_sequence(self, msg):

        ret = ManageCommandManagerResponse()

        if self.command_action_server.is_active() == True:
            ret.ret.success=False
            ret.ret.message = '%s::Unable to remove sequences if any sequence is active ' % (self._node_name)
        else:
            if msg.Command.id in self.sequences_dict:
                try:
                    del self.sequences_dict[msg.Command.id]
                    self.save_yaml_file(self.sequences_dict,self.sequences_file)
                    ret.ret.success=True
                    ret.ret.message='%s::OK Removing sequence %s' % (self._node_name,msg.Command.id)
                except :
                    ret.ret.success=False
                    ret.ret.message = '%s::Unable to remove sequence' % (self._node_name)
            else:
                ret.ret.success=False
                ret.ret.message = '%s::Unable to remove unexistent sequence %s' % (self._node_name,msg.Command.id)

        if ret.ret.success:
            rospy.loginfo("%s",ret.ret.message)
        else:
            rospy.logerr("%s",ret.ret.message)
        return ret


    def dict_to_list(self):
        l = CommandManagerArray()

        for seq in self.sequences_dict:
            C = CommandManager()
            C.id = seq
            bStart = True
            for c in self.sequences_dict[seq]:
                if (bStart):
                    C.command = c
                    bStart = False
                else:
                    C.command = C.command + " ; " + c
            l.commands.append(C)
        return l


    def get_sequences(self, msg):

        ret = GetCommandManagerListResponse()
        ret.list = self.dict_to_list()

        ret.success = True
        return ret

    def publish_log(self):
        self.log.node_name = self._node_name
        self.log.process_name = self.current_sequence_list_id
        self.log.command = self.current_command
        self.command_log_publisher.publish(self.log)

    def ready_state(self):
        """Actions performed in ready state"""

        self.current_sequence_string = '-Seq_id: %s Command: %s Remaining Commands:%d ' % (self.current_sequence_list_id,
                                                                                           self.current_command,
                                                                                           self.current_sequence_list_remaining)
        #self.current_sequence_publisher.publish(self.current_sequence_string)

        if self.command_action_server.is_active() == True:                  # Action server running
            if self.current_sequence_list == None:                           # No sequence active
                self.error_detected = False
                # Clearing sequence error execution
                self.error_sequence_executed = False
                if self.sequences_list_queue.empty() == False:               # There are pending sequences in the queue
                    self.current_sequence_list = self.sequences_list_queue.get()
                    self.current_sequence_list_id = self.current_sequence_list.get_id()
                    self.current_sequence_list_remaining = self.current_sequence_list.get_count_remaining_commands()
                    rospy.loginfo('%s::ready_state: Getting new sequence (%s) from the queue. %d remaining in the queue',
                                  self._node_name, self.current_sequence_list_id, self.sequences_list_queue.qsize())
                    self.log.status = "Getting Next Sequence"
                    self.log.result = "OK"
                    self.log.error = False
                # FINISH!
                else:      # No more sequences
                    rospy.loginfo('%s::ready_state: the sequence queue is empty!', self._node_name)
                    self.finish_action(StatusCodes.SUCCEEDED)
                    self.log.status = "No More Sequences"
                    self.log.result = "OK"
                    self.log.error = False
            else:
                # No commands in execution
                if self.current_command == '':

                    current_command = self.current_sequence_list.get_next_command()

                    if current_command != None:
                        self.current_command = current_command
                        # SEND THE ACTION
                        ret, ret_msg = self.send_cmd(self.current_command)
                        if ret == True:
                            rospy.loginfo('%s::ready_state: sending new command: %s',
                                          self._node_name, self.current_command)
                            self.log.status = "Send"
                            self.log.result = "OK"
                            self.log.error = False
                        else:
                            rospy.logerr('%s::ready_state: error sending command: %s. Finishing sequence',
                                         self._node_name, self.current_command)
                            self.log.status = "End"
                            self.log.result = "Error Sending Command:"+ret_msg
                            self.log.error = True
                            self.finish_action(StatusCodes.FAILED)

                    else:  # The sequence is finished

                        # If the sequence is in loop
                        if self.current_sequence_list.is_loop() == True:
                            rospy.logwarn('%s::ready_state: the sequence (%s) has finished but it is in loop mode! (%d cycles)',
                                          self._node_name, self.current_sequence_list.get_id(), self.current_sequence_list.get_loop_cycles())
                            self.current_sequence_list.reset()
                            self.log.status = "Loop"
                            self.log.result = ""
                            self.log.error = False
                        else:
                            rospy.loginfo('%s::ready_state: the sequence (%s) has finished!',
                                          self._node_name, self.current_sequence_list.get_id())
                            self.log.status = "End Sequence"
                            self.log.result = "End Sequence"
                            self.log.error = False
                            self.current_sequence_list = None

                else:   # Waits  for the end of the command
                    self.update_feedback()
                    self.publish_feedback()
                    self.current_sequence_list_remaining = self.current_sequence_list.get_count_remaining_commands()
                    rospy.loginfo_throttle(5, '%s::ready_state: waiting for the end of the command: %s' %
                                           (self._node_name, self.current_command))

                    action_client_state = self.action_client.get_state()
                    communication_timeout = ((rospy.Time.now() - self.last_time_received_action_client_feedback)
                                             >= self.communication_timeout)

                    if communication_timeout == True:

                        rospy.logerr('%s::ready_state: Timeout getting feedback from the client. Command %s. Finishing sequence',
                                     self._node_name, self.current_command)
                        self.finish_action(StatusCodes.FAILED, msg='Timeout: No response from the action client')
                        self.log.status = "End"
                        self.log.result = "Timeout Error"
                        self.log.error = True

                    elif action_client_state != GoalStatus.ACTIVE and action_client_state != GoalStatus.PENDING:

                        action_client_result = self.action_client.get_result()

                        if action_client_result == None:  # If nothing is received from get_result
                            if action_client_state == GoalStatus.SUCCEEDED:
                                self.log.status = "End"
                                self.log.result = "End Sequence command: " + self.current_command
                                self.log.error = False
                                rospy.loginfo('%s::ready_state: the command (%s) has finished correctly!',
                                              self._node_name, self.current_command)
                                self.current_command = ''
                            else:
                                rospy.logerr('%s::ready_state: the command (%s) has finished wrongly (goal_status = %d)!',
                                             self._node_name, self.current_command, action_client_state)
                                self.finish_action(StatusCodes.FAILED, msg="goal_status = %d" % (action_client_state))

                                self.log.status = "End"
                                self.log.result = "Command Finished wrongly: " + self.current_command
                                self.log.error = True
                                self.current_command = ''
                                self.error_detected = True

                        elif action_client_result.result.success == True:

                            self.log.status = "End"
                            self.log.result = "End Sequence: " + self.current_command
                            self.log.error = False
                            rospy.loginfo('%s::ready_state: the command (%s) has finished correctly!',
                                          self._node_name, self.current_command)
                            self.current_command = ''
                        else:
                            rospy.logerr('%s::ready_state: the command (%s) has finished wrongly (code = %d, msg = %s)!',
                                         self._node_name, self.current_command, action_client_result.result.code, action_client_result.result.message)
                            #self.finish_action(StatusCodes.FAILED, msg=action_client_result.result.message)

                            self.log.status = "End"
                            self.log.result = "Command: "+ self.current_command +" Finished wrongly: " + action_client_result.result.message
                            self.log.error = True
                            self.current_command = ''
                            self.error_detected = True
            self.publish_log()
        else:
            self.publish_log()
            self.log.status = ""
            self.log.result = ""
            self.log.error = False
            self.last_sequence_list_id = self.current_sequence_list_id
            self.current_sequence_list_id=''

        if self.error_detected:
            # Error detected
            if self.allow_error_sequence:
                if not(self.error_sequence_executed):
                    if not(self.command_action_server.is_active()):
                        #self.current_sequence_list.reset()
                        self.current_sequence_list_id = self.last_sequence_list_id + "_ERROR"
                        #self.current_sequence_list = SequenceList(self.sequences_dict[self.current_sequence_list_id])
                        rospy.logerr('%s::Error detected, executing sequence %s' % (self._node_name,self.current_sequence_list_id))
                        cmd = CommandString()
                        cmd.command = self.current_sequence_list_id
                        self.service_cmd_cb(cmd)
                        self.error_sequence_executed=True
                        self.log.status = "End"
                        self.log.result = "Executing Error Sequence: " + self.current_sequence_list_id
                        self.log.error = True
                    else:
                        self.topic_cmd_cancel_cb("")
            else:
                self.error_detected = False

        if self.charge_battery_received:
            # Battery Charge Received
           if not(self.command_action_server.is_active()):
                self.current_sequence_list_id = self.current_sequence_list_id + "_BATLOW"
                rospy.logerr('%s::Bat Charge Rec, executing sequence %s' % (self._node_name,self.current_sequence_list_id))
                cmd = CommandString()
                cmd.command = self.current_sequence_list_id
                self.service_cmd_cb(cmd)
                self.charge_battery_received=False
                self.log.status = "End"
                self.log.result = "Executing Battery Charge Sequence: " + self.current_sequence_list_id
                self.log.error = False


    def shutdown(self):
        """Shutdowns device

        Return:
            0 : if it's performed successfully
            -1: if there's any problem or the component is running
        """
        self.command_sub.unregister()

        return RComponent.shutdown(self)

    def switch_to_state(self, new_state):
        """Performs the change of state"""

        return RComponent.switch_to_state(self, new_state)

    def topic_cmd_cb(self, msg):
        """Subscriber command_sub callback

        Forward the message to the action server

        Args:
            msg : CommandString object that contains the user command
        """
        if self._state != State.READY_STATE:
            rospy.logerr_throttle(5, '%s::topic_cmd_cb: command not accepted because the component is not READY' % (
                self._node_name))
            return

        if self.command_action_server.is_active() == True and self.allow_queuing == False:
            return

        is_valid, ret_msg, parsed_command = self.validate_command(msg.command)
        if is_valid == True:
            self.forward_command_to_local_action_server(msg.command)
        else:
            rospy.logerr_throttle(5, '%s::topic_cmd_cb: command %s is not valid: %s' % (
                self._node_name, msg.command, ret_msg))

    def topic_robot_status_cb(self, msg):
        if (self.use_robot_status_emergency_stop):
            #rospy.loginfo('%s:: Using safety: %d' % (self._node_name, self.use_safety_emergency_stop))
            if (msg.robot_status.emergency_stop):
                #rospy.loginfo('%s:: Using safety: %d ---------' % (self._node_name, self.use_safety_emergency_stop))
                if (not self.emergency_detected):
                    self.topic_cmd_cancel_cb("")
                    self.emergency_detected = True
            else:
                self.emergency_detected = False

    def topic_safety_emergency_stop_cb(self, msg):

        if (self.use_safety_emergency_stop):
            if (msg.data):
                if (not self.emergency_detected):
                    self.topic_cmd_cancel_cb("")
                    self.emergency_detected = True
            else:
                self.emergency_detected = False

    def topic_cmd_cancel_cb(self, msg):
        """Subscriber command_sub_cancel callback

        Forwards the cancel to the action server

        Args:
            msg : Empty object to cancel the current command
        """
        if self.command_action_server.is_active():
            self.cancel_command_of_local_action_server()

    def service_cmd_cb(self, msg):
        """Service command_service callback

        Forward the message to the send_cmd method

        Args:
            msg : SetCommandString object that contains the user command

        Return:
            An SetCommandStringResponse that conatins a bool and str
        """

        res = SetCommandStringResponse()

        if self._state != State.READY_STATE:
            res.ret.success = False
            res.ret.code = StatusCodes.REJECTED
            res.ret.message = "New command not accepted because the component is not READY"
            return res

        if self.command_action_server.is_active() == True and self.allow_queuing == False:
            res.ret.success = False
            res.ret.code = StatusCodes.REJECTED
            res.ret.message = "New command not allowed while another command is running"
            return res

        is_valid, ret_msg, parsed_command = self.validate_command(msg.command)
        if is_valid == True:
            self.forward_command_to_local_action_server(msg.command)
            res.ret.code = StatusCodes.ACTIVE
            res.ret.success = True
            res.ret.message = 'Command received and processed'
        else:
            res.ret.success = False
            res.ret.code = StatusCodes.REJECTED
            res.ret.message = ret_msg

        return res

    def service_cmd_cancel_cb(self, msg):
        """Service command_service_cancel callback

        Runs cancel_cmd method when recieves a cancel msg

        Args:
            msg : Trigger object to cancel the current command

        Return:
            A TriggerResponse object that contains a bool and a str
        """
        ret = TriggerResponse()

        if self.command_action_server.is_active():
            self.cancel_command_of_local_action_server()
            ret.success = True
            ret.message = 'Command correctly cancelled'
        else:
            ret.success = False
            ret.message = 'No command running to cancel'

        return ret

    def service_cmd_charge_cb(self, msg):
        """Service command_service_charge callback

        Runs charge battery sequence

        Args:
            msg : Trigger object to run charge battery sequence

        Return:
            A TriggerResponse object that contains a bool and a str
        """
        ret = TriggerResponse()

        if self.allow_charge_battery_sequence:
            self.charge_battery_received = True
            ret.success = True
            ret.message = 'Command Battery Charge OK Rec'
        else:
            ret.success = False
            ret.message = 'No command Battery Charge Allowed, check complex sequencer params'

        return ret

    def action_goal_cb(self):
        """Action server goal callback

        Accepts the new goal if not command running.  Rejects the incoming
        goal if a command is running
        """
        if self.command_action_server.is_active() == False:
            goal = self.command_action_server.accept_new_goal()
            if self._state != State.READY_STATE:
                rospy.logerr('%s::action_goal_cb: command not accepted because the component is not READY' %
                             (self._node_name))
                result = RobotSimpleCommandResult()
                result.result.command = goal.command.command
                result.result.success = False
                result.result.message = ret_msg
                result.result.code = StatusCodes.REJECTED
                self.command_action_server.set_aborted(result=result, text=result.result.message)

            is_valid, ret_msg, parsed_command = self.validate_command(goal.command.command)
            if is_valid == True:
                ret, ret_msg = self.process_command(parsed_command)
                if ret == True:
                    rospy.loginfo('%s::action_goal_cb: command %s processed. Ret = (%s, %s)',
                                  self._node_name, goal.command.command, ret, ret_msg)
                else:
                    # Aborting the action
                    rospy.logerr('%s::action_goal_cb: command %s is not valid: %s' %
                                 (self._node_name, goal.command.command, ret_msg))
                    result = RobotSimpleCommandResult()
                    result.result.command = goal.command.command
                    result.result.success = False
                    result.result.message = ret_msg
                    result.result.code = StatusCodes.REJECTED
                    self.command_action_server.set_aborted(result=result, text=result.result.message)
            else:
                # Aborting the action
                rospy.logerr('%s::action_goal_cb: command %s is not valid: %s' %
                             (self._node_name, goal.command.command, ret_msg))
                result = RobotSimpleCommandResult()
                result.result.command = goal.command.command
                result.result.success = False
                result.result.message = ret_msg
                result.result.code = StatusCodes.REJECTED
                self.command_action_server.set_aborted(result=result, text=result.result.message)
        else:

            goal = self.command_action_server.accept_new_goal()

            if self.allow_queuing == True:
                is_valid, ret_msg, parsed_command = self.validate_command(goal.command.command)
                if is_valid == True:
                    ret, ret_msg = self.process_command(parsed_command)
                    if ret == True:
                        rospy.loginfo('%s::action_goal_cb: command %s processed. Ret = (%s, %s)',
                                      self._node_name, goal.command.command, ret, ret_msg)
                    else:
                        # Aborting the action
                        rospy.logerr('%s::action_goal_cb: command %s is not valid: %s' %
                                     (self._node_name, goal.command.command, ret_msg))
                else:
                    # Not valid
                    rospy.logerr('%s::action_goal_cb: command %s is not valid: %s' %
                                 (self._node_name, goal.command.command, ret_msg))
            # do nothing -> discards the command
            else:
                rospy.logwarn(
                    '%s::action_goal_cb: New command not allowed while another command is running', self._node_name)

    def action_preempt_cb(self):
        """Action server preempt callback

        Cancels the current active goal or ignore the incoming goal if
        the preempt request has been triggered by new goal available.
        """

        if self.command_action_server.is_active():
            has_new_goal = self.command_action_server.is_new_goal_available()
            # If preempt request by new action
            if has_new_goal == False:
                self.cancel_cmd()
            else:
                rospy.logwarn(
                    '%s::action_preempt_cb: preemption due to a new command is not allowed', self._node_name)
        else:
            rospy.logwarn('%s::action_preempt_cb: No command is running', self._node_name)

    # END CALLBACKS TO MANAGE CLIENT INPUT

    def send_cmd(self, msg):
        """Sends the command to the command server

        Args:
            msg : String that contains the user command

        Return:
            Returns a tuple (bool, str) that shows if the command has been
            correctly sent
        """

        command_sent = True
        ret_msg = 'OK'

        goal = RobotSimpleCommandGoal()
        goal.command.command = msg

        try:
            self.action_client.send_goal(goal, feedback_cb=self.action_client_feedback_cb)
            self.last_time_received_action_client_feedback = rospy.Time.now()
        except Exception as e:
            command_sent = False
            msg = 'Error sending command %s: %s' % (msg, e)
            rospy.logerr('%s::send_cmd: %s' % (self._node_name, ret_msg))

        return (command_sent, ret_msg)

    def cancel_cmd(self):
        """Cancels the current command running in the remote action server and finish the local action

        Return:
            True if the command has been correctly cancelled.
            False otherwihse.
        """
        self.action_client.cancel_all_goals()

        self.finish_action(StatusCodes.CANCELLED)

        rospy.logwarn('%s::cancel_cmd: Command %s correctly cancelled' %
                      (self._node_name, self.current_command))
        return True

    def split_command(self, command):
        """Splits the command from string to list of strings

        Splits the msg string to a list.

        Args:
            msg : string that contains the command

        Return:
            A list with the current command
        """
        return re.sub(' +', ' ', command).strip().split(" ")

    def split_command_semicolon(self, command):
        """Splits the command from string to list of strings

        Splits the msg string to a list.

        Args:
            msg : string that contains the command

        Return:
            A list with the current command
        """
        return re.sub(' +', ' ', command).strip().split(";")

    def cmd_is_active(self):
        """Checks if the current command is active

        Return:
            Returns a bool that shows if the command is active
        """
        # is_active = self.handlers_interfaces[self.current_handler].is_active()
        is_active = False
        return is_active

    def update_feedback(self):
        """Updates feedback messages

        Updates both feedback_msg and feedback_action_msg with information
        provided by the current_handler
        """
        current_sequence_list = self.current_sequence_list.get_current_sequence_id()
        self.feedback_msg.command = '%s:%s' % (current_sequence_list, self.current_command)
        try:
            self.feedback_msg.status = self.action_client_feedback_msg.feedback.status
        except:
            self.feedback_msg.status = "Error"

        self.feedback_action_msg.feedback = self.feedback_msg

    def publish_feedback(self):
        """Publishes the updated feedback
        """
        self.command_feedback_pub.publish(self.feedback_msg)
        self.command_action_server.publish_feedback(self.feedback_action_msg)

    def reject_goal(self, command):
        result = RobotSimpleCommandResult()
        result.result.command = command  # self.current_command
        result.result.success = False
        result.result.message = 'This goal has been rejected. Probably because ' +\
            'other command is running. Cancel before ' +\
            'sending other goal.'
        result.result.code = StatusCodes.REJECTED

        return

    def finish_action(self, code, msg=''):
        """Finishes the current action

        Args:
            code : int value codified as StatusCodes object
            msg: string to set the message of the result
        """
        result = RobotSimpleCommandResult()
        result.result.command = self.current_sequence_list_id
        result.result.message = msg
        result.result.code = code

        if code == StatusCodes.CANCELLED:
            result.result.success = False
            self.command_action_server.set_preempted(result=result, text=result.result.message)
        elif code == StatusCodes.SUCCEEDED:
            result.result.success = True
            self.command_action_server.set_succeeded(result=result, text=result.result.message)
        elif code == StatusCodes.FAILED:
            result.result.success = False
            self.command_action_server.set_aborted(result=result, text=result.result.message)
        else:
            result.result.success = False
            rospy.logwarn(
                '%s::finish_action: the code passed is not being handled! Setting the action as aborted', self._node_name)
            self.command_action_server.set_aborted(result=result, text=result.result.message)

        self.clear_command()

    def clear_command(self):
        """Reset some internal variables"""
        self.feedback_msg = CommandStringFeedback()
        self.feedback_action_msg = RobotSimpleCommandFeedback()
        self.current_command = ''
        self.current_sequence_list = None
        self.sequences_list_queue = Queue.Queue(self.maxsize_queue)

    def forward_command_to_local_action_server(self, command):
        """Sends the command to itself to the action server

        Args:
            command : string

        Return:
            0 if OK
            -1 if error
        """
        goal = RobotSimpleCommandGoal()
        goal.command.command = command
        self.command_action_server_client.send_goal(goal)

        return 0

    def cancel_command_of_local_action_server(self):
        """Cancels current action

        Args:


        Return:
            0 if OK
            -1 if error
        """

        self.command_action_server_client.cancel_all_goals()

        return 0

    def validate_command(self, command):
        """Checks if the command is valid (tag exists, number of arguments is correct)

        Args:
            command: string with command

        Return:
            Returns a tuple of (bool, msg, string[])
            True,msg,[parsed_command] if it is valid
            False,msg,[] if it is NOT valid
        """
        parsed_command = self.split_command(command)
        keyword_counter = 0

        for cmd in parsed_command:
            if cmd in self._sequence_reserved_keywords:
                keyword_counter += 1
            elif cmd not in self.sequences_dict:
                return False, "The sequence %s does not exist" % (cmd), []

        # At least one
        if len(parsed_command) < 1 + keyword_counter:
            return False, "Invalid command", []

        return True, "OK", parsed_command

    def process_command(self, command):
        """Processes the command and put it in the execution queue

        Args:
            command: string[] with valid sequences

        Return:
            Returns a tuple of (bool, msg)
            True,msg if it is valid
            False,msg if it is NOT valid
        """
        keyword_counter = 0

        try:
            sequence_list = SequenceList(command)
            for cmd in command:
                if cmd == 'LOOP':
                    sequence_list.set_loop(True)
                if cmd in self._sequence_reserved_keywords:
                    keyword_counter += 1
                else:
                    command_list = self.sequences_dict[cmd]
                    sequence_list.append(cmd, command_list)
                    rospy.loginfo('%s::process_command: sequence %s -> %s', self._node_name, cmd, str(command_list))
        except KeyError as e:
            return False, "Sequence %s is not valid: %s" % (cmd,e)

        # At least one
        if len(command) < 1 + keyword_counter:
            return False, "Empty command"

        if sequence_list.is_loop() == True and len(command) < 2:
            return False, "Empty command"

        try:
            self.sequences_list_queue.put(sequence_list)
        except Queue.Full as e:
            return False, "Queue of sequences is full: %s"%(e)

        return True, "OK"

    def action_client_feedback_cb(self, feedback):
        """
            Callback for the feedback from the action client
        """
        self.action_client_feedback_msg = feedback
        self.last_time_received_action_client_feedback = rospy.Time.now()
        # self.parse_feedback()

    def validate_sequence_keywords(self):
        """
            Checks that the sequences id are not reserved keywords
            In case there were, it removes them
        """
        keys_to_remove = []
        for sequence in self.sequences_dict:
            if sequence in self._sequence_reserved_keywords:
                keys_to_remove.append(sequence)

        for sequence in keys_to_remove:
            rospy.logerr(
                '%s::validate_sequence_keywords: the sequence %s is a keyword that cannot be used. It has been removed', self._node_name, sequence)
            del self.sequences_dict[sequence]
