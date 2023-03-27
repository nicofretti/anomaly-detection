#!/usr/bin/env python
import re

import rospy
import actionlib

from std_msgs.msg import String, Empty
from std_srvs.srv import Trigger, TriggerResponse

from robot_simple_command_manager_msgs.msg import *
from robot_simple_command_manager_msgs.srv import *

from rcomponent.rcomponent import *


class SimpleCommandManager(RComponent):
    """
    A class used to load simple command handlers

    ...

    Attributes
    ----------
    feedback_msg : CommandStringFeedback
        structured feedback message that contains the current_command and a
        status
    feedback_action_msg : RobotSimpleCommandFeedback
        structured feedback message for action server. Contains the feedback_msg
    action_command_sent : bool
        flag to monitor if a command has been sent through the action server
    command_sent : bool
        flag to monitor if a command has been sent
    current_command : str
        command list codified as string received as user input by topic, service
        or action (i.e: TAG arg1 arg2)
    current_handler : str
        equals to the TAG of the current command
    command_name : str
        name of the service and topic used to received commands. If the 'command_name'
        ROS param is not set, the default value is used 'command'.
    desired_handlers : list
        list of strings that contains the name (TAG) of the handlers that will
        be loaded. If the 'command_handlers' ROS param is not set, the default
        value is used []
    """

    def __init__(self):
        self.feedback_msg = CommandStringFeedback()
        self.feedback_action_msg = RobotSimpleCommandFeedback()
        self.status_msg = CommandManagerStatus()

        #self.action_command_sent = False
        #self.command_sent = False

        self.current_command = ''
        self.current_handler = ''

        self.command_name = 'command'
        self.desired_handlers = []
        self.pending_handlers = []

        self.goal = None
        self.cancel_request = False

        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)

        self.command_name = rospy.get_param('~command_name', self.command_name)

        self.desired_handlers = rospy.get_param('~command_handlers', self.desired_handlers)
        self.pending_handlers = rospy.get_param('~command_handlers', self.desired_handlers)

    def ros_setup(self):
        """Creates and inits ROS components"""

        RComponent.ros_setup(self)

        self.command_sub = rospy.Subscriber(
            '~' + self.command_name, CommandString, self.topic_cmd_cb)
        self.command_cancel_sub = rospy.Subscriber('~cancel', Empty, self.topic_cmd_cancel_cb)

        self.command_status_pub = rospy.Publisher(
            '~status', CommandManagerStatus, queue_size=1)
        self.command_feedback_pub = rospy.Publisher(
            '~feedback', CommandStringFeedback, queue_size=1)

        self.command_service = rospy.Service(
            '~' + self.command_name, SetCommandString, self.service_cmd_cb)
        self.command_cancel_service = rospy.Service('~cancel', Trigger, self.service_cmd_cancel_cb)
        self.get_handlers_service = rospy.Service('~get_handlers', GetHandlers, self.get_handlers_cb)
        self.get_handler_info_service = rospy.Service('~get_handler_info', GetHandlerInfo, self.get_handler_info_cb)

        self.command_action_server = actionlib.SimpleActionServer(
            '~action', RobotSimpleCommandAction, None, False)
        self.reseted_goal = self.command_action_server.next_goal
        self.command_action_server.register_goal_callback(self.action_goal_cb)
        self.command_action_server.register_preempt_callback(self.action_preempt_cb)
        self.command_action_server.start()
        # simple action client to call the local action server
        self.command_action_server_client = actionlib.SimpleActionClient(
            '~action', RobotSimpleCommandAction)

        self.handlers_params = {}
        self.handlers_interfaces = {}
        self.loaded_handlers = []
        self.load_handlers()

        return 0

    def ready_state(self):
        """Actions performed in ready state"""

        if self.command_action_server.is_active() == True:
            if self.cmd_is_active() == False:
                if self.handlers_interfaces[self.current_handler].has_succeeded() == True:
                    self.finish_action(StatusCodes.SUCCEEDED)
                else:
                    rospy.logerr('%s::ready_state: command execution of %s failed',
                                 self._node_name, self.current_command)
                    self.finish_action(StatusCodes.FAILED)
                self.clear_command()
            else:

                if self.cancel_request == True:
                    self.cancel_cmd()
                    self.cancel_request = False
                else:
                    self.update_feedback()
                    self.update_status(self.current_command, self.feedback_msg.status, StatusCodes.ACTIVE)
                    self.publish_feedback()

        elif self.command_action_server.is_new_goal_available() == True:
            goal = self.command_action_server.accept_new_goal()
            # Before sending the command to the handler, it is validated
            is_valid, ret_msg = self.validate_command(goal.command.command)
            if is_valid == True:
                self.goal = goal
                ret, ret_msg = self.send_cmd(self.goal.command)
                if ret != True:
                    rospy.logerr('%s::ready_state: error sending the command %s' %
                                 (self._node_name, self.goal.command.command))
                    result = RobotSimpleCommandResult()
                    result.result.command = self.goal.command.command
                    result.result.success = False
                    result.result.message = ret_msg
                    result.result.code = StatusCodes.FAILED
                    self.command_action_server.set_aborted(
                        result=result, text=result.result.message)
                else:
                    rospy.loginfo('%s::ready_state: command %s sent correctly' %
                                 (self._node_name, self.goal.command.command))
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

    def ros_publish(self):
        self.command_status_pub.publish(self.status_msg)

    def load_handlers(self, handler_list=None):
        """Loads the configured handlers"""
        if handler_list is None:
            handler_list = self.desired_handlers

        error_msg = ''
        lodaded_message = self._node_name + '::load_handlers: Loaded handlers: ' + str(self.loaded_handlers)

        for handler_name in handler_list:
            if rospy.has_param('~'+handler_name):
                self.handlers_params[handler_name] = rospy.get_param('~'+handler_name)

                try:
                    handler_interface = self.load_handler_interface(handler_name)
                    self.handlers_interfaces[handler_name] = handler_interface
                    self.loaded_handlers.append(handler_name)
                    rospy.loginfo('%s::load_handlers: Handler %s loaded.'
                                  % (self._node_name, handler_name))

                    lodaded_message = self._node_name + '::load_handlers: Loaded handlers: ' + str(self.loaded_handlers)
                    rospy.loginfo(lodaded_message)
                except Exception as identifier:
                    error_msg += '\nFailed to load handler ' + handler_name + ' -> ' + str(identifier)

            else:
                rospy.logerr('%s::load_handlers: Handler %s parameters not found.\
                             ' % (self._node_name, handler_name))

        rospy.loginfo_throttle(5, lodaded_message)
        if error_msg != '':
            rospy.logerr_throttle(5, '%s::load_handlers: %s' % (self._node_name, error_msg))

        self.pending_handlers = list(set(self.desired_handlers) - set(self.loaded_handlers))
        if len(self.pending_handlers) > 0:
            reload = threading.Timer(1, self.reload_handlers)
            # Avoids the timer when the node is being shutdown
            if rospy.is_shutdown() == False:
                reload.start()

        self.status_msg.loaded_commands = self.loaded_handlers
        self.status_msg.paused_commands = self.pending_handlers

        return True

    def reload_handlers(self):
        self.load_handlers(self.pending_handlers)

    def load_handler_interface(self, handler_name):
        """Load the desired handler given its handler_name

        Reads the parameters of the desired handler and loads its python module.

        Args:
            handler_name : string that indentifies the handler that is going to be loaded.
                Used to get the parameters from the handlers_params dict.

        Return:
            An object of the specific handler type
        """
        parameters = self.handlers_params[handler_name]

        type_list = parameters['type'].split('/')

        if len(type_list) == 2:
            handler_module_prefix = ""
            handler_module = type_list[0]
            handler_class = type_list[1]
        elif len(type_list) == 3:
            handler_module_prefix = type_list[0] + "."
            handler_module = type_list[1]
            handler_class = type_list[2]

        if 'action_interface' in handler_module:
            handler_module = handler_module_prefix + 'handlers.actions.' + handler_module
        elif 'service_interface' in handler_module:
            handler_module = handler_module_prefix + 'handlers.services.' + handler_module
        elif 'procedure_interface' in handler_module:
            handler_module = handler_module_prefix + 'handlers.procedures.' + handler_module
        elif 'subscriber_interface' in handler_module:
            handler_module = handler_module_prefix + 'handlers.subscribers.' + handler_module
        else:
            # TODO: raise exception or return None
            print('Unknown')
        # This should be tested. This may raise an exception if the
        # python module imported does not exists
        module = __import__(handler_module, fromlist=['object'])
        handler_interface = getattr(module, handler_class)

        return handler_interface(handler_name, parameters)

    # START CALLBACKS TO MANAGE CLIENT INPUT

    def topic_cmd_cb(self, msg):
        """Subscriber command_sub callback

        Forward the message to the action server

        Args:
            msg : CommandString object that contains the user command
        """
        # New commands are not allowed if the component is not ready
        if self._state != State.READY_STATE:
            msg = 'command not accepted because the component is not READY'
            rospy.logerr_throttle(5, '%s::topic_cmd_cb: %s' % (self._node_name, msg))
            return

        self.forward_command_to_local_action_server(msg.command)

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

        message = ''
        code = 0
        # New commands are not allowed if the component is not ready
        if self._state != State.READY_STATE:
            res.ret.success = False
            code = StatusCodes.REJECTED
            res.ret.code = code
            message = 'New command not accepted because the component is not READY'
            res.ret.message = message
            rospy.logerr('%s::service_cmd_cb: %s' % (self._node_name, msg))
            self.update_status(msg.command, message, code)

            return res

        # New commands are not allowed if there is other command running
        if self.command_action_server.is_active() == True:
            res.ret.success = False
            code = StatusCodes.REJECTED
            res.ret.code = code
            message = 'New command not allowed while another command is running'
            res.ret.message = message
            rospy.logerr('%s::service_cmd_cb: %s' % (self._node_name, message))
        else:
            # Before forwarding the command, it is validated
            is_valid, ret_msg = self.validate_command(msg.command)
            if is_valid == True:
                self.forward_command_to_local_action_server(msg.command)
                code = StatusCodes.ACTIVE
                res.ret.code = code
                res.ret.success = True
                message = 'Command received and processed'
                res.ret.message = message
            else:
                res.ret.success = False
                code = StatusCodes.REJECTED
                res.ret.code = code
                message = ret_msg
                res.ret.message = ret_msg
                rospy.logerr('%s::service_cmd_cb: %s' % (self._node_name, message))

        self.update_status(msg.command, message, code)

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

    def action_goal_cb(self):
        """Action server goal callback

        Accepts the new goal if not command running.  Rejects the incoming
        goal if a command is running
        """
        if self.command_action_server.is_active() == False:

            # New commands are not allowed if the component is not ready
            if self._state != State.READY_STATE:

                msg = 'New command not accepted because the component is not READY'
                rospy.logerr('%s::action_goal_cb: %s' % (self._node_name, msg))
                result = RobotSimpleCommandResult()
                result.result.command = goal.command.command
                result.result.success = False
                result.result.message = msg
                result.result.code = StatusCodes.REJECTED
                self.command_action_server.set_aborted(result=result, text=result.result.message)

        else:
            # do nothing -> discards the command
            msg = 'New command not allowed while another command is running'
            rospy.logwarn(
                '%s::action_goal_cb: %s', self._node_name, msg)
            goal = self.command_action_server.accept_new_goal()

    def action_preempt_cb(self):
        """Action server preempt callback

        Cancels the current active goal or ignore the incoming goal if
        the preempt request has been triggered by new goal available.
        """

        if self.command_action_server.is_active():
            has_new_goal = self.command_action_server.is_new_goal_available()
            # If preempt request by new action
            if has_new_goal == False:
                self.cancel_request = True
            else:
                rospy.logwarn(
                    '%s::action_preempt_cb: preemption due to a new command is not allowed', self._node_name)
        else:
            rospy.logwarn('%s::action_preempt_cb: No command is running', self._node_name)

    def get_handlers_cb(self, msg):
        res = GetHandlersResponse()
        res.handlers = self.loaded_handlers

        return res

    def get_handler_info_cb(self, msg):
        res = GetHandlerInfoResponse()
        res.success = False
        res.description = [msg.name + " is not loaded"]
        res.types = [msg.name + " is not loaded"]
        if msg.name in self.loaded_handlers:
            res.success = True
            res.types = self.handlers_interfaces[msg.name].get_info()[0]
            res.description = self.handlers_interfaces[msg.name].get_info()[1]

        return res

    # END CALLBACKS TO MANAGE CLIENT INPUT

    def send_cmd(self, msg):
        """Sends the command to the specific handler

        Args:
            msg : CommandString object that contains the user command

        Return:
            Returns a tuple (bool, str) that shows if the command has been
            correctly sent
        """
        self.current_command = msg.command
        parsed_command = self.split_command(msg.command)
        self.current_handler = parsed_command[0]
        command_sent = False
        ret_msg = ''

        try:
            command_sent,handler_msg = self.handlers_interfaces[self.current_handler].send_command(
                parsed_command[1:])

            if command_sent:
                ret_msg = 'Command %s sent!' % (msg.command)
                rospy.loginfo('%s::send_cmd: %s' % (self._node_name, ret_msg))
            else:
                ret_msg = 'Error sending the command %s: %s' % (msg.command, handler_msg)
                rospy.logerr('%s::send_cmd: %s: %s' % (self._node_name, ret_msg, handler_msg))

        except Exception as e:
            ret_msg = 'Error sending command %s' % (msg.command)
            rospy.logerr('%s::send_cmd: %s : %s' % (self._node_name, ret_msg, e))

        return (command_sent, ret_msg)

    def cancel_cmd(self):
        """Cancales the current command

        Return:
            True if the command has been correctly cancelled.
            False otherwihse.
        """
        # if self.command_sent == True:
        self.handlers_interfaces[self.current_handler].cancel_cmd()

        self.finish_action(StatusCodes.CANCELLED)

        rospy.logwarn('%s::cancel_cmd: Command %s correctly cancelled' %
                      (self._node_name, self.current_command))
        self.clear_command()
        return True
        # else:
        #    rospy.logwarn('%s::cancel_cmd: No cmd running.'%(self.command_name))
        #    return False

    def split_command(self, command):
        """Splits the command from string to list of strings

        Splits the msg string to a list.

        Args:
            msg : string that contains the command

        Return:
            A list with the current command
        """
        return re.sub(' +', ' ', command).strip().split(" ")

    def cmd_is_active(self):
        """Checks if the current command is active

        Return:
            Returns a bool that shows if the command is active
        """
        is_active = self.handlers_interfaces[self.current_handler].is_active()
        return is_active

    def update_feedback(self):
        """Updates feedback messages

        Updates both feedback_msg and feedback_action_msg with information
        provided by the current_handler
        """
        try:

            msg = self.handlers_interfaces[self.current_handler].get_feedback()
            self.feedback_msg.command = self.current_command
            self.feedback_msg.status = msg

            self.feedback_action_msg.feedback = self.feedback_msg
        except Exception as e:
            rospy.logerr('%s::update_feedback: %s', self._node_name, e)

    def update_status(self, command, msg, code):
        self.status_msg.loaded_commands = self.loaded_handlers
        self.status_msg.paused_commands = self.pending_handlers
        self.status_msg.command = command
        self.status_msg.code = self.code_to_string(code)
        self.status_msg.msg = msg

    def code_to_string(self, code):
        if code == StatusCodes.ACTIVE:
            return 'ACTIVE'
        elif code == StatusCodes.CANCELLED:
            return 'CANCELLED'
        elif code == StatusCodes.SUCCEEDED:
            return 'SUCCEEDED'
        elif code == StatusCodes.FAILED:
            return 'FAILED'
        elif code == StatusCodes.REJECTED:
            return 'REJECTED'
        else:
            return 'UNKNOWN'

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

        self.command_action_server.new_goal = False
        self.command_action_server.new_goal_preempt_request = False
        self.command_action_server.next_goal.set_rejected(result, "Goal rejected by the server")
        self.command_action_server.next_goal = self.reseted_goal
        return

    def finish_action(self, code):
        """Finishes the current action

        Args:
            code : int value codified as StatusCodes object
        """

        result = self.build_result(code)

        msg = ''
        if code == StatusCodes.CANCELLED:
            msg = "Cancelled by user"
            self.command_action_server.set_preempted(result=result, text=result.result.message)
        elif code == StatusCodes.SUCCEEDED:
            msg = result.result.message
            self.command_action_server.set_succeeded(result=result, text=msg)
        elif code == StatusCodes.FAILED:
            msg = result.result.message
            rospy.logerr('%s::finish_action: %s aborted. %s', self._node_name, self.current_command, msg)
            self.command_action_server.set_aborted(result=result, text=msg)
        else:
            rospy.logwarn(
                '%s::finish_action: the code passed is not being handled! Setting the action as aborted', self._node_name)
            self.command_action_server.set_aborted(result=result, text=result.result.message)
            msg = "Code no handled, command aborted"

        self.update_status(self.current_command, msg, code)

        # Why?
        '''if self.command_action_server.is_new_goal_available():
            self.command_action_server.accept_new_goal()
            self.command_action_server.set_preempted()'''

    def build_result(self, code):
        """Build the action result

        Args:
            code : int value codified as StatusCodes object

        Return:
            Returns a RobotSimpleCommandResult object that contains the
            current_comand, if the action has been correctly performed,
            a info message and a code
        """
        result = RobotSimpleCommandResult()
        result.result.command = self.current_command
        succeeded = self.handlers_interfaces[self.current_handler].has_succeeded()
        result.result.success = succeeded
        result.result.message = self.handlers_interfaces[self.current_handler].get_result()
        result.result.code = code

        return result

    def clear_command(self):
        """Reset some internal variables"""
        self.feedback_msg = CommandStringFeedback()
        self.feedback_action_msg = RobotSimpleCommandFeedback()
        #self.command_sent = False
        #self.action_command_sent = False
        self.current_command = ''
        self.current_handler = ''

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
            Returns a tuple of (bool, msg)
            True,msg if it is valid
            False,msg if it is NOT valid
        """
        parsed_command = self.split_command(command)
        # At least one
        if len(parsed_command) < 1:
            return False, "A command needs at least the command tag"

        parsed_handler = parsed_command[0]
        parsed_args = []

        if len(parsed_command) >= 2:
            parsed_args = parsed_command[1:]

        #rospy.loginfo('%s::validate_command::  HANDLER: [%s] - ARGS: %s'%(self._node_name, parsed_handler, parsed_args))

        if parsed_handler not in self.loaded_handlers:
            return False, "Command tag does not exist"

        try:
            converted_parsed_args = self.handlers_interfaces[parsed_handler].parse_args(parsed_args)
        except Exception as e:
            return False, str(e)

        '''
        if self.current_handler in self.loaded_handlers:

            try:
                command_sent = self.handlers_interfaces[self.current_handler].send_command(parsed_command[1:])
                msg = 'Command sent!'
                rospy.loginfo('%s::send_cmd: %s'%(self._node_name, msg))
        '''

        return True, "Command structure and types are correct"
