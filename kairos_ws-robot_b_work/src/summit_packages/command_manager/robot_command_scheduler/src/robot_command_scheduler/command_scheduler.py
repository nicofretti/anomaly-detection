#!/usr/bin/env python
import re

import rospy
import actionlib

from std_msgs.msg import String, Empty
from std_srvs.srv import Trigger, TriggerResponse

from robot_simple_command_manager_msgs.msg import *
from robot_simple_command_manager_msgs.srv import ManageCommandManager, ManageCommandManagerResponse

from robot_simple_command_manager_msgs.srv import GetCommandSchedulerList, GetCommandSchedulerListResponse
from robot_simple_command_manager_msgs.srv import AddSchedule
from actionlib_msgs.msg import GoalStatus
from rcomponent.rcomponent import *


import Queue
from datetime import date, datetime, timedelta

import yaml

"""
# Samples of schedules
#id: Day of Week(A or 0..6) Hour (A or 0..23) Minute (0..59) Sequence
#
# Example:
#
# Schedule S1: Run sequence M1 on Wednesday at 16:50
#  S1: 2 16 50 M1
#
# Schedule S2: Run sequence M2 Every Day at 16:50
#  S2: A 16 50 M2


"""

class Schedule():
    """
        A schedule is a command and a time
    """

    def __init__(self,id,wk,date_start,date_end,hour,minute,repeat_hour,enabled,commands):
        """
            Args:
                id: String
                commands: String[]
        """
        self.id = id
        self.wk = []
        for i in wk:
            self.wk.append(i)
        self.date_start = date_start
        self.date_end = date_end
        self.hour = hour
        self.minute =minute
        self.repeat_hour = repeat_hour
        self.enabled = enabled
        self.commands = commands
        # inits last run 5 min before now
        self.last_run = datetime.now() - timedelta(minutes=5)

        rospy.loginfo('Adding schedule id:%s,wk:%s,date_start:%s,date_end:%s,hour:%d,minute:%d,repeat_hour:%d,Enabled:%d,Commands:%s'%\
            (self.id,self.wk,self.date_start,self.date_end,self.hour,self.minute,self.repeat_hour,self.enabled,self.commands))

    def get_last_run(self):
        """
            Gets the last run
        """
        return self.last_run

    def get_commands(self):
        """
            Gets the command
        """
        return self.commands

    def update_commands(self,command):
        """
            Updates the command
        """
        self.commands = command


    def update_reph(self,repeat_hour):
        """
            Updates the command
        """
        self.repeat_hour = repeat_hour

    def enable(self,enable):
        """
            Updates enabled
        """
        self.enabled = enable

    def update_date_start(self,date):
        """
            Updates date_start
        """
        self.date_start = date

    def update_last_run(self):
        """
            Updates last run
        """
        self.last_run =  datetime.now()

    def update_hour(self,hour):
        """
            Updates hour
        """
        self.hour = hour

    def update_minute(self,minute):
        """
            Updates minute
        """
        self.minute = minute

    def update_date_end(self,date):
        """
            Updates date_end
        """
        self.date_end = date

    def update_dayofweek(self,wk):
        """
            Updates the dayofweek for the schedule
        """
        self.wk = []
        for i in wk:
            self.wk.append(i)

    def get_dayofweek(self):
        """
            Gets the dayofweek for the schedule
        """
        return self.wk

    def get_hour(self):
        """
            Gets the hour for the schedule
        """
        return self.hour

    def get_reph(self):
        """
            Gets the repeat_hour
        """
        return self.repeat_hour

    def get_enabled(self):
        """
            Gets the enabled
        """
        return self.enabled


    def get_minute(self):
        """
            Gets the minute for the schedule
        """
        return self.minute

    def get_id(self):
        """
            Gets the time
        """
        return self.id

    def get_date_start_datetime(self):
        """
            Gets the start date
        """
        try:
            date_time_obj = datetime.strptime(self.date_start, '%Y/%m/%d').date()
        except:
            date_time_obj = datetime.strptime("1900/01/01", '%Y/%m/%d').date()

        #print ("Start",date_time_obj)

        return date_time_obj

    def get_date_end_datetime(self):
        """
            Gets the end date
        """
        try:
            date_time_obj = datetime.strptime(self.date_end, '%Y/%m/%d').date()
        except:
            date_time_obj = datetime.strptime("2100/01/01", '%Y/%m/%d').date()

        #print ("End",date_time_obj)

        return date_time_obj

    def get_date_start(self):
        """
            Gets the start date
        """
        if self.date_start==None:
            return "1900/01/01"
        return self.date_start

    def get_date_end(self):
        """
            Gets the end date
        """
        if self.date_start==None:
            return "2100/01/01"
        return self.date_end

def split_string(command):
        """Splits the string to list of strings by spaces

        Splits the string to a list.

        Args:
            msg : string

        Return:
            A list with text splitted by spaces
        """
        return re.sub(' +', ' ', command).strip().split(" ")

class ScheduleList():
    """
        Manages the list of schedules
    """

    def __init__(self):
        """
            Args:
        """
        # Array of schedules
        self.schedules = []

    def append_from_dict_line(self,id,dict_line):
        wk = dict_line['wk']
        date_start=dict_line['date_start']
        date_end=dict_line['date_end']
        hour = dict_line['hour']
        minute = dict_line['minute']
        repeat_hour = dict_line['repeat_hour']
        enabled = dict_line['enabled']
        commands = dict_line['commands']
        return self.append(id,wk,date_start,date_end,hour,minute,repeat_hour,enabled,commands)


    def append_from_dict(self,dict_input):
        for key in dict_input.keys():
            self.append_from_dict_line(key,dict_input[key])

    def remove(self, schedule_id):
        for s in self.schedules:
            if (s.get_id()==schedule_id):
                self.schedules.remove(s)
                text = "Schedule:"+schedule_id+" Removed"
                rospy.loginfo("%s" %(text))
                return True,text
        text = "Schedule:"+schedule_id+" Not Found"
        rospy.loginfo("%s" %(text))
        return True,text

    def checkscheduletime(self,command_input):

        split = split_string(command_input)

        if not(self.checktimevalue(split[0],0,6)):
            text = " Time Schedule Day of Week not valid "
            return False,text
        if not(self.checktimevalue(split[1],0,23)):
            text = " Time Schedule Hour not valid "
            return False,text
        if not(self.checktimevalue(split[2],0,59)):
            text = " Time Schedule Minute not valid "
            return False,text
        return True," Time Schedule OK"

    def checktimevalue(self,text,min,max):
        if not(unicode(text,'utf-8').isnumeric()):
            if text == 'A' or text == 'a':
                return True
            else:
                return False
        else:
            if int(text)>=min and int(text)<=max:
                return True
            else:
                return False

    def append(self, schedule_id,wk,date_start,date_end,hour,minute,repeat_hour,enabled,commands):
        """
            Adds a new schedule

            Args:
                schedule_id: string as the schedule id
                time: time to launch the schedule
                commands: string [] as the commands that contains the schedule
        """
        #ret,msg= self.checkscheduletime(time)
        #if ret==False:
        #    return False,msg

        bFound = False
        for s in self.schedules:
            if (s.get_id()==schedule_id):
                s.update_commands(commands)
                s.update_dayofweek(wk)
                s.update_date_start(date_start)
                s.update_date_end(date_end)
                s.update_hour(hour)
                s.update_minute(minute)
                s.update_reph(repeat_hour)
                s.enable(enabled)
                text = "Schedule:"+schedule_id+" updated"
                rospy.loginfo("%s" %(text))
                return True,text
        if not(bFound):
            self.schedules.append(Schedule(schedule_id,wk,date_start,date_end,hour,minute,repeat_hour,enabled,commands))
            text = "Schedule:"+schedule_id+" added"
            rospy.loginfo("%s" %(text))
            return True,text
    def get_schedules_dict(self):
        res_dict={}
        for s in self.schedules:
            s_dict={}
            s_dict['wk']=s.wk
            s_dict['date_start']=s.date_start
            s_dict['date_end']=s.date_end
            s_dict['hour']=s.hour
            s_dict['minute']=s.minute
            s_dict['repeat_hour']=s.repeat_hour
            s_dict['enabled']=s.enabled
            s_dict['commands']=s.commands
            print (s_dict)
            res_dict[s.id]=s_dict
        print (res_dict)
        return res_dict

    def get_schedules_list(self):
        """
            Gets Schedules List
        """
        l = EventSchedulerArray()
        for s in self.schedules:
            C = EventScheduler()
            C.id = s.get_id()
            C.wk =  s.get_dayofweek()
            C.date_start = s.get_date_start()
            C.date_end = s.get_date_end()
            C.hour = s.get_hour()
            C.minute = s.get_minute()
            C.repeat_hour = s.get_reph()
            C.enabled = s.get_enabled()
            C.commands = s.get_commands()
            l.Events.append(C)

        return l

    def get_next_schedule(self):
        """
            Gets next schedule
        """

        current_datetime = datetime.now()
        today = datetime.date(current_datetime)
        H=current_datetime.hour
        M=current_datetime.minute
        D=current_datetime.weekday()

        for s in self.schedules:
            if (s.get_enabled()):
                #print ("id:%s enabled"%s.get_id() )
                if (today>=s.get_date_start_datetime()):
                    #print ("id:%s date_start OK"%s.get_id() )
                    if (today<=s.get_date_end_datetime()):
                        #print ("id:%s date_end OK"%s.get_id() )
                        #print ("id:%s day_of_week OK curren dow:%d ddd:%s"%(s.get_id(),D,s.get_dayofweek()))
                        if (D in s.get_dayofweek() or len(s.get_dayofweek())==0):
                            #print ("id:%s day_of_week OK curren dow:%d ddd:%s"%(s.get_id(),D,s.get_dayofweek()))
                            hh=int(s.get_hour())
                            rep=int(s.get_reph())
                            #print ("id:%s day_of_week OK rep:%d, hour:%d"%(s.get_id(),rep,hh))
                            while hh<24:
                                if H==hh:
                                    #print ("id:%s Hour OK"%s.get_id() )
                                    #print ("M",M,"MM",s.get_minute())
                                    mm = int(s.get_minute())
                                    if M<mm:
                                        rospy.loginfo_throttle(10, "%s::get_next_schedule: the command %s will start in less than %d minutes"%(rospy.get_name(), s.get_id(), mm-M))
                                    elif M==mm:
                                        if (datetime.now()-s.get_last_run())>timedelta(minutes=2):
                                            rospy.loginfo("%s::get_next_schedule: the command %s is READY to Run"%(rospy.get_name(),s.get_id()))
                                            s.update_last_run()
                                            return s
                                        else:
                                            rospy.loginfo_throttle(5,"%s::get_next_schedule: the command %s was run recently... the execution is not allowed."%(rospy.get_name(), s.get_id()))
                                hh=hh+rep
                                if (rep==0):
                                    hh = 24

        return None


class CommandScheduler(RComponent):
    """
    A class used to schedule commands

    ...


    """

    def __init__(self):
        self.feedback_msg = CommandStringFeedback()
        self.feedback_action_msg = RobotSimpleCommandFeedback()
        self.action_client_feedback_msg = None
        # Available schedule of commands
        self.schedules_dict = {}
        # current command sent/in execution
        self.current_command = ''
        # Saves the time of the feedback msg
        self.last_time_received_action_client_feedback = rospy.Time(0)
        # Timeout to control the lack of communication from the action server
        self.communication_timeout = rospy.Duration.from_sec(10)
        self.maxsize_queue = 100


        self.schedule_list = ScheduleList()

        self.current_command = ''
        self.current_schedule_id = ''

        self.schedules_file = ''

        # sending status INIT -> SEND -> RUNNING -> INIT
        self.sending_status = 'INIT'

        # log data
        self.log = CommandLog()

        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)
        #self.schedules_dict = rospy.get_param('~schedules', {})
        self.schedules_file = rospy.get_param('~schedules_file', 'schedules_default.yaml')
        rospy.loginfo('%s:: Using schedules file: %s' % (self._node_name, self.schedules_file))
        self.action_client_namespace = rospy.get_param('~action_client_namespace', 'complex_sequencer/action')
        rospy.loginfo('%s:: Using action client namespace: %s' % (self._node_name, self.action_client_namespace))
        self.add_schedule_name = rospy.get_param('~add_schedule', '~add_schedule')
        rospy.loginfo('%s:: Using add schedule service: %s' % (self._node_name, self.add_schedule_name))
        self.remove_schedule_name = rospy.get_param('~remove_schedule', '~remove_schedule')
        rospy.loginfo('%s:: Using remove schedule service: %s' % (self._node_name, self.remove_schedule_name))
        self.get_schedules_name = rospy.get_param('~get_schedules', '~get_schedules')
        rospy.loginfo('%s:: Using get schedule service: %s' % (self._node_name, self.get_schedules_name))

    def load_schedules(self,file):
        self.schedules_dict = self.load_yaml_file(file)
        #print("------------------------------")
        #print (self.schedules_dict)
        #print("------------------------------")
        try:
            self.schedules_dict = self.load_yaml_file(file)
            #print("------------------------------")
            #print (self.schedules_dict)
            #print("------------------------------")
        except:
            rospy.logerr('%s::Unable to load schedules from: %s' % (self._node_name, self.schedules_file))
            return
        if (self.schedules_dict!=None):
            if (len(self.schedules_dict)>0):
                self.schedule_list.append_from_dict(self.schedules_dict)

    def load_yaml_file(self,file):
        with open(file) as f:
            # use safe_load instead load
            data = yaml.safe_load(f)
            return data

    def save_yaml_file(self,data_dict,file):
        with open(file, 'w+') as f:
            yaml.dump(data_dict, f, allow_unicode=True)


    def ros_setup(self):
        """Creates and inits ROS components"""

        RComponent.ros_setup(self)

        self.command_feedback_pub = rospy.Publisher(
            '~feedback', CommandStringFeedback, queue_size=1)

        self.command_log_publisher = rospy.Publisher(
            'command_log', CommandLog, queue_size=1)

        self.action_client = actionlib.SimpleActionClient(
            self.action_client_namespace, RobotSimpleCommandAction)

        self.add_schedule_service = rospy.Service(self.add_schedule_name,
                                                  AddSchedule,
                                                  self.add_schedule)

        self.get_schedules_service = rospy.Service(self.get_schedules_name,
                                                  GetCommandSchedulerList,
                                                  self.get_schedules)

        self.remove_schedule_service = rospy.Service(self.remove_schedule_name,
                                                  ManageCommandManager,
                                                  self.remove_schedule)

        return 0

    def get_schedules(self, msg):

        ret = GetCommandSchedulerListResponse()

        ret.list = self.schedule_list.get_schedules_list()
        ret.success=True

        return ret

    def add_schedule(self, msg):

        #print (msg)

        res,res_msg = self.schedule_list.append(msg.event.id,msg.event.wk,msg.event.date_start,msg.event.date_end,
                                                msg.event.hour,msg.event.minute,msg.event.repeat_hour,
                                                msg.event.enabled,msg.event.commands)

        #if res:
        #    self.schedules_dict['schedules'][msg.Command.id]=msg.Command.command
        #    self.save_yaml_file(self.schedules_dict,self.schedules_file)

        sch_dict = self.schedule_list.get_schedules_dict()
        self.save_yaml_file(sch_dict,self.schedules_file)
        rospy.loginfo('%s::add_schedule: service adding ID:%s, Command:%s msg:%s',
                                self._node_name, msg.event.id,msg.event.commands,res_msg)


        ret = ReturnMessage()

        ret.success = res
        ret.message = res_msg

        return ret

    def remove_schedule(self, msg):
        res,res_msg = self.schedule_list.remove(msg.Command.id)

        if res:
            try:
                del self.schedules_dict[msg.Command.id]
                self.save_yaml_file(self.schedules_dict,self.schedules_file)
            except:
                rospy.logerr('%s::remove_schedule: Error removing ID:%s from Dict, msg:%s',
                                self._node_name, msg.Command.id,res_msg)

        rospy.loginfo('%s::remove_schedule: removing ID:%s, msg:%s',
                                self._node_name, msg.Command.id,res_msg)


        ret = ManageCommandManagerResponse()

        ret.ret.success = res
        ret.ret.message = res_msg

        return ret

    def init_state(self):

        self.load_schedules(self.schedules_file)
        self.switch_to_state(State.READY_STATE)

    def publish_log(self):
        self.log.node_name = self._node_name
        self.log.process_name = self.current_schedule_id
        self.log.command = self.current_command
        self.command_log_publisher.publish(self.log)

    def ready_state(self):

        #rospy.loginfo('%s::ready_state next none' % (self._node_name))
        if self.sending_status == 'INIT':
            # No current schedule check for new
            self.current_command = ''
            self.current_schedule_id = ''
            s = self.schedule_list.get_next_schedule()
            #rospy.loginfo('%s::ready_state current %s' % (self._node_name,id))
            if s!=None:
                # new schedule
                self.current_schedule_id = s.get_id()
                self.current_command = s.get_commands()
                self.sending_status = 'SEND'


        if self.sending_status == "SEND":
            # SEND THE ACTION
            ret, ret_msg = self.send_cmd(self.current_command)
            if ret == True:
                rospy.loginfo('%s::ready_state: sending new command: %s ret_msg:%s',
                                self._node_name, self.current_command,ret_msg)
                self.log.status = "Start"
                self.log.result = ret_msg
                self.log.error = False
                self.sending_status = 'RUNING'
            else:
                rospy.logerr('%s::ready_state: error sending command: %s ret_msg:%s. Finishing schedule',
                                self._node_name, self.current_command,ret_msg)
                self.log.status = "Start"
                self.log.result = ret_msg
                self.log.error = True
                self.sending_status = 'INIT'

        if self.sending_status == "RUNING":
            self.update_feedback()
            self.publish_feedback()
            rospy.loginfo_throttle(5, '%s::ready_state: waiting for the end of the command: %s' %
                                    (self._node_name, self.current_command))

            action_client_state = self.action_client.get_state()
            communication_timeout = ((rospy.Time.now() - self.last_time_received_action_client_feedback)
                                        >= self.communication_timeout)



            if communication_timeout == True:

                rospy.logerr('%s::ready_state: Timeout getting feedback from the client. Command %s. Finishing schedule',
                                self._node_name, self.current_command)

                self.log.status = "End"
                self.log.result = "Command Timeout"
                self.log.error = True
                self.sending_status = 'INIT'

            elif action_client_state != GoalStatus.ACTIVE and action_client_state != GoalStatus.PENDING:
                action_client_result = self.action_client.get_result()
                if action_client_result == None:  # If nothing is received from get_result
                    if action_client_state == GoalStatus.SUCCEEDED:
                        rospy.loginfo('%s::ready_state: the command (%s) has finished correctly!',
                                        self._node_name, self.current_command)
                        self.log.status = "End"
                        self.log.result = "OK"
                        self.log.error = False
                    else:
                        rospy.logerr('%s::ready_state: the command (%s) has finished wrongly (goal_status = %d)!',
                                        self._node_name, self.current_command, action_client_state)
                        self.log.status = "End"
                        self.log.result = action_client_state
                        self.log.error = False
                        self.sending_status = 'INIT'

                elif action_client_result.result.success == True:
                    rospy.loginfo('%s::ready_state: the command (%s) has finished correctly!',
                                    self._node_name, self.current_command)
                    self.log.status = "End"
                    self.log.result = "OK"
                    self.log.error = False
                    self.sending_status = 'INIT'
                else:
                    rospy.logerr('%s::ready_state: the command (%s) has finished wrongly (code = %d, msg = %s)!',
                                    self._node_name, self.current_command, action_client_result.result.code, action_client_result.result.message)
                    self.log.status = "End"
                    self.log.result = action_client_result.result.message
                    self.log.error = True
                    self.sending_status = 'INIT'

        # Publish log data
        self.publish_log()

    def shutdown(self):
        """Shutdowns device

        Return:
            0 : if it's performed successfully
            -1: if there's any problem or the component is running
        """

        return RComponent.shutdown(self)

    def switch_to_state(self, new_state):
        """Performs the change of state"""

        return RComponent.switch_to_state(self, new_state)


    def publish_feedback(self):
        """Publishes the updated feedback
        """
        self.command_feedback_pub.publish(self.feedback_msg)


    def action_client_feedback_cb(self, feedback):
        """
            Callback for the feedback from the action client
        """
        self.action_client_feedback_msg = feedback
        self.last_time_received_action_client_feedback = rospy.Time.now()
        # self.parse_feedback()

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
            rospy.logerr('%s::send_cmd: %s' % (self._node_name, ret_msg))

        return (command_sent, ret_msg)

    def update_feedback(self):
        """Updates feedback messages

        Updates both feedback_msg and feedback_action_msg with information
        provided by the current_handler
        """

        self.feedback_msg.command = '%s:%s' % (self.current_schedule_id, self.current_command)
        try:
            self.feedback_msg.status = self.action_client_feedback_msg.feedback.status
        except:
            self.feedback_msg.status = "Error"

        self.feedback_action_msg.feedback = self.feedback_msg
