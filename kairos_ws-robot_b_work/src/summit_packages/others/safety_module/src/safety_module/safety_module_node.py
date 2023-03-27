#!/usr/bin/env python

import rospy

import time
import threading
import os
import sys

from robotnik_msgs.srv import set_modbus_register, set_modbus_registerRequest
from robotnik_msgs.srv import set_digital_output, set_digital_outputRequest, set_digital_outputResponse
from robotnik_msgs.srv import SetLaserMode, SetLaserModeResponse, SetNamedDigitalOutput
from robotnik_msgs.msg import inputs_outputs, State
from robotnik_msgs.msg import SafetyModuleStatus, LaserStatus, LaserMode, BoolArray, named_inputs_outputs, named_input_output
from std_srvs.srv import SetBool, SetBoolResponse
from std_msgs.msg import Bool, Int32
from nav_msgs.msg import Odometry
import math

DEFAULT_FREQ = 5.0
MAX_FREQ = 5.0
RECEIVED_IO_TIMEOUT = 10.0
# defines the bits in the register corresponding to the watchdog signal
WATCHDOG_UPPER_BIT = 16
WATCHDOG_LOWER_BIT = 15
WATCHDOG_WRITE_MODE_FULL = 'full_register'


class RobotnikFlexisoft:
    def __init__(self, args):
        self.node_name = rospy.get_name()
        # .replace('/','')
        self.desired_freq = args['desired_freq']

        # Checks value of freq
        if self.desired_freq <= 0.0 or self.desired_freq > MAX_FREQ:
            rospy.loginfo('%s::init: Desired freq (%f) is not possible. Setting desired_freq to %f' %
                          (self.node_name, self.desired_freq, DEFAULT_FREQ))
            self.desired_freq = DEFAULT_FREQ

        self.real_freq = 0.0

        # Saves the state of the component
        self.state = State.INIT_STATE
        # Saves the previous state
        self.previous_state = State.INIT_STATE
        # flag to control the initialization of the component
        self.initialized = False
        # flag to control the initialization of ROS stuff
        self.ros_initialized = False
        # flag to control that the control loop is running
        self.running = False
        # Variable used to control the loop frequency
        self.time_sleep = 1.0 / self.desired_freq
        # State msg to publish
        self.msg_state = State()
        self.publish_state_timer = 1

        self.lasers = args['lasers']
        self.set_modbus_registers_service_name = args['set_modbus_registers_service_name']
        self.set_digital_outputs_service_name = args['set_digital_outputs_service_name']
        self.laser_mode_output_addr = args['address_registers/laser_mode_output']
        self.current_speed_addr = args['address_registers/current_speed']
        # If set, laser changes mode to standby (to use less power when stop). It does not send any data in this mode
        self.standby_output = args['outputs/standby']
        self.watchdog_signals_output = args['outputs/watchdog_signals']

        self.watchdog_signals_frequency = args['watchdog_signals_frequency']
        if self.watchdog_signals_frequency <= 0.0:
            self.watchdog_signals_frequency = 1.0
        self.watchdog_signals_period = 1.0/self.watchdog_signals_frequency
        self.watchdog_write_mode = args['watchdog_write_mode']

        self.emergency_stop_input = args['inputs/emergency_stop']          # If set, e-stop is not pressed
        # If set, robot is stopped due to the safety system (either e-stop or obstacle inside protection area)
        self.safety_stop_input = args['inputs/safety_stop']
        # If set, safety mode is in emergency (elevator lowers, brakes are free, it can be pushed, key to the right)
        self.laser_mute_input = args['inputs/laser_mute']
        # If set, auto mode is enabled (key is to the left)
        self.mode_auto_input = args['inputs/mode_auto']
        # If set, manual mode is enabled (key is in the middle)
        self.mode_manual_input = args['inputs/mode_manual']
        # If set, maintenance mode is enabled (key is in the right)
        self.mode_maintenance_input = args['inputs/mode_maintenance']
        # If set, safety is overrided in safety overridable mode (key in the middle)
        self.laser_on_standby_input = args['inputs/standby']

        # structure to save inputs
        self.inputs_ = {}
        self.inputs_['wheels_power_enabled'] = {'number': args['inputs/wheels_power_enabled'], 'value': False}
        self.inputs_['laser_ok'] = {'number': args['inputs/laser_ok'], 'value': False}
        self.inputs_['edm_ok'] = {'number': args['inputs/edm_ok'], 'value': False}
        self.inputs_['emergency_stop_fault'] = {'number': args['inputs/emergency_stop_fault'], 'value': False}
        self.inputs_['motion_enabled'] = {'number': args['inputs/motion_enabled'], 'value': False}
        self.inputs_['selector_ok'] = {'number': args['inputs/selector_ok'], 'value': False}
        self.inputs_['watchdog_ok'] = {'number': args['inputs/watchdog_ok'], 'value': False}
        self.inputs_['emergency_stop_sw'] = {'number': args['inputs/emergency_stop_sw'], 'value': False}

        # structure to save laser modes
        self.laser_modes_available_ = args['laser_modes']

        if len(self.laser_modes_available_) == 0:
            rospy.logwarn('%s::init: no laser modes defined!' % (self.node_name))
        else:
            for mode in self.laser_modes_available_:
                if not self.laser_modes_available_[mode].has_key('input'):
                    rospy.logerr('%s::init: laser_modes format has to have the key input. Invalid format: %s',
                                 self.node_name, str(self.laser_modes_available_))
                    exit(-1)
                if not self.laser_modes_available_[mode].has_key('output'):
                    rospy.logerr('%s::init: laser_modes format has to have the key output. Invalid format: %s',
                                 self.node_name, str(self.laser_modes_available_))
                    exit(-1)
                # add field to save the input value
                self.laser_modes_available_[mode]['input_value'] = False

        # custom inputs
        for ci in args['custom_inputs']:
            if ci not in self.inputs_:
                self.inputs_[ci] = {'number': args['custom_inputs'][ci], 'value': False}
            else:
                rospy.logerr('%s::init: custom input %s already defined as standard' % (self.node_name, ci))

        if len(args['custom_inputs']) == 0:
            rospy.logwarn('%s::init: no custom inputs defined' % (self.node_name))

            # new structure to save the outputs
        self.outputs_ = {}
        self.outputs_['emergency_stop_sw'] = {
            'number': args['outputs/emergency_stop_sw'], 'value': False, 'desired_value': False}
        # custom outputs
        for co in args['custom_outputs']:
            if co not in self.outputs_:
                self.outputs_[co] = {'number': args['custom_outputs'][co], 'value': False, 'desired_value': False}
            else:
                rospy.logerr('%s::init: custom output %s already defined as standard' % (self.node_name, co))

        if len(args['custom_outputs']) == 0:
            rospy.logwarn('%s::init: no custom outputs defined' % (self.node_name))

        self.laser_mode = "unknown"
        self.inputs_outputs_msg = inputs_outputs()
        self.emergency_stop_msg = Bool()
        self.safety_stop_msg = Bool()
        self.last_enable_charge_msg_sent = False
        self.mode_auto = False
        self.laser_mute = False
        self.safety_mode = "unknown"
        self.operation_mode = "unknown"
        self.sm_status_msg = SafetyModuleStatus()
        self.emergency_stop = False
        self.safety_stop = False
        self.laser_on_standby = False
        self.named_io_msg = named_inputs_outputs()

        if (self.watchdog_write_mode != WATCHDOG_WRITE_MODE_FULL and len(self.watchdog_signals_output) == 2) or self.watchdog_write_mode == WATCHDOG_WRITE_MODE_FULL:
            self.watchdog_signals_enabled = True
            self.watchdog_signals_registers = [True, False]
            # self.watchdog_frequency
            rospy.loginfo('%s::init: watchdog signals is enabled! outputs %s' %
                          (self.node_name, str(self.watchdog_signals_output)))
        else:
            self.watchdog_signals_enabled = False
            rospy.logwarn('%s::init: watchdog signals is disabled! outputs %s' %
                          (self.node_name, str(self.watchdog_signals_output)))

        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)

        # current absolute speed in cm/s
        self.current_speed = 0.0
        # flag true if receiving speed
        self.receiving_speed = False
        # Saves time of last msg received
        self.odom_last_stamp = rospy.Time(0)
        #  Saves time of last watchdog
        self.watchdog_last_stamp = rospy.Time(0)

        # saves the desired laser mode: by default no actions performed. Only on demand
        self.set_desired_laser_mode = None
        # saves the desired value of standby mode: by default no actions performed. Only on demand
        self.set_desired_standby_mode = None

    def setup(self):
        '''
            Initializes de hand
            @return: True if OK, False otherwise
        '''
        self.initialized = True
        return 0

    def rosSetup(self):
        '''
            Creates and inits ROS components
        '''
        if self.ros_initialized:
            rospy.logwarn("%s::rosSetup: already initialized" % self.node_name)
            return 0

        self.io_last_stamp = 0

        self.now = rospy.get_time()

        # Publishers
        self.state_publisher = rospy.Publisher('~state', State, queue_size=10)
        self.emergency_stop_pub = rospy.Publisher('~emergency_stop', Bool, queue_size=1)
        self.safety_stop_pub = rospy.Publisher('~safety_stop', Bool, queue_size=1)
        self.safety_module_state_pub = rospy.Publisher('~status', SafetyModuleStatus, queue_size=1)
        if self.watchdog_signals_enabled:
            self.watchdog_signales_pub = rospy.Publisher('~watchdog_signals', BoolArray, queue_size=1)
        self.named_io_pub = rospy.Publisher('~named_io', named_inputs_outputs, queue_size=1)

        # Subscribers
        self.modbus_io_sub = rospy.Subscriber("robotnik_modbus_io/input_output",
                                              inputs_outputs, self.modbusIOCb, queue_size=1)

        self.odometry_sub = rospy.Subscriber("robotnik_base_control/odom", Odometry, self.odometryCb, queue_size=1)
        self.speed_feedback_pub = rospy.Publisher('~speed_feedback', Int32, queue_size=10)

        # Service Servers
        self.set_laser_mode_server = rospy.Service('~set_laser_mode',  SetLaserMode, self.setLaserModeCb)
        self.set_standby_server = rospy.Service('~set_to_standby', SetBool, self.setStandByCb)
        self.set_named_outputs_server = rospy.Service(
            '~set_named_output', SetNamedDigitalOutput, self.setNamedDigitalOutputCb)

        # Service Clients
        self.write_register_client = rospy.ServiceProxy(self.set_modbus_registers_service_name, set_modbus_register)
        self.write_digital_output_client = rospy.ServiceProxy(self.set_digital_outputs_service_name, set_digital_output)

        self.ros_initialized = True
        self.publishROSstate()

        return 0

    def shutdown(self):
        if self.running or not self.initialized:
            return -1
        rospy.loginfo('%s::shutdown' % self.node_name)

        # Cancels current timers
        self.t_publish_state.cancel()

        '''if self.watchdog_signals_enabled:
			self.t_watchdog_signals_timer.cancel()'''

        self.state_publisher.unregister()

        self.initialized = False

        return 0

    def rosShutdown(self):
        '''
            Shutdows all ROS components
            @return: 0 if it's performed successfully, -1 if there's any problem or the component is running
        '''
        if self.running or not self.ros_initialized:
            return -1

        # Performs ROS topics & services shutdown
        self.state_publisher.unregister()

        self.ros_initialized = False

        return 0

    def stop(self):
        '''
            Creates and inits ROS components
        '''
        self.running = False

        return 0

    def start(self):
        '''
            Runs ROS configuration and the main control loop
            @return: 0 if OK
        '''
        self.rosSetup()

        if self.running:
            return 0

        self.running = True

        self.controlLoop()

        return 0

    def controlLoop(self):
        '''
            Main loop of the component
            Manages actions by state
        '''
        while self.running and not rospy.is_shutdown():
            t1 = rospy.Time.now()

            if self.state == State.INIT_STATE:
                self.initState()

            elif self.state == State.STANDBY_STATE:
                self.standbyState()

            elif self.state == State.READY_STATE:
                self.readyState()

            elif self.state == State.EMERGENCY_STATE:
                self.emergencyState()

            elif self.state == State.FAILURE_STATE:
                self.failureState()

            elif self.state == State.SHUTDOWN_STATE:
                self.shutdownState()

            self.allState()

            t2 = rospy.Time.now()
            tdiff = (t2 - t1).to_sec()

            t_sleep = self.time_sleep - tdiff

            if t_sleep > 0.0:
                try:
                    rospy.sleep(t_sleep)
                except rospy.exceptions.ROSInterruptException:
                    rospy.loginfo('%s::controlLoop: ROS interrupt exception' % self.node_name)
                    self.running = False

            t3 = rospy.Time.now()
            self.real_freq = 1.0/(t3 - t1).to_sec()

        self.running = False
        # Performs component shutdown
        self.shutdownState()
        # Performs ROS shutdown
        self.rosShutdown()
        rospy.loginfo('%s::controlLoop: exit control loop' % self.node_name)

        return 0

    def rosPublish(self):
        '''
            Publish topics at standard frequency
        '''
        # Publish emergency stop
        self.emergency_stop_pub.publish(self.emergency_stop_msg)
        # Publish safety stop
        self.safety_stop_pub.publish(self.safety_stop_msg)
        # Publish the safety module status
        self.safety_module_state_pub.publish(self.sm_status_msg)
        # Named IO
        self.named_io_pub.publish(self.named_io_msg)

        return 0

    def initState(self):
        '''
            Actions performed in init state
        '''
        if not self.initialized:
            self.setup()

        else:
            self.switchToState(State.STANDBY_STATE)

        return

    def standbyState(self):
        '''
            Actions performed in standby state
        '''
        self.switchToState(State.READY_STATE)

        return

    def readyState(self):
        '''
            Actions performed in ready state
        '''
        self.now = rospy.get_time()

        # Check if i/o is been received
        if (self.now - self.io_last_stamp) < RECEIVED_IO_TIMEOUT:

            self.emergency_stop = not self.inputs_outputs_msg.digital_inputs[self.emergency_stop_input - 1]
            self.safety_stop = not self.inputs_outputs_msg.digital_inputs[self.safety_stop_input - 1]

            # Get the robot mode
            self.mode_auto = self.inputs_outputs_msg.digital_inputs[self.mode_auto_input - 1]
            self.mode_manual = self.inputs_outputs_msg.digital_inputs[self.mode_manual_input - 1]
            self.mode_maintenance = self.inputs_outputs_msg.digital_inputs[self.mode_maintenance_input - 1]
            self.laser_mute = self.inputs_outputs_msg.digital_inputs[self.laser_mute_input - 1]

            # Check if robot is stopped due to safety system
            self.safety_stop = not self.inputs_outputs_msg.digital_inputs[self.safety_stop_input - 1]

            # Check if lasers are on standby mode
            self.laser_on_standby = self.inputs_outputs_msg.digital_inputs[self.laser_on_standby_input - 1]

            if self.laser_on_standby:
                rospy.logwarn_throttle(5, "%s::readyState: lasers on standby" % self.node_name)

            # Get the laser mode input values
            for mode in self.laser_modes_available_:
                self.laser_modes_available_[
                    mode]['input_value'] = self.inputs_outputs_msg.digital_inputs[self.laser_modes_available_[mode]['input'] - 1]
            # Process the rest of signals
            for signal in self.inputs_:
                self.inputs_[signal]['value'] = self.inputs_outputs_msg.digital_inputs[self.inputs_[signal]['number'] - 1]

            self.laser_status = []
            for laser in self.lasers:
                laser_msg = LaserStatus()
                laser_msg.name = laser['name']
                laser_msg.detecting_obstacles = not self.inputs_outputs_msg.digital_inputs[laser['detecting_obstacles'] - 1]
                laser_msg.contaminated = self.inputs_outputs_msg.digital_inputs[laser['contamination_led'] - 1]
                laser_msg.free_warning = self.inputs_outputs_msg.digital_inputs[laser['free_warning'] - 1]
                self.laser_status.append(laser_msg)

            # Get the emergency stop info
            self.emergency_stop_msg.data = self.emergency_stop
            self.safety_stop_msg.data = self.safety_stop

            self.updateRobotMode()
            self.updateLaserMode()
            self.updateSafetyModuleStatus()
            self.updateNamedIO()

            # WRITING OUTPUTS
            if self.set_desired_standby_mode != None and self.set_desired_standby_mode != self.laser_on_standby:
                if not self.setDigitalOutput(self.standby_output, self.set_desired_standby_mode):
                    rospy.logerr_throttle(2, '%s::readyState: error trying to set the standby signal %d to %s' % (
                        self.node_name, self.standby_output, self.set_desired_standby_mode))

            if self.set_desired_laser_mode != None and self.set_desired_laser_mode != self.laser_mode:
                if not self.switchLaserMode(self.set_desired_laser_mode):
                    rospy.logerr_throttle(2, '%s::readyState: error trying to switch the LaserMode from %s to %s' % (
                        self.node_name, self.laser_mode, self.set_desired_laser_mode))
            elif self.set_desired_laser_mode == self.laser_mode:
                self.set_desired_laser_mode = None		# avoid continuous control

            # WATCHDOG & SPEED FEEDBACK
            if self.watchdog_signals_enabled:
                ret = self.watchDogSignalsLoop()

            # OUTPUT signals
            for o_name in self.outputs_:
                if self.outputs_[o_name]['value'] != self.outputs_[o_name]['desired_value']:
                    if not self.setDigitalOutput(self.outputs_[o_name]['number'], self.outputs_[o_name]['desired_value']):
                        rospy.logerr_throttle(2, '%s::readyState: error trying to set the output %s(%d) to %s' % (
                            self.node_name, o_name, self.outputs_[o_name]['number'], self.outputs_[o_name]['desired_value']))
                    else:
                        self.outputs_[o_name]['value'] = self.outputs_[o_name]['desired_value']
                        rospy.loginfo('%s::readyState: set the output %s(%d) to %s' % (self.node_name,
                                                                                       o_name, self.outputs_[o_name]['number'], self.outputs_[o_name]['desired_value']))

        else:
            rospy.logwarn('%s::readyState: Not receiving i/o messages' % self.node_name)
            self.switchToState(State.EMERGENCY_STATE)

        return

    def shutdownState(self):
        '''
            Actions performed in shutdown state
        '''
        if self.shutdown() == 0:
            self.switchToState(State.INIT_STATE)

        return

    def emergencyState(self):
        '''
            Actions performed in emergency state
        '''
        if (self.now - self.io_last_stamp) < RECEIVED_IO_TIMEOUT:
            self.switchToState(State.READY_STATE)
        else:
            rospy.logwarn_throttle(5, '%s::emergencyState: Not receiving i/o messages' % self.node_name)

        return

    def failureState(self):
        '''
            Actions performed in failure state
        '''
        return

    def switchToState(self, new_state):
        '''
            Performs the change of state
        '''
        if self.state != new_state:
            self.previous_state = self.state
            self.state = new_state
            rospy.loginfo('%s::switchToState: %s' % (self.node_name, self.stateToString(self.state)))

        return

    def allState(self):
        '''
            Actions performed in all states
        '''
        self.rosPublish()

        if (rospy.Time.now() - self.odom_last_stamp).to_sec() > 1.0:
            self.receiving_speed = False
        else:
            self.receiving_speed = True

        return

    def stateToString(self, state):
        '''
            @param state: state to set
            @type state: State
            @returns the equivalent string of the state
        '''
        if state == State.INIT_STATE:
            return 'INIT_STATE'

        elif state == State.STANDBY_STATE:
            return 'STANDBY_STATE'

        elif state == State.READY_STATE:
            return 'READY_STATE'

        elif state == State.EMERGENCY_STATE:
            return 'EMERGENCY_STATE'

        elif state == State.FAILURE_STATE:
            return 'FAILURE_STATE'

        elif state == State.SHUTDOWN_STATE:
            return 'SHUTDOWN_STATE'
        else:
            return 'UNKNOWN_STATE'

    def publishROSstate(self):
        '''
            Publish the State of the component at the desired frequency
        '''
        self.msg_state.state = self.state
        self.msg_state.state_description = self.stateToString(self.state)
        self.msg_state.desired_freq = self.desired_freq
        self.msg_state.real_freq = self.real_freq
        self.state_publisher.publish(self.msg_state)

        self.t_publish_state = threading.Timer(self.publish_state_timer, self.publishROSstate)
        self.t_publish_state.start()

    def setLaserModeCb(self, req):
        '''
            ROS service server to change the laser mode
        '''
        response = SetLaserModeResponse()

        if self.state != State.READY_STATE:
            rospy.logerr('%s::setLaserModeCb: component not in READY STATE' % self.node_name)
            response.ret = False
        else:
            if not self.checkValidLaserMode(req.mode):
                rospy.logerr('%s::setLaserModeCb: invalid mode: %s', self.node_name, req.mode)
                response.ret = False
            else:
                self.set_desired_laser_mode = req.mode
                rospy.loginfo('%s::setLaserModeCb: setting laser mode: %s', self.node_name, req.mode)
                response.ret = True

        return response

    def setStandByCb(self, req):
        '''
            ROS service server to enable/disable safety override
        '''
        response = SetBoolResponse()

        self.set_desired_standby_mode = req.data

        response.success = True
        response.message = 'received petition to set the standby mode to %d' % req.data

	rospy.logerr(response.message)

        return response

    def setNamedDigitalOutputCb(self, req):
        '''
            ROS service server to set any named output
        '''
        if req.name in self.outputs_:
            self.outputs_[req.name]['desired_value'] = req.value
            return True, "OK"
        else:
            return False, 'Output [%s] not available' % req.name

    def switchLaserMode(self, new_mode):
        '''
            Change the laser mode by sending the new mode through a service
        '''
        mode = self.getLaserModeCode(new_mode)

        if mode == None:
            rospy.logerr('%s::switchLaserMode: invalid new mode: %s', self.node_name, new_mode)
            return False

        res = self.setModbusRegister(self.laser_mode_output_addr, mode)

        if res == True:
            rospy.loginfo_throttle(2, '%s::switchLaserMode: switching to mode: %s ' % (self.node_name, new_mode))
        else:
            rospy.logerr('%s::switchLaserMode: could not switch to mode: %s ' % (self.node_name, new_mode))

        return res

    def setModbusRegister(self, addr, value):
        '''
            Service client
        '''
        # rospy.wait_for_service('robotnik_modbus_io/set_modbus_register')
        try:
            request = set_modbus_registerRequest()
            request.address = addr
            request.value = value
            response = self.write_register_client(request)
            if response.ret == False:
                rospy.logerr('%s::setModbusRegister: could not write register %d with value %d ' %
                             (self.node_name, addr, value))
            return response.ret
        except rospy.ServiceException as e:
            rospy.logerr('%s::setModbusRegister: exception: %s', self.node_name, str(e))
            return False
        except rospy.exceptions.ROSInterruptException as e:
            rospy.logerr('%s::setModbusRegister: exception: %s', self.node_name, str(e))
            return False

    def setDigitalOutput(self, output, value):
        '''
           Function that writes the info received into the write_digital_output service
        '''
        # rospy.wait_for_service('robotnik_modbus_io/write_digital_output')
        try:
            request = set_digital_outputRequest()
            request.output = output
            request.value = value
            response = self.write_digital_output_client(request)
            return response.ret
        except rospy.ServiceException, e:
            rospy.logerr('%s::setDigitalOutput: service call failed: %s', self.node_name, str(e))
            return False
        except Exception, e:
            rospy.logerr('%s::setDigitalOutput: service call failed: %s', self.node_name, str(e))
            return False

    def modbusIOCb(self, msg):
        '''
            Callback for robotnik modbus io
            @param msg: received message
            @type msg: robotnik_msgs/inputs_outpus
        '''
        self.inputs_outputs_msg = msg
        self.io_last_stamp = rospy.get_time()

    def odometryCb(self, msg):
        '''
            Callback for odometry
            @param msg: received message
            @type msg: navigation_msgs/Odometry
        '''

        self.odom_last_stamp = rospy.Time.now()
        x = msg.twist.twist.linear.x
        y = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(x*x+y*y)*100
        #rospy.logwarn_throttle(2, '%s::odometryCb: speed = %.3lf cm/s'%(self.node_name, self.current_speed))

    def updateRobotMode(self):
        if self.mode_auto == True:
			self.operation_mode = SafetyModuleStatus.OM_AUTO
			self.safety_mode = SafetyModuleStatus.SAFE
        elif self.mode_manual == True:
			self.operation_mode = SafetyModuleStatus.OM_MANUAL
			if self.laser_mute == True:
				self.safety_mode = SafetyModuleStatus.LASER_MUTE
			else:
				self.safety_mode = SafetyModuleStatus.SAFE
        elif self.mode_maintenance == True:
			self.operation_mode = SafetyModuleStatus.OM_MAINTENANCE
			# TO check...
			self.safety_mode = SafetyModuleStatus.LASER_MUTE
        else:
			self.operation_mode = 'unknown'
			self.safety_mode = 'unknown'
            
        return

    def updateLaserMode(self):
        '''
            Updates current laser mode based on the current inputs received
        '''
        for mode in self.laser_modes_available_:
            if self.laser_modes_available_[mode]['input_value'] == True:
                if self.laser_mode != mode:
                    rospy.loginfo('%s::updateLaserMode: updating laser mode from %s to %s',
                                  self.node_name, self.laser_mode, mode)
                    self.laser_mode = mode
        return

    def updateSafetyModuleStatus(self):
        # Robot mode
        self.sm_status_msg.safety_mode = self.safety_mode
        self.sm_status_msg.operation_mode = self.operation_mode

        # lasers
        self.sm_status_msg.lasers_mode.name = self.laser_mode
        self.sm_status_msg.lasers_status = self.laser_status

        # emergency stop
        self.sm_status_msg.emergency_stop = self.emergency_stop

        # lasers on standby
        self.sm_status_msg.lasers_on_standby = self.laser_on_standby

        # Check the safety override status
        self.sm_status_msg.safety_stop = self.safety_stop

        # speed (m/s)
        self.sm_status_msg.current_speed = self.current_speed/100.0

        return

    def updateNamedIO(self):

        msg = named_inputs_outputs()
        for signal in self.inputs_:
            m = named_input_output()
            m.name = signal
            m.value = self.inputs_[signal]['value']
            msg.digital_inputs.append(m)

        for signal in self.outputs_:
            m = named_input_output()
            m.name = signal
            m.value = self.outputs_[signal]['value']
            msg.digital_outputs.append(m)
        self.named_io_msg = msg

    def watchDogSignalsLoop(self):
        '''
            Control loop for the signaling
            @Returns 0 if OK
                     -1 if the module does not have feedback from current speed
                     -2 if ERROR writing in modbus
        '''
        ##rospy.loginfo('%s::watchDogSignalsLoop: freq = %lf, setting out %d to %d and %d to %d', self.node_name, self.watchdog_signals_frequency, self.watchdog_signals_output[0], self.watchdog_signals_registers[0],self.watchdog_signals_output[1], self.watchdog_signals_registers[1])
        t_now = rospy.Time.now()

        if (t_now - self.watchdog_last_stamp).to_sec() >= self.watchdog_signals_period:

            if self.receiving_speed:
                register = int(self.current_speed)
                # writes signal in just one call
                if self.watchdog_write_mode == WATCHDOG_WRITE_MODE_FULL:

                    if self.watchdog_signals_registers[1] == True:
                        register = self.toggleKthBit(register, WATCHDOG_UPPER_BIT)
                    else:
                        register = self.toggleKthBit(register, WATCHDOG_LOWER_BIT)

                    #rospy.logwarn('%s::watchDogSignalsLoop: register = %d, register_toggled = %s', self.node_name, int(self.current_speed), register)
                    if self.setModbusRegister(self.current_speed_addr, int(register)) == False:
                        rospy.logerr('%s::watchDogSignalsLoop: error setting modbus register %d',
                                     self.node_name, self.current_speed_addr)
                        return -2

                # writes signals in three calls (compatible with older versions)
                else:
                    if not self.setDigitalOutput(self.watchdog_signals_output[0], self.watchdog_signals_registers[0]) or not self.setDigitalOutput(self.watchdog_signals_output[1], self.watchdog_signals_registers[1]):
                        rospy.logerr('%s::watchDogSignalsLoop: error setting the outputs', self.node_name)
                        return -2
                    if self.setModbusRegister(self.current_speed_addr, int(register)) == False:
                        rospy.logerr('%s::watchDogSignalsLoop: error setting modbus register %d',
                                     self.node_name, self.current_speed_addr)
                        return -2

                watchdog_msg = BoolArray()
                watchdog_msg.data = self.watchdog_signals_registers

                self.watchdog_signales_pub.publish(watchdog_msg)
                self.speed_feedback_pub.publish(int(self.current_speed))

                self.watchdog_signals_registers[1] = not self.watchdog_signals_registers[1]
                self.watchdog_signals_registers[0] = not self.watchdog_signals_registers[0]

            else:
                rospy.logwarn_throttle(5, '%s::watchDogSignalsLoop: odometry is not being received' % (self.node_name))

            self.watchdog_last_stamp = t_now
            return 0

    def checkValidLaserMode(self, mode):
        '''
            Checks if a laser mode is valid
            @Return True if it's correct
            @Return False otherwise
        '''
        return (mode in self.laser_modes_available_)

    def getLaserModeCode(self, mode):
        '''
            Returns the register value for the mode
        '''
        if self.checkValidLaserMode(mode) != True:
            return None

        return self.laser_modes_available_[mode]['output']

    def toggleKthBit(self, n, k):
        '''
            toggles a bit of the passed number
        '''
        return (n ^ (1 << (k-1)))


def main():
    rospy.init_node("safety_module")
    _name = rospy.get_name()

    arg_defaults = {
        'desired_freq': DEFAULT_FREQ,
        'address_registers/laser_mode_output': 2001,
        'address_registers/current_speed': 2002,
        'inputs/emergency_stop': 228,
        'inputs/safety_stop': 225,
        'inputs/standby': 238,
        'inputs/mode_auto': 230,
        'inputs/mode_manual': 231,
        'inputs/mode_maintenance': 232,
        'inputs/laser_mute': 231,
        'inputs/wheels_power_enabled': 225,
        'inputs/laser_ok': 226,
        'inputs/edm_ok': 227,
        'inputs/emergency_stop_fault': 229,
        'inputs/motion_enabled': 235,
        'inputs/selector_ok': 236,
        'inputs/watchdog_ok': 248,
        'inputs/emergency_stop_sw': 246,
        'laser_modes': {},
        'outputs/standby': 13,
        'outputs/watchdog_signals': [],
        'outputs/emergency_stop_sw': 1,
        'watchdog_signals_frequency': 2.0,
        'watchdog_write_mode': '',
        'set_modbus_registers_service_name':  'robotnik_modbus_io/set_modbus_registers',
        'set_digital_outputs_service_name': 'robotnik_modbus_io/write_digital_output',
    }

    args = {}

    for name in arg_defaults:
        try:
            if rospy.has_param('~%s' % (name)):
                # Adding the name of the node, because the param has the namespace of the node
                args[name] = rospy.get_param('~%s' % (name))
            else:
                args[name] = arg_defaults[name]
                rospy.logwarn(_name + ' default value for param ' + name + ': ' + str(arg_defaults[name]))
            #print name
        except rospy.ROSException, e:
            rospy.logerr('%s:ROSException. Error getting params: %s' % (_name, e))
            sys.exit(0)
        except KeyError, e:
            rospy.logerr('%s:KeyError. Error getting params: %s' % (_name, e))
            sys.exit(0)

    try:
        laser_list = rospy.get_param('~inputs/lasers')
        lasers = []
    except KeyError, e:
        rospy.logerr('%s: lasers param needs to be defined. %s' % (_name, e))
        sys.exit(0)

    for laser_item in laser_list:
        laser = {'name': laser_item}
        laser.update(laser_list[laser_item])
        lasers.append(laser)

    args['lasers'] = lasers

    try:
        args['custom_inputs'] = rospy.get_param('~custom_inputs')
    except KeyError, e:
        args['custom_inputs'] = {}

    try:
        args['custom_outputs'] = rospy.get_param('~custom_outputs')
    except KeyError, e:
        args['custom_outputs'] = {}

    rc_node = RobotnikFlexisoft(args)

    rospy.loginfo('%s: starting' % (_name))

    rc_node.start()


if __name__ == "__main__":
    main()
