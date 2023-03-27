#!/usr/bin/env python
from sensor_msgs.msg import JointState

from rcomponent.rcomponent import *

from fake_joint_position_publisher.srv import SetJoint, SetJointResponse, SetJoints, SetJointsResponse

class JointPublisher(RComponent):

    def __init__(self):
        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)
        
        self.joints = rospy.get_param('~joints', [])
        self.joints_config = {}
        for joint in self.joints:
            
            if self.check_joint_parameters(joint) == True:
                self.joints_config[joint] = rospy.get_param('~' + joint, {}) 
            else:
                exit(-1)
    
    def ros_setup(self):
        """Creates and inits ROS components"""

        RComponent.ros_setup(self)
        self.joint_states = JointState()
        self.desired_joint_states = JointState()
        self.fill_joint_state_list()

        self.current_time = rospy.get_time()
        self.last_time = rospy.get_time()

        self.joint_state_publisher = rospy.Publisher('~joint_states', JointState, queue_size=1)
        self.service_srv = rospy.Service('~set_joint', SetJoint, self.set_joint_cb)
        self.service_srv = rospy.Service('~set_joints', SetJoints, self.set_joints_cb)

    def ready_state(self):
        self.current_time = rospy.get_time()
        elapsed_time = self.current_time - self.last_time

        update_positions = self.joint_states.position != self.desired_joint_states.position

        if update_positions == True:
            self.update_joints_positions(elapsed_time)
        else:
            self.joint_states.velocity = [0] * len(self.joint_states.velocity)

        self.joint_states.header.stamp = rospy.Time.now()
        self.joint_state_publisher.publish(self.joint_states)

        self.last_time = self.current_time

    def check_joint_parameters(self, joint):
        """Check that the necessary joint parameters have been set

        Args:
            joint : joint name. Equals to the name of the parameter namespace

        Return:
            Return True if all the parameters are set
            Return False if any parameter is not set
        """
        if rospy.has_param('~' + joint) == False:
            self.error_parameter_not_exists(joint)
            return False
        
        max_value = '~' + joint + '/max_value'
        if rospy.has_param(max_value) == False:
            self.error_parameter_not_exists(max_value)
            return False

        min_value = '~' + joint + '/min_value'
        if rospy.has_param(min_value) == False:
            self.error_parameter_not_exists(min_value)
            return False
        
        velocity = '~' + joint + '/velocity'
        if rospy.has_param(velocity) == False:
            self.error_parameter_not_exists(velocity)
            return False

        
        return True
    
    def error_parameter_not_exists(self, parameter):
        """Prints prameter not set error"""
        rospy.logerr('%s::check_joint_parameters: %s parameter does not exists'%(self._node_name, parameter))

    def fill_joint_state_list(self):
        """Fills the joint_state message with the initial value"""
        for joint in self.joints_config:
            self.joint_states.name.append(joint)
            self.joint_states.position.append(self.joints_config[joint]['min_value'])
            self.joint_states.velocity.append(0.0)
            self.joint_states.effort.append(0.0)

            self.desired_joint_states.name.append(joint)
            self.desired_joint_states.position.append(self.joints_config[joint]['min_value'])
            self.desired_joint_states.velocity.append(0.0)
            self.desired_joint_states.effort.append(0.0)

    def set_joint_cb(self, msg):
        """Service server callback"""
        args_are_correct = self.check_set_joint_args(msg)
        message = ''
        if args_are_correct[0] == True:
            self.write_desired_joint_position(msg.joint_name.strip(), msg.position)
            message = msg.joint_name + ' sent to ' + str(msg.position) + ' position'
            rospy.logwarn('%s::set_joint_cb: %s'%(self._node_name, message))
        else:
            message = args_are_correct[1]
        
        return self.set_joint_response(args_are_correct[0], message)

    def set_joints_cb(self, msg):
        """Service server callback"""
        args_are_correct = self.check_set_joints_args(msg)
        message = ''
        if args_are_correct[0] == True:
            for index in range(0, len(msg.joints_names)):
                name = msg.joints_names[index].strip()
                position = msg.positions[index]
                self.write_desired_joint_position(name, position)
                message += name + ' sent to ' + str(position) + ' position. '
                rospy.logwarn('%s::set_joints_cb: %s'%(self._node_name, message))
        else:
            message = args_are_correct[1]
        
        return self.set_joints_response(args_are_correct[0], message)

    def check_set_joint_args(self, msg):
        """ Checks that the set_joint request arguments are correct
        
        Args:
            msg: SetJointRequest 
                Containts the user input
        Return: (bool, str)
            A tuple that indicates if the arguments are correctly set.
            (True, msg)  : All is correctly set
            (False, msg) : Something is wrong. The error is spread throgh the msg 
        """
        name = msg.joint_name.strip()

        name_is_correct = self.check_set_joint_name(name)
        if name_is_correct[0] == False:
            return name_is_correct

        pos = msg.position
        min_pos = self.joints_config[name]['min_value']
        max_pos = self.joints_config[name]['max_value']

        position_is_correct = self.check_set_joint_position(pos, min_pos, max_pos)
        if  position_is_correct[0] == False:
            return position_is_correct

        return (True, 'Arguments are correct')
        
    def check_set_joints_args(self, msg):
        """ Checks that the set_joints request arguments are correct
        
        Args:
            msg: SetJointsRequest 
                Containts the user input
        Return: (bool, str)
            A tuple that indicates if the arguments are correctly set.
            (True, msg)  : All is correctly set
            (False, msg) : Something is wrong. The error is spread throgh the msg 
        """
        if len(msg.joints_names) != len(msg.positions):
            return (False, 'joints_names and positions have not the same size')

        for index in range(0, len(msg.joints_names)):
            name = msg.joints_names[index].strip()
            
            name_is_correct = self.check_set_joint_name(name)
            
            if name_is_correct[0] == False:
                return name_is_correct

            pos = msg.positions[index]
            min_pos = self.joints_config[name]['min_value']
            max_pos = self.joints_config[name]['max_value']

            position_is_correct = self.check_set_joint_position(pos, min_pos, max_pos)
            if  position_is_correct[0] == False:
                return position_is_correct

        return (True, 'Arguments are correct')

    def check_set_joint_name(self, name):
        """ Checks that the name of the set_joint request is correct
        
        Args:
            msg: str 
                Containts the user input
        Return: (bool, str)
            A tuple that indicates if the name is correctly set.
            (True, msg)  : The name is correctly set
            (False, msg) : Something is wrong. The error is spread throgh the msg 
        """
        msg = 'Name is correct'
        if name == '':
            msg = 'joint_name cannot be empty'
            rospy.logerr('%s::check_set_joint_name: %s'%(self._node_name, msg))
            return (False, msg)
        
        if name not in self.joints:
            msg = 'joint_name not in the configurable joints'
            rospy.logerr('%s::check_set_joint_name: %s'%(self._node_name, msg))
            return (False, msg)
        
        return (True, msg)

    def check_set_joint_position(self, pos, min_pos, max_pos):
        """ Checks that the position of the set_joint request is correct
        
        Checks that the position (pos) is between the limits min_pos and
        max_pos

        Args:
            pos: float 
                Containts the user input
            min_pos: float
                Minimum position allowed by the joint
            max_pos: float
                Maximum position allowed by the joint

        Return: (bool, str)
            A tuple that indicates if the position is correctly set.
            (True, msg)  : The position is correctly set
            (False, msg) : Something is wrong. The error is spread throgh the msg 
        """
        msg = 'Position is correct'
        if pos < min_pos:
            msg = 'position is lower than the min_value (' + str(pos) + ' < '\
                  + str(min_pos) + ')'
            rospy.logerr('%s::check_set_joint_position: %s'%(self._node_name, msg))
            return (False, msg)
        
        if pos > max_pos:
            msg = 'position is higher than the max_value (' + str(pos) + ' > '\
                  + str(max_pos) + ')'
            rospy.logerr('%s::check_set_joint_position: %s'\
                        %(self._node_name, msg))
            return (False, msg)
        
        return (True, msg)

    def set_joint_response(self, success, message):
        """Builds the set_joint service response

        Args:
            success : bool
                Indicates if the set_joint request has been correctly received or not
            message : str
                Additional information about the request
        Return:
            Retruns a SetJointResponse that contains the specified
            information passed by argument
        """
        response = SetJointResponse()
        response.success = success
        response.message = message

        return response

    def set_joints_response(self, success, message):
        """Builds the set_joint service response

        Args:
            success : bool
                Indicates if the set_joint request has been correctly received or not
            message : str
                Additional information about the request
        Return:
            Retruns a SetJointResponse that contains the specified
            information passed by argument
        """
        response = SetJointsResponse()
        response.success = success
        response.message = message

        return response
    
    def write_desired_joint_position(self, joint_name, position):
        index = self.get_joint_index(joint_name)
        self.desired_joint_states.position[index] = position

    def get_joint_index(self, joint_name):
        """Returns the index of the joint_name in the joint_states list"""
        return self.joint_states.name.index(joint_name)

    def update_joints_positions(self, elapsed_time):
        num_positions = len(self.joint_states.position) 
        for index in range(0, num_positions):
            self.update_joint_position(index, elapsed_time)
            
    def update_joint_position(self, index, elapsed_time):
        real_position = self.joint_states.position[index]
        desired_position = self.desired_joint_states.position[index]
        if (real_position != desired_position):
            joint_name = self.joint_states.name[index]
            velocity = self.joints_config[joint_name]['velocity']

            if velocity == 0.0:
                increment = desired_position - real_position
            else:
                increment = velocity * elapsed_time

            if real_position < desired_position:
                self.increase_position(index, increment, desired_position)
            elif real_position > desired_position:
                self.decrease_position(index, increment, desired_position)

            self.joint_states.velocity[index] = velocity
        else:
            self.joint_states.velocity[index] = 0.0

    def increase_position(self, index, increment, desired_position):
        self.joint_states.position[index] += increment
        real_position = self.joint_states.position[index]
        if real_position >= desired_position: 
            self.joint_states.position[index] = desired_position
        
    def decrease_position(self, index, increment, desired_position):
        self.joint_states.position[index] -= increment
        real_position = self.joint_states.position[index]
        if real_position <= desired_position: 
            self.joint_states.position[index] = desired_position

def main():
    rospy.init_node("JointPublisher")
    publisher = JointPublisher()
    rospy.loginfo('%s: starting'%(rospy.get_name()))
    publisher.start()

if __name__ == "__main__":
    main()
    exit()
