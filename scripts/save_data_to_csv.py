#!/usr/bin/env python

#  THIS SCRIPT SAVES THE DATA FROM A BAG FILE normal_plan.bag INTO A .csv FILE WHICH WILL BE PROCESSED BY create_hmm.py

import rospy 
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose

# to print messages in a pretty format
from pprint import pprint


# To load the HMM model already trained
import math
import numpy as np

# we want position (X, Y), orientation and lasers


cols = 6
train = np.empty((0, cols), float)
row = np.full(cols, np.nan)
semaphore = False


# USED TOPICS
front_laser_topic = "/fufi/front_laser/scan"
robot_pose_topic = "/fufi/robot_pose"

def quaternion_to_euler(w,x,y,z):
    """
    Converts the quaternion given by ROS into the corresponding Euler angles.
    """
    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = 2 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = math.asin(t2)
        
    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


laser_data = np.empty((0, 541),float)
# callback which is executed every time we receive a message from the wanted topic 
def front_laser_callback(data):
    # TO SAVE THE RAW DATA
    global laser_data
    laser_range = np.array(data.ranges)
    laser_data = np.append(laser_data, np.array([laser_range]), axis = 0)
    np.savetxt("laser_data_nominal_0.csv", laser_data, delimiter=',')
    print("Finish to save laser data")
    


robot_pose = np.empty((0, 7), float)
X_first = None
Y_first = None
def robot_pose_callback(data):
    # TO SAVE THE RAW DATA
    global row
    global robot_pose 
    position = data.position
    quaternion_orientation = data.orientation
    row = np.array([position.x, position.y, position.z, quaternion_orientation.x, quaternion_orientation.y, quaternion_orientation.z, quaternion_orientation.w])
    robot_pose = np.append(robot_pose, np.array([row]), axis = 0)
    np.savetxt("robot_pose_nominal_0.csv", robot_pose, delimiter=',')
    print("Finish to save pose data")

    

# main function in which we call all the above callbacks for each topic
def save_data_to_csv():
    
    rospy.init_node('save_data_to_csv', anonymous=True)
    # front_laser average_rate : 13.083 Hz
    rospy.Subscriber(front_laser_topic, LaserScan, front_laser_callback) 
    # robot_pose average_rate : 10.085 Hz
    rospy.Subscriber(robot_pose_topic, Pose, robot_pose_callback) 
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    save_data_to_csv()