#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan, JointState, Temperature, Imu
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry

from robot_local_control_msgs.msg import LocalizationStatus

# to see execution time later
import time
from datetime import datetime

# To load the HMM model already trained
import pickle
import math
import numpy as np
import pandas as pd
from hmmlearn import hmm

# Plot the data in real time
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go

# CSV library
import csv

import psutil

import os
import signal


def quaternion_to_euler(w, x, y, z):
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


def H1(i, cov1, cov2):
    """Computes H1 metric as in the paper"""
    sigma1 = cov1[i, i]
    sigma2 = cov2[i, i]
    return ((np.sqrt(2) * (sigma1 * sigma2) ** (0.25)) / ((sigma1 + sigma2) ** (0.5)))


def H2(i, mean1, cov1, mean2, cov2):
    """Computes H1 metric as in the paper"""
    mean1_i = mean1[i]
    mean2_i = mean2[i]
    sigma1 = cov1[i, i]
    sigma2 = cov2[i, i]
    return (((mean1_i - mean2_i) ** 2) / (4 * (sigma1 + sigma2)))


def avoid_null_variance(model_v, most_frequent_variance):
    """Avoid to have null variance when we have 0 at the denominator"""
    for i in range(len(model_v[0])):
        temp = model_v[i, i] * 0.5
        model_v[i, i] = model_v[i, i] + temp
        most_frequent_variance[i] = most_frequent_variance[i] + temp
    return model_v, most_frequent_variance


def hellinger_distance(mean1, cov1, mean2, cov2):
    """Hellinger distance manual computation as in the paper"""
    first_operator = (np.dot(np.linalg.det(cov1) ** (0.25), np.linalg.det(cov2) ** (0.25)) / (
                np.linalg.det((cov1 + cov2) / 2.0) ** (0.5)))
    second_operator = np.exp(np.dot((-1 / 8.0),
                                    np.dot(np.dot(np.transpose(mean1 - mean2), np.linalg.inv((cov1 + cov2) / 2.0)),
                                           (mean1 - mean2))))
    hel = 1 - np.dot(first_operator, second_operator)
    return hel


def is_anomaly(h2_variables, state_means, state_covariance):
    """Used to compute a threshold for the H2 metric"""
    global h2_thr
    h2_variables = np.array(h2_variables)
    state_variance = np.diag(state_covariance)
    state_std = np.sqrt(state_variance)
    threshold = np.abs(state_means + (10 * state_std))
    # is_anomaly was an experiment to see if a single threshold 
    # works also for the variables inside the H2 metric. 
    # Experiment failed so you can delete this part
    is_anomaly = np.where(h2_variables > threshold, True, False)
    h2_thr = np.where(h2_variables > h2_thr, h2_variables, h2_thr)
    return is_anomaly


def window_processing(dataHMM, hmm_model):
    '''Predict anomalies of KAIROS robot by processing a window of data'''
    global h2_variables
    # global hellinger_distances
    hmm.GaussianHMM(algorithm='viterbi')
    prediction = hmm_model.predict(dataHMM)  # predict a state for each read row in dataHMM
    most_frequent_state = np.argmax(np.bincount(prediction))  # check which is the most frequent state
    data_frequent = dataHMM[prediction == most_frequent_state]  # returns the rows with the most frequent state
    # mean and variance
    mean = np.array(np.mean(data_frequent, axis=0))  # mean of all the rows in data_frequent
    variance = np.var(data_frequent, axis=0)
    # Hellinger base score
    model_m = hmm_model.means_[most_frequent_state]
    model_v = hmm_model.covars_[most_frequent_state]
    n = len(data_frequent)
    if n == 1:
        print("ERROR: can't calculate the confidence interval because of n = 1 --> df = 0\n")
        exit(-1)
    # variance correction by adding a delta to avoid a null value
    model_v, variance = avoid_null_variance(model_v, variance)
    hel_score = hellinger_distance(mean1=model_m, cov1=model_v, mean2=mean, cov2=np.diag(variance))
    size = np.shape(model_v)[0]  # square matrix so rows = cols, 6*6
    # LOOP WHICH COMPUTES THE HELLINGER DECOMPOSITION
    variables_decomposition = []
    for i in range(0, size):
        # h1 = H1(i=i, cov1=model_v, cov2=np.diag(variance))
        h2 = H2(i=i, mean1=model_m, cov1=model_v, mean2=mean, cov2=np.diag(variance))
        variables_decomposition.append(h2)
    # anomaly_behaviour = is_anomaly(h2_variables=variables_decomposition,state_means=model_m, state_covariance=model_v)
    h2_variables = np.append(h2_variables, np.array([variables_decomposition]), axis=0)

    with open(h2_filename, 'a') as h2_csv:
        writer = csv.writer(h2_csv)
        writer.writerow(variables_decomposition)

    # with open('anomaly_1_decomposition.npy', 'wb') as f:
    #     np.save(f, h2_variables)
    return hel_score, np.array(variables_decomposition)


# callback which is executed every time we receive a message from the wanted topic 
def front_laser_callback(data):
    global dataHMM
    global row
    global semaphore
    laser_range = np.nan_to_num(data.ranges)
    left_lasers = np.mean(laser_range[0:180])
    front_lasers = np.mean(laser_range[180:360])
    right_lasers = np.mean(laser_range[360:541])
    if not semaphore:
        semaphore = True
        row[3:6] = [left_lasers, front_lasers, right_lasers]


def robot_pose_callback(data):
    global dataHMM
    global row
    global semaphore
    global X_first
    global Y_first
    global points
    global anomaly

    position = data.position
    quaternion_orientation = data.orientation
    euler_orientation = quaternion_to_euler(quaternion_orientation.w, quaternion_orientation.x,
                                            quaternion_orientation.y, quaternion_orientation.z)
    Z_orientation = euler_orientation[2]
    X = position.x
    Y = position.y
    # Z_orientation = position.theta

    # check_nan = np.sum(row)
    if semaphore:
        if np.shape(dataHMM)[0] == 0:
            X_first = X
            Y_first = Y
            # real_time_position_plot(X_current = (X - X_first), Y_current = (Y - Y_first), color = 'g+', marker = 13)
        X = X - X_first
        Y = Y - Y_first
        row[0:3] = [X, Y, Z_orientation]
        dataHMM = np.append(dataHMM, np.array([row]), axis=0)
        with open(training_filename, 'a') as train_csv:
            writer = csv.writer(train_csv, delimiter=',')
            writer.writerow(row)
        n_rows = np.shape(dataHMM)[0]
        # when the number of rows, so the number of collected data is equal to the window length
        # I process the data using the HMM
        if n_rows == w:
            hel_score, h2 = window_processing(dataHMM=dataHMM, hmm_model=model)
            # all hellinger distances in order to have a complete plot over time
            hellinger_distances.append(hel_score)

            # with open('hellingers_anomaly_1.npy', 'wb') as f:
            #     np.save(f, hellinger_distances)

            # POSITION ANOMALY USING THE TOTAL HELLINGER
            # if hel_score > threshold_nominal_1_data:
            #     anomaly = True
            # else:
            #     anomaly = False

            # POSITION ANOMALY USING THE HELLINGER DECOMPOSITION - ANOMALOUS IF ONE SENSOR ANOMALOUS
            anomaly = np.any(h2 > h2_thr)
            dataHMM = np.delete(dataHMM, (0), axis=0)
        # use points to plot the map
        points.append([X, Y, anomaly])

        # WRITE ALL THE ROWS ON A CSV FILE IN ORDER TO PLOT THE DATA
        with open(map_filename, 'a') as map_csv:
            writer = csv.writer(map_csv)
            writer.writerow([X, Y, anomaly])

        row = np.full(cols, np.nan)
        semaphore = False


# main function in which we call all the above callbacks for each topic
def listener():
    rospy.init_node('listener', anonymous=True)
    # Subscribe to the KAIROS sensors
    # The HMM has to process mainly the front_laser and the robot_pose
    # robot_pose average_rate : 10.085 Hz
    rospy.Subscriber(robot_pose_topic, Pose, robot_pose_callback)
    # NEW TOPIC WITH ROBOTNIK UPDATED TOPIC
    # rospy.Subscriber(robot_pose_topic, LocalizationStatus, robot_pose_callback)
    # front_laser average_rate : 13.083 Hz
    rospy.Subscriber(front_laser_topic, LaserScan, front_laser_callback)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


# NEW GUI: TO BE FINISHED
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1000,
            n_intervals=0
        )
    ]
)


@app.callback(Output('live-graph', 'figure'), Input('graph-update', 'n_intervals'))
def update_graph(n):
    # Here we will call our sensors to plot
    global points
    if len(points) > 0:
        points_array = np.array(points)
        x = points_array[:, 0]
        y = points_array[:, 1]
        anomaly = points_array[:, 2]
        color = np.where(anomaly == 1, 'red', 'blue')
        data = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=color
            )
        )
        return {
            'data': [data],
            'layout': go.Layout(
                xaxis=dict(range=[-1.5, 0.5]),
                yaxis=dict(range=[-5, 1])
            )}


if __name__ == '__main__':

    map_filename = "map_data.csv"
    h2_filename = "h2_decomposition_data.csv"
    training_filename = "training_data.csv"
    with open(map_filename, 'w') as map_csv:
        writer = csv.writer(map_csv)
        head = ['X', 'Y', 'anomaly']
        writer.writerow(head)

    with open(h2_filename, 'w') as h2_csv:
        writer = csv.writer(h2_csv)
        head = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
        writer.writerow(head)

    with open(training_filename, 'w') as h2_csv:
        writer = csv.writer(h2_csv)
        head = ['X', 'Y', 'O', 'LS', 'LC', 'LD']
        writer.writerow(head)

    # OLD USED TOPICS
    front_laser_topic = "/fufi/front_laser/scan"
    robot_pose_topic = "/fufi/robot_pose"

    # NEW TOPICS REMAPPED IN ICE LAB
    # front_laser_topic = "/robot/front_laser/scan"
    # robot_pose_topic = "/robot/robot_local_control/LocalizationComponent/status"

    # WINDOW LENGTH
    w = 30
    # w = 40

    # THRESHOLD OBTAINED RUNNING THE NODE ON THE nominal_1 file with w = 30
    threshold_nominal_1_data = 0.9994715267699249
    # THRESHOLD OBTAINED RUNNING THE NODE ON THE nominal_1 file with w = 40
    # threshold_nominal_1_data = 0.9996722536629078

    # COLUMNS INTO WHICH WE GROUP THE DATA -> X, Y, ORIENTATION, LEFT_LASERS, FRONT_LASERS, RIGHT_LASERS
    cols = 6

    # EMPTY ARRAY INTO WHICH WE GROUP THE DATA THAT ARE RECEIVED AT SEVERAL SAMPLE RATES
    dataHMM = np.empty((0, cols), float)
    # hellinger_distances = []

    row = np.full(cols, np.nan)
    semaphore = False
    # use thresh to choose a threshold
    anomaly = False

    points = []
    h2_variables = np.empty((0, cols), float)
    hellinger_distances = []
    # h2_thr = np.zeros((6))
    # DATASET NOMNALE 2
    h2_thr = [1.26020238, 6.67861522, 0.4251171, 0.70920265, 0.94272347, 0.89692743]

    # HMM model takes the X, Y, Orientation, Left, Front and Right Lasers.
    # The orientation is taken has Euler Angles, so remember to convert them.
    #
    # Ad ogni ciclo colleziono i dati nuovi, li preprocesso, passo tutta la matrice all'anomaly detector
    # gia' caricata nel programma.

    # file name of the hmm saved model. Insert the model path
    # model_filename = './src/anomaly_detection/data/HMM_models/hmm.pkl'
    model_filename = 'src/anomaly_detection/data/HMM_models/hmm_ros_data_centered_9_states.pkl'

    # Load the HMM from the disk
    model = pickle.load(open(model_filename, 'rb'))

    if model is not None:
        print("HMM model loaded!")
        print("Models means: " + str(np.shape(model.means_)))
        print("Model covariance: " + str(np.shape(model.covars_)))

    # Generate samples
    X_sample, Z_sample = model.sample(250)

    X_first = None
    Y_first = None

    # ROS NODE FOR READING SENSORS
    listener()

    app.run_server(Debug=True)
    print("Running Server!")
