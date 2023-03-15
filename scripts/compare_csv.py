import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_time_series(ts_given, ts_created, rows, labels = ["ts_given", "ts_created"], filename = 'plot.png', title='figure'):
    '''plot the time series in order to verify if they are more or less equals'''

    fig, axs = plt.subplots(rows)
    fig.suptitle(title)
    t_given = ts_given.index.values
    t_created = ts_created.index.values
    for i, col in enumerate(ts_given.columns):
        axs[i].plot(t_given, ts_given[col], 'r', linewidth=3, label=labels[0])
        axs[i].plot(t_created, ts_created[col], label=labels[1])
        axs[i].title.set_text(col)
    # plt.legend(loc = 'best')
    plt.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.savefig(filename, dpi=500)
    plt.show()
    

def remove_data_front_laser_dataframe(front_laser_dataframe):
    '''Function to keep just the ranges data from the laser sensor'''
    cols_to_keep = []
    for i in range(0, 541):
        element = 'ranges_' + str(i)
        cols_to_keep.append(element)
    front_laser_dataframe = front_laser_dataframe[cols_to_keep]
    return front_laser_dataframe


def compare_bag_files():
    # TO COMPARE THE FILES I COPIED THE FIRST ROW FROM THE GIVEN CSVs AND USE THE TIME OF THE GIVEN
    # CSV ALSO FOR MY DATA

    # NOMINAL 0
    robot_pose_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan/fufi-robot_pose.csv')
    robot_pose_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/robot_pose_nominal_0.csv')
    robot_pose_mine["Time"] = robot_pose_given["Time"]
    robot_pose_given["Time"] = robot_pose_given["Time"]
    robot_pose_mine["Time"] = robot_pose_mine["Time"]
    robot_pose_given = robot_pose_given.set_index("Time")
    robot_pose_mine = robot_pose_mine.set_index("Time")
    
    plot_time_series(robot_pose_given, robot_pose_mine, rows = 7, filename='robot_pose_bag_nominal_0.png', title="robot_poses_raw_nominal_0")


    front_laser_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan/fufi-front_laser-scan.csv')
    front_laser_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/laser_data_nominal_0.csv')
    front_laser_mine["Time"] = front_laser_given["Time"]
    front_laser_given["Time"] = front_laser_given["Time"]
    front_laser_mine["Time"] = front_laser_mine["Time"]
    front_laser_given = front_laser_given.set_index("Time")
    front_laser_mine = front_laser_mine.set_index("Time")
    
    # left_lasers = np.mean(laser_range[0:180])
    # front_lasers = np.mean(laser_range[180:360])
    # right_lasers = np.mean(laser_range[360:541])
    
    front_laser_given = remove_data_front_laser_dataframe(front_laser_given)
    front_laser_given["LL"] = front_laser_given[front_laser_given.columns[0:180]].mean(axis=1)
    front_laser_given["LC"] = front_laser_given[front_laser_given.columns[180:360]].mean(axis=1)
    front_laser_given["LR"] = front_laser_given[front_laser_given.columns[360:541]].mean(axis=1)
    front_laser_mine["LL"] = front_laser_mine[front_laser_mine.columns[0:180]].mean(axis=1)
    front_laser_mine["LC"] = front_laser_mine[front_laser_mine.columns[180:360]].mean(axis=1)
    front_laser_mine["LR"] = front_laser_mine[front_laser_mine.columns[360:541]].mean(axis=1)
    means_given = front_laser_given[["LL", "LC", "LR"]]
    means_mine = front_laser_mine[["LL", "LC", "LR"]]
    plot_time_series(means_given, means_mine, rows = 3, filename='lasers_bag_nominal_0.png', title="robot_lasers_raw_nominal_0")


    # NOMINAL 1
    robot_pose_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan_1/fufi-robot_pose.csv')
    robot_pose_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/robot_pose_nominal_1.csv')
    robot_pose_mine["Time"] = robot_pose_given["Time"]
    robot_pose_given["Time"] = robot_pose_given["Time"]
    robot_pose_mine["Time"] = robot_pose_mine["Time"]
    robot_pose_given = robot_pose_given.set_index("Time")
    robot_pose_mine = robot_pose_mine.set_index("Time")
    
    plot_time_series(robot_pose_given, robot_pose_mine, rows = 7, filename='robot_pose_bag_nominal_1.png', title="robot_poses_raw_nominal_1")


    front_laser_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan_1/fufi-front_laser-scan.csv')
    front_laser_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/laser_data_nominal_1.csv')
    front_laser_mine["Time"] = front_laser_given["Time"]
    front_laser_given["Time"] = front_laser_given["Time"]
    front_laser_mine["Time"] = front_laser_mine["Time"]
    front_laser_given = front_laser_given.set_index("Time")
    front_laser_mine = front_laser_mine.set_index("Time")
    
    # left_lasers = np.mean(laser_range[0:180])
    # front_lasers = np.mean(laser_range[180:360])
    # right_lasers = np.mean(laser_range[360:541])
    
    front_laser_given = remove_data_front_laser_dataframe(front_laser_given)
    front_laser_given["LL"] = front_laser_given[front_laser_given.columns[0:180]].mean(axis=1)
    front_laser_given["LC"] = front_laser_given[front_laser_given.columns[180:360]].mean(axis=1)
    front_laser_given["LR"] = front_laser_given[front_laser_given.columns[360:541]].mean(axis=1)
    front_laser_mine["LL"] = front_laser_mine[front_laser_mine.columns[0:180]].mean(axis=1)
    front_laser_mine["LC"] = front_laser_mine[front_laser_mine.columns[180:360]].mean(axis=1)
    front_laser_mine["LR"] = front_laser_mine[front_laser_mine.columns[360:541]].mean(axis=1)
    means_given = front_laser_given[["LL", "LC", "LR"]]
    means_mine = front_laser_mine[["LL", "LC", "LR"]]
    means_given = means_given.reset_index(drop=True)
    means_mine = means_mine.reset_index(drop=True)
    plot_time_series(means_given, means_mine, rows = 3, filename='lasers_bag_nominal_1.png', title="robot_lasers_raw_nominal_1")


    # ANOMALY 0
    robot_pose_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/window_plan/fufi-robot_pose.csv')
    robot_pose_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/robot_pose_anomaly_0.csv')
    robot_pose_mine["Time"] = robot_pose_given["Time"]
    robot_pose_given["Time"] = robot_pose_given["Time"]
    robot_pose_mine["Time"] = robot_pose_mine["Time"]
    robot_pose_given = robot_pose_given.set_index("Time")
    robot_pose_mine = robot_pose_mine.set_index("Time")
    
    plot_time_series(robot_pose_given, robot_pose_mine, rows = 7, filename='robot_pose_bag_anomaly_0.png', title="robot_poses_raw_anomaly_0")


    front_laser_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/window_plan/fufi-front_laser-scan.csv')
    front_laser_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/laser_data_anomaly_0.csv')
    front_laser_mine["Time"] = front_laser_given["Time"]
    front_laser_given["Time"] = front_laser_given["Time"]
    front_laser_mine["Time"] = front_laser_mine["Time"]
    front_laser_given = front_laser_given.set_index("Time")
    front_laser_mine = front_laser_mine.set_index("Time")
    
    # left_lasers = np.mean(laser_range[0:180])
    # front_lasers = np.mean(laser_range[180:360])
    # right_lasers = np.mean(laser_range[360:541])
    
    front_laser_given = remove_data_front_laser_dataframe(front_laser_given)
    front_laser_given["LL"] = front_laser_given[front_laser_given.columns[0:180]].mean(axis=1)
    front_laser_given["LC"] = front_laser_given[front_laser_given.columns[180:360]].mean(axis=1)
    front_laser_given["LR"] = front_laser_given[front_laser_given.columns[360:541]].mean(axis=1)
    front_laser_mine["LL"] = front_laser_mine[front_laser_mine.columns[0:180]].mean(axis=1)
    front_laser_mine["LC"] = front_laser_mine[front_laser_mine.columns[180:360]].mean(axis=1)
    front_laser_mine["LR"] = front_laser_mine[front_laser_mine.columns[360:541]].mean(axis=1)
    means_given = front_laser_given[["LL", "LC", "LR"]]
    means_mine = front_laser_mine[["LL", "LC", "LR"]]
    means_given = means_given.reset_index(drop=True)
    means_mine = means_mine.reset_index(drop=True)

    plot_time_series(means_given, means_mine, rows = 3, filename='lasers_bag_anomaly_0.png', title="robot_lasers_raw_anomaly_0")

    # ANOMALY 1
    robot_pose_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/window_plan_1/fufi-robot_pose.csv')
    robot_pose_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/robot_pose_anomaly_1.csv')
    robot_pose_mine["Time"] = robot_pose_given["Time"]
    robot_pose_given["Time"] = robot_pose_given["Time"]
    robot_pose_mine["Time"] = robot_pose_mine["Time"]
    robot_pose_given = robot_pose_given.set_index("Time")
    robot_pose_mine = robot_pose_mine.set_index("Time")
    
    plot_time_series(robot_pose_given, robot_pose_mine, rows = 7, filename='robot_pose_bag_anomaly_1.png', title="robot_poses_raw_anomaly_1")
    

    front_laser_given = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/window_plan_1/fufi-front_laser-scan.csv')
    front_laser_mine = pd.read_csv('src/anomaly_detection/data/csv/raw_data_ros/laser_data_anomaly_1.csv')
    front_laser_mine["Time"] = front_laser_given["Time"]
    front_laser_given["Time"] = front_laser_given["Time"]
    front_laser_mine["Time"] = front_laser_mine["Time"]
    front_laser_given = front_laser_given.set_index("Time")
    front_laser_mine = front_laser_mine.set_index("Time")
    front_laser_given["mean"] = front_laser_given.mean(axis=1)
    
    # left_lasers = np.mean(laser_range[0:180])
    # front_lasers = np.mean(laser_range[180:360])
    # right_lasers = np.mean(laser_range[360:541])
    
    front_laser_given = remove_data_front_laser_dataframe(front_laser_given)
    front_laser_given["LL"] = front_laser_given[front_laser_given.columns[0:180]].mean(axis=1)
    front_laser_given["LC"] = front_laser_given[front_laser_given.columns[180:360]].mean(axis=1)
    front_laser_given["LR"] = front_laser_given[front_laser_given.columns[360:541]].mean(axis=1)

    front_laser_mine["LL"] = front_laser_mine[front_laser_mine.columns[0:180]].mean(axis=1)
    front_laser_mine["LC"] = front_laser_mine[front_laser_mine.columns[180:360]].mean(axis=1)
    front_laser_mine["LR"] = front_laser_mine[front_laser_mine.columns[360:541]].mean(axis=1)
    
    means_given = front_laser_given[["LL", "LC", "LR"]]
    means_mine = front_laser_mine[["LL", "LC", "LR"]]
    
    means_given = means_given.reset_index(drop=True)
    means_mine = means_mine.reset_index(drop=True)
    
    plot_time_series(means_given, means_mine, rows = 3, filename='lasers_bag_anomaly_1.png', title="robot_lasers_raw_anomaly_1")

    # NOMINAL 0 COMPARED TO NOMINAL 1
    # # Comparison of the 2 nominal files
    # # nominal 0
    # robot_pose_nominal = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan/fufi-robot_pose.csv')
    # front_laser_nominal = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan/fufi-front_laser-scan.csv')
    # # nominal 1
    # robot_pose_nominal_1 = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan_1/fufi-robot_pose.csv')
    # front_laser_nominal_1 = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/normal_plan_1/fufi-front_laser-scan.csv')

    # robot_pose_nominal_1["Time"] = robot_pose_nominal["Time"]
    # robot_pose_nominal = robot_pose_nominal.set_index("Time")
    # robot_pose_nominal_1 = robot_pose_nominal_1.set_index("Time")

    # front_laser_nominal = remove_data_front_laser_dataframe(front_laser_nominal)
    # front_laser_nominal_1 = remove_data_front_laser_dataframe(front_laser_nominal_1)

    # front_laser_nominal["LL"] = front_laser_nominal[front_laser_nominal.columns[0:180]].mean(axis=1)
    # front_laser_nominal["LC"] = front_laser_nominal[front_laser_nominal.columns[180:360]].mean(axis=1)
    # front_laser_nominal["LR"] = front_laser_nominal[front_laser_nominal.columns[360:541]].mean(axis=1)
    
    # front_laser_nominal_1["LL"] = front_laser_nominal_1[front_laser_nominal_1.columns[0:180]].mean(axis=1)
    # front_laser_nominal_1["LC"] = front_laser_nominal_1[front_laser_nominal_1.columns[180:360]].mean(axis=1)
    # front_laser_nominal_1["LR"] = front_laser_nominal_1[front_laser_nominal_1.columns[360:541]].mean(axis=1)
    
    # means_nominal = front_laser_nominal[["LL", "LC", "LR"]]
    # means_nominal_1 = front_laser_nominal_1[["LL", "LC", "LR"]]

    # plot_time_series(ts_given=robot_pose_nominal, ts_created=robot_pose_nominal_1, rows=7, labels=["nominal", "nominal_1"], filename='robot_pose_nominal_0_1.png', title="robot_poses_raw_nominal_0_1")
    # plot_time_series(ts_given=means_nominal, ts_created=means_nominal_1, rows=3, labels=["nominal", "nominal_1"], filename='laser_nominal_0_1.png', title="robot_lasers_raw_nominal_0_1")

def compare_offline_online_detection():
    offline_csv = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/Offline_Anomaly_Detection_Csv/dpNm0.csv')
    # online_csv = pd.read_csv('./src/anomaly_detection/data/csv/trainHMM.csv')
    online_csv = pd.read_csv('src/anomaly_detection/data/csv/preprocess_data_ros/nominal_0.csv')
    online_csv = online_csv[::2]
    online_csv = online_csv.reset_index(drop=True)
    plot_time_series(ts_given=offline_csv, ts_created=online_csv, rows=6, labels=['dpNm0', 'nominal_0'], filename='nominal_0.png', title="NOMINAL 0 PREPROCESSED")

    offline_csv = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/Offline_Anomaly_Detection_Csv/dpNm1.csv')
    # online_csv = pd.read_csv('./src/anomaly_detection/data/csv/trainHMM.csv')
    online_csv = pd.read_csv('src/anomaly_detection/data/csv/preprocess_data_ros/nominal_1.csv')
    online_csv = online_csv[::2]
    online_csv = online_csv.reset_index(drop=True)
    plot_time_series(ts_given=offline_csv, ts_created=online_csv, rows=6, labels=['dpNm1', 'nominal_1'], filename='nominal_1.png', title="NOMINAL 1 PREPROCESSED")

    offline_csv = pd.read_csv('src/anomaly_detection/data/csv/all_data_offline_work/Offline_Anomaly_Detection_Csv/dpNmA.csv')
    # online_csv = pd.read_csv('./src/anomaly_detection/data/csv/trainHMM.csv')
    online_csv = pd.read_csv('src/anomaly_detection/data/csv/preprocess_data_ros/anomaly_0.csv')
    online_csv = online_csv[::2]
    online_csv = online_csv.reset_index(drop=True)
    plot_time_series(ts_given=offline_csv, ts_created=online_csv, rows=6, labels=['dpNmA', 'anomaly_0'], filename='anomaly_0.png', title="ANOMALY 0 PREPROCESSED")


if __name__ == "__main__":
    # Comparison #1: bag files nominals vs mines and anomalous vs mines (without preprocessing)
    pd.set_option('display.max_columns', 10)
    # compare_bag_files()
    # Comparison #2: already preprocess csv files vs mines for nominals and anomalous data
    # compare_offline_online_detection()

    hel_nominal_0_ros = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/ROS/hellingers_nominal_0.csv')
    hel_nominal_0_offline = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/OFFLINE/hellingers_learning.csv')
    hel_nominal_0_ros = hel_nominal_0_ros[::2]
    hel_nominal_0_ros = hel_nominal_0_ros.reset_index(drop=True)
    plt.plot(hel_nominal_0_ros.index.values, hel_nominal_0_ros, 'r', label=hel_nominal_0_ros.columns.values[0])
    plt.plot(hel_nominal_0_offline.index.values, hel_nominal_0_offline, label=hel_nominal_0_offline.columns.values[0])
    plt.legend()
    plt.show()

    hel_nominal_1_ros = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/ROS/hellingers_nominal_1.csv')
    hel_nominal_1_offline = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/OFFLINE/hellingers_validation.csv')
    hel_nominal_1_ros = hel_nominal_1_ros[::2]
    hel_nominal_1_ros = hel_nominal_1_ros.reset_index(drop=True)
    plt.plot(hel_nominal_1_ros.index.values, hel_nominal_1_ros, 'r', label = hel_nominal_1_ros.columns.values[0])
    plt.plot(hel_nominal_1_offline.index.values, hel_nominal_1_offline, label = hel_nominal_1_offline.columns.values[0])
    plt.legend()
    plt.show()

    hel_anomaly_0_ros = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/ROS/hellingers_anomaly_0.csv')
    hel_anomaly_0_offline = pd.read_csv('src/anomaly_detection/data/csv/hellingers_offline_online/OFFLINE/hellingers_anomaly.csv')
    hel_anomaly_0_ros = hel_anomaly_0_ros[::2]
    hel_anomaly_0_ros = hel_anomaly_0_ros.reset_index(drop=True)
    plt.plot(hel_anomaly_0_ros.index.values, hel_anomaly_0_ros, 'r', label = hel_anomaly_0_ros.columns.values[0])
    plt.plot(hel_anomaly_0_offline.index.values, hel_anomaly_0_offline, label = hel_anomaly_0_offline.columns.values[0])
    plt.legend()
    plt.show()