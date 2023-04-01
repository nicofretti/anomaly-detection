# Anomaly detection on Kairos
- `kairos_ws-robot_b_work`: contains all source packages for the robot
- `data`: contains all data used for the anomaly detection
- `scripts`: contains all scripts used for the anomaly detection, using Python2.7
- `web`: web interface for the anomaly detection, using Python3

## Environment setup
- Ubuntu 18.04
- ROS Melodic
- Python 2.7 for the anomaly detection scripts (root `requirements.txt`)
- Python 3.8 for the web interface (`web/requirements.txt`)

## Installation and run(TODO)
1. Install ROS Melodic following the documentation
2. Install the packages located in `kairos_ws-robot_b_work/src/deb` 
3. Install the requirements.txt in your python2.7 (attention that the requirements.txt includes only python libraries without ROS dependencies, them are installed in the previous points)
4. Make the folder catkin_ws following the ROS documentation
5. Clone the repository in your catkin_ws/src
6. Build the workspace
7. Source the workspace
8. Run the `rosrun anomaly_detection listener.py` script
 
