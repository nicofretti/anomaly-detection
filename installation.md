## Installation lab ICE

There are two configuration files, one located in `scripts/config.ini` for the robot configuration and one located in `web/config.ini` for the web interface configuration. The topics must be the same in both files.

The `scripts` and `web` are two different projects, so they must be installed separately.

### Robot installation
For the robot is needed the folder `scripts` (the trained model is located in the same folder as `hmm_best.pkl`) and the other files located in the root directory excepted for the folder `web`. Extract them in a directory `anomalydetection` to create a ROS node. Then in the `config.ini` edit:
- `kairos - robot_name`: the robot name (one must be 'robot_1' and the other 'robot_2')
- `mqtt - map_topic`: is the topic for the map update (attention: the topic will be `${map_topic}/${robot_name}`, where `${robot_name}` is the robot name as specified in the previous point)
- `mqtt - decomposition_topic`: is the topic for the decomposition update (attention: the topic will be `${decomposition_topic}/${robot_name}`)
- `mqtt - host`: is the mqtt broker host
- `mqtt - port`: is the mqtt broker port

To run the anomaly detection script after creating the ROS node, type:
```bash
rosrun anomalydetection listener.py
```

### Web interface installation
In the folder `web` there everything needed for the web interface. After extracting the folder there is the `README.md` for running the docker. In the `config.ini` edit:
- `mqtt - host`: is the mqtt broker host
- `mqtt - port`: is the mqtt broker port
- `mqtt - map_topic`: must be the same as the robot (attention: is not needed to specify the robot name, the web interface will subscribe to all robots "robot_1" and "robot_2")
- `mqtt - decomposition_topic`: must be the same as the robot (attention: is not needed to specify the robot name, the web interface will subscribe to all robots "robot_1" and "robot_2")
- `app - nominal_0`: is the path of the nominal route (located in `web/data/nominal_0.csv`)

To change the port of where the web interface is listening, edit the `Dockerfile` and change the `EXPOSE` command, and is needed to type `docker run` with the changed port.


