# rosbag_manager 
Package intended to start and stop rosbag processes

## subscribed topics

None

## published topics

**~/state** (rosbag_manager_msgs/RosbagManagerState)

Current state of the component

Example:
```
header: 
  seq: 2464
  stamp: 
    secs: 4496
    nsecs: 570000000
  frame_id: ''
state: 
  state: 200
  desired_freq: 10.0
  real_freq: 7.1961607933
  state_description: "STANDBY_STATE"
recording: False
time_recording: 4
path: "/home/user/test_rosbags/"
bag_name: "2019-04-01_14_59_10"
compression: True
stored_size: 0.031852
```

## advertised services

**~/set_recording** (rosbag_manager_msgs/Record)

Starts/stops the rosbag on demand.

*service params*

* action (string): RecordRequest.START (start), RecordRequest.STOP (stop), RecordRequest.DISCARD (discard)  
* name (string): filename of the rosbag 
* path (string): path folder to save the rosbag. If empty it'll take the defined in the ROS params.

Example:
```
 rosservice call /robot/rosbag_manager/set_recording "action: 'start'
name: ''
path: ''"
```

## parameters

**output_path** (string, default: "rosbag_manager")

Path to save all the files.

Path can be absolute, relative to running folder and with environment variables (which will be expanded, such as $HOME or ~)

**buffer_size** (int, default: 2048)

Size in MB.
This is the message queue of the recorder object, before messages are being passed on to the bag. Lowering this value might result in messages being dropped before they reach the recording process.

**chunk_size** (int, default: 768)

Advanced. Record to chunks of SIZE KB (Default: 768). This is a buffer within the bag file object. Lowering this value will result in more writes to disk.

**topics** (array of strings, default: [])
List of topics to save in the rosbag

Topics can be:
* absolute
* relative to current namespace (public namespace of the node, not private)
* expressed as regular expressions

**split**  (bool, default: False)

Split the bag when maximum size or duration is reached

**split_size** (int, default: 1024)

Size of every splitted bag in MB.

**compressed** (bool, true)

Flag to compress the rosbag
