# marker_mapping

Package to map visual markers and use them to initialize the robot position with amcl localization(amcl_pose).

It works either with AR tags or frames (new functionality).

## 1 marker_mapping_node

### 1.1 Params

* **desired_freq** (double)
  * Frequency of the internal control loop. 10 hz by default
* **frame_id** (string)
  * Frame to refer the saved markers. 'map' by default
* **base_frame_id** (string)
  * Frame to transform the marker with respect to the base. 'base_link' by default
* **publish_saved_markers_tf** (bool)
  * Flag to enable the tf publication of the markers
* **folder_path** (string)
  * System path to save markers config file
* **markers_filename** (string)
  * Filename of the markers config file.
* **load_markers_on_init** (bool)
  * Flag to load the saved markers when it starts
* **max_marker_id** (int)
  * Filter the markers with values higher than this one
* **min_marker_id** (int)
  * Filter the markers with values lower than this one
* **max_marker_distance** (double)
  * Filter the max distance to marker in order to allow saving markers and initializating the pose. If the distance to the marker is greater than this value, no action will be executed. 1.2 m by default

### 1.2 Topics

#### 1.2.1 Publishers

* **~state** (robotnik_msgs/State)
  * publishes the standard Robotnik's state of the component
  * The state is READY as long as it's detecting previously recorded markers.
  ```
  state:
  state: 300
  desired_freq: 10.0
  real_freq: 9.93986320496
  state_description: READY_STATE
  ```

* **~/status** (marker_mapping/MarkerMappingState)
  * publishes the state of the component
  * the state is READY if a marker is being detected
  * the ids of the detected and saved markers is published
 ```
 state:
  state: 300
  desired_freq: 10.0
  real_freq: 9.93986320496
  state_description: READY_STATE
 ids_recorded: [0, 1, 2]
 ids_detected: [2]
 ```
* **/initialpose** (geometry_msgs/PoseWithCovarianceStamped)
  * publishes initial pose (read by amcl)

#### 1.2.2 Subscribers

* /ar_pose_marker (ar_track_alvar_msgs/AlvarMarkers)
  * Gets the markers detected by the node ar_trac_alvar
 ```
 header:
  seq: 10177
  stamp:
    secs: 0
    nsecs:         0
  frame_id: ''
markers:
  -
    header:
      seq: 0
      stamp:
        secs: 1488819030
        nsecs: 565245573
      frame_id: base_link
    id: 2
    confidence: 0
    pose:
      header:
        seq: 0
        stamp:
          secs: 0
          nsecs:         0
        frame_id: ''
      pose:
        position:
          x: 0.763555542968
          y: -0.301370897543
          z: 0.69163953281
        orientation:
          x: -0.479586486245
          y: 0.528171186928
          z: 0.517824549675
          w: -0.472111994409

 ```

### 1.3 Services

* **~/save_maker** (marker_mapping/SaveMarker)
  * Saves the current or desired marker with respect to the frame_id
  * Params:
    * id as [int]. If no id is specified, it saves all the markers received in the topic ar_pose_marker
    * filename as string. It specifies the filename to save the config. If not set, it's saved in the default filename
  * Example:
  ```
  rosservice call /marker_mapping_node/save_maker "id: []
  filename: ''"
  ```
* **~/init_pose_from_marker** (marker_mapping/InitPoseFromMarker)
  * If a markers is being detected and it has been previously saved with respect to frame_id, it calculates the robot position and sends the initpose to the amcl node.
  * Params:
    * id as [int]. The id of the marker to use for the initialization. If no id provided, the node will use the first one received in ar_pose_marker as long as it has been mapped previously.
  * Example:
 ```
 rosservice call /marker_mapping_node/init_pose_from_marker "id: []"
 ```
* **~/init_pose** (std_srvs/Trigger)
  * Same functionality than init_pose_from_marker, but using a standard service message. No params available.  The node will use the closest detected marker as long as it has been saved previously.
  * Example:
  ```
  rosservice call /frame_mapping/init_pose "{}"
  ```

* **~/set_frame_id** (marker_mapping/SetFrameId)
  * Sets the current global frame id to save markers, init position and publish tf of the markers
  * Params:
    * frame_id as [string]
  * Example:
  ```
   rosservice call /marker_mapping/set_frame_id "frame_id: 'rb1_base_a_map'"
  ```


## 2 frame_mapping_node

### 2.1 Params
* **desired_freq** (double, default = 10)
  * Frequency of the internal control loop.
* **frame_id** (string, default = map)
  * Frame to refer the saved markers.
* **base_frame_id** (string, default = base_link)
  * Frame to transform the marker with respect to the base.
* **publish_saved_frames_tf** (bool, default = true)
  * Flag to enable the tf publication of the saved frames
* **folder_path** (string, default = marker_mapping/config)
  * System path to save markers config file
* **config_filename** (string, defautl = frames.yaml)
  * Filename of the frames config file. This file contains all the saved markers.
* **load_frames_on_init** (bool, default = true)
  * Flag to load the saved frams when it starts
* **max_frame_distance** (double, default = 1.25)
  * Filters the max distance from the base_frame to the frame in order to allow either saving frames or initializing the pose. If the distance to the marker is greater than this value, no action will be executed.
* **max_frame_orientation_diff** (double, default = 0.79)
  * Filters the max orientation difference from the base_frame to the frame in order to allow either saving frames or initializing the pose. ( ***NOT WORKING***)
* **default_pose_covariance** (double[36], default = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
  * Matrix of covariances sent to amcl initpose when we want to initialize the amcl pose.

### 2.2 Topics

#### 2.2.1 Publishers

* **~state** (robotnik_msgs/State)
  * publishes the standard Robotnik's state of the component
  * The state is READY as long as it's detecting previously recorded frames.
  ```
  state:
  state: 300
  desired_freq: 10.0
  real_freq: 9.93986320496
  state_description: READY_STATE
  ```

* **~status** (marker_mapping/FrameMappingStatus)
  * publishes the state of the component
  * the ids of the detected and saved markers is published.
  * ids_detected provides extra information
  ```
  header:
    seq: 572
    stamp:
      secs: 2042
      nsecs: 206000000
    frame_id: ''
  ids_recorded: ['1', frame_1, '2']
  ids_detected:
    -
      id: "2"
      current_distance: 0.912821173668
      max_distance: 1.25
      pose_from_frame:
        position:
          x: 0.436495314647
          y: 3.10270536264
          z: 8.60743489364e-05
        orientation:
          x: 0.0
          y: 0.0
          z: 0.710104515641
          w: 0.704096283804
  ```

* **/initialpose** (geometry_msgs/PoseWithCovarianceStamped)
  * publishes initial pose (read by amcl)


* **~pose_from_frames** (geometry_msgs/PoseArray)
  * publishes an array of poses of the robot based on the current detected ids. It shows what would be the position if we initialize the pose base on the detected frames.

#### 2.2.2 Subscribers

* **tf**

### 2.3 Services

* **~/save_frame** (marker_mapping/SaveFrame)
   * Saves the desired frame with respect to the frame_id
   * Params:
      * frame_id as [string]. If no id is specified, it returns error.
   * Example:
   ```
   rosservice call /rb1_base/frame_mapping/save_frame "frame_id: '2'"
   ```

* **~/init_pose_from_frame** (marker_mapping/InitPoseFromFrame)
  * If a markers is being detected and it has been previously saved with respect to frame_id, it calculates the robot position and sends the initpose to the amcl node.
  * Params:
    * frame_id as [string]. The frame id to use for the initialization. If no id provided, the node will use the closest detected frame as long as it has been saved previously.
  * Example:
  ```
  rosservice call /frame_mapping/init_pose_from_frame "frame_id: '1'"
  ```

* **~/init_pose** (std_srvs/Trigger)
  * Same functionality than init_pose_from_frame, but using a standard service message. No params available.  The node will use the closest detected frame as long as it has been saved previously.
  * Example:
  ```
  rosservice call /frame_mapping/init_pose "{}"
  ```

* **set_frame_id** (marker_mapping/SetFrameId)
  * Sets the current global frame id to save markers, init position and publish tf of the markers
  * Params:
    * frame_id as [string]
  * Example:
  ```
   rosservice call /rb1_base_a/marker_mapping/set_frame_id "frame_id: 'rb1_base_a_map'"
  ```

### 2.4 Frames configuration file

List of frames with the following params:
* id (string): frame id
* frame (string): parent frame used
* position(double[3]): position x, y, z of the frame
* orientation(double[4]): quaternion of the frame
* max_distance(double): max distance allowed to init the pose if we are detecting this frame
* pose_covariance(double[36]): matrix of covariance applied to the pose initialization

```
- id: 1
  frame: rb1_base_map
  position: [0.15255686833022147, 4.039648064779637, 0.13304999948717783]
  orientation: [0.0, 0.0, 0.6894313983660931, 0.7243509832580981]
  max_distance: 1.25
  pose_covariance: [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
- id: 2
  frame: rb1_base_map
  position: [0.6034587776849643, 4.027172764179653, 0.12861956381533607]
  orientation: [0.0, 0.0, 0.6996469231932343, 0.7144887562909863]
  max_distance: 1.25
  pose_covariance: [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]

```
### 2.5 Launch

Launch file available.
Arguments:

```
<arg name="frame_id" default="map"/>
<arg name="base_frame_id" default="base_footprint"/>
<arg name="config_file" default="frames.yaml"/>
<arg name="config_folder" default="$(find marker_mapping)/config" />
<arg name="publish_saved_markers_tf" default="true" />
<arg name="load_frames_on_init" default="true" />
```

Example:

```
roslaunch marker_mapping frames_mapping.launch frame_id:=rb1_base_map base_frame_id:=rb1_base_base_footprint
```
