# icp_pose_corrector

Package that uses a registration algorithm to make small corrections in the global localization system

Some localization algorithms (specially AMCL) treat all points equal when computing the pose of a robot in an environment. In highly unstructured environments
this can lead to errors due to:

* points belonging to moving obstacles being mismatched to obstacles in the map,

* closer points being used to compute the orientation, which is best computed using points that are far from the robot,

* farther points being used to compute the position, which is best computed using points that are closer to the robot.

Assuming a continuous proper localization:

* some points can be discarded as not belonging to an object present in the map,

* closer and farther points can be used to compute the position and orientation respectively.

The algorithm works like this:

1. Find points that are closer to a map point than threshold.

1. Use a registration algorithm to compute the transformation between the scan and the map.

1. Decrease the distance threshold for map points and go back to step 1.

During the computing of the algorithm, time and distance travelled by robot must be taken into account.

The algorithm can be triggered by:

* Topic, used to correct the output of another localization algorithm.

* Service, used to trigger the registration algorithm.

## subscribed topics

**input_pose** (geometry_msgs/PoseWithCovarianceStamped)

Topic used to trigger the registration algorithm. Intended to be the output of another localization algorithm

**initialpose_with_registration** (geometry_msgs/PoseWithCovarianceStamped)

Topic used to trigger the initialization of a localization system, with a registration step.

**scan** (sensor_msgs/LaserScan)

Scans used to register with map

**map** (nav_msgs/OccupancyGrid)

Map used to register incoming scans

**/tf** (tf/tfMessage)

TF Topics 

**/tf_static** (tf2_msgs/TFMessage)

TF Topics 

## published topics:

**correctpose** (geometry_msgs/PoseWithCovarianceStamped)

Topic used to correct the pose of another localization algorithm, without triggering an additional computation.

**initialpose** (geometry_msgs/PoseWithCovarianceStamped)

Topic used to initialize the pose of another localization algorithm, maybe also triggering an additional computation in the other algorithm.

**test** (sensor_msgs/PointCloud2)

Debug cloud

**/tf** (tf/tfMessage)

TF Topics 


## advertised services

**register_localization** (std_srvs/Trigger)

Used to trigger the initialization of the underlying localization system. Outputs using the *initialpose* topic.

**request_correction** (std_srvs/Trigger)

Used to trigger the correction of the underlying localization system. Outputs using the *correctpose* topic.

**toggle_correction** (std_srvs/SetBool)

Used to enable or disable the registration algorithm


## parameters

**base_frame** (string, default: "base_footprint")

Frame of the mobile base.

**odom_frame** (string, default: "odom")

Frame of the odometry of the mobile base.

**global_frame** (string, default: "map")

Frame of the global localization system.

**scan_topic** (string, default: "scan")

Input topic used to read scans, if not wanting to use topic remap

**input_pose_topic** (string, default: "input_pose")

Input topic used to trigger the registration algorithm, if not wanting to use topic remap

**initial_pose_topic** (string, default: "initialpose_with_registration")

Input topic used to trigger a initialization of the localization system, if not wanting to use topic remap

**map_topic** (string, default: "map")

Input topic used to read the map, if not wanting to use topic remap

**set_initital_pose_topic** (string, default: "initialpose")

Output topic used to initialize a localization algorithm, if not wanting to use topic remap.

**set_correct_pose_topic** (string, default: "correctpose")

Output topic used to correct a localization algorithm, if not wanting to use topic remap.

**inflation_distance** (double, default: "0.15")

Maximum distance of a laser point to the closest map point, so it is used in the registation.

**obstacle_value** (int, default: "100")

Map value of obstacles.

**outer_iterations** (int, 5)

Iterations of recursive execution

**inner_iterations** (int, 250)

Iterations for each registration execution

**max_allowed_distance_correction** (double, 0.4)

Maximum distance correction, if it is more, it will discard the correction.

**max_allowed_angular_correction**, (double, 0.4)

Maximum angular correction, if it is more, it will discard the correction.

**update_min_d** (double, 0.1)

Euclidean distance before performing a pose registration

**update_min_a**, (double, 0.1)

Angular distance before performing a pose registration

**minimum_coincidences**, (int, 5)

Minimum number of coincidences to accept registration

**minimum_area_covered**, (double, 100)

Minimum area covered (sum of distances traveled by lasers) to accept registration


Some of this parameters are also available through a Dynamic Reconfigure interface

## TODO

- Change name to Registration Localization 
