string NAVIGATION_TYPE_POINT_TO_POINT = POINT_TO_POINT
string NAVIGATION_TYPE_END_POINT= END_POINT

string GOAL_TARGET_TYPE_GPS = GPS
string GOAL_TARGET_TYPE_CARTESIAN = CARTESIAN

geometry_msgs/PoseStamped[] goal_target_pose
string[] controller
string[] planner
string[] recovery_behaviors
geometry_msgs/Pose[] goal_tolerance
string navigation_type
string goal_target_type
float32 max_vel
