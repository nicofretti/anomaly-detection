# goals is a set of 2D poses that will be reached by the robot
Pose2DStamped[] goals
# twist can be either empty, in that case no maximum velocity is
# specified, or a set of the same size as goals, in which each goal
# will be reached with its maximum velocity
Twist2D[] max_velocities
