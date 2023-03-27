# Type of localization systems used
string LOCALIZATION_TYPE_NONE=none
# amcl
string LOCALIZATION_TYPE_1=1
# gps
string LOCALIZATION_TYPE_2=2
# cartogphr
string LOCALIZATION_TYPE_3=3
# gmapping
string LOCALIZATION_TYPE_4=4

# state of the navigation node running
string LOCALIZATION_STATE_INIT=init
string LOCALIZATION_STATE_READY=ready
string LOCALIZATION_STATE_EMERGENCY=emergency
string LOCALIZATION_STATE_FAILURE=failure
string LOCALIZATION_STATE_UNKNOWN=unknown

# Pose of the robot based on the current localization algorithm
Pose2DStamped pose

# reliability
bool reliable

# Localization environment 
string environment

# Type of localization technology
string type

# INIT, READY, EMERGENCY, FAILURE
string state
