# MISSION STATES
# States inside a mission
string UNKNOWN=unknown
string RUNNING=running
string PAUSED=paused
string FINISHED=finished
string CANCELLED=cancelled
string ERROR=error
string QUEUED=queued
# running: mission is running
# paused: mission is paused
# finished: mission finished
# canceled: mission has been canceled
# error: an error occurred during the execution of the mission
# queued: the mission is queued


#
int32 id

# ID of the mision based on string format: GOTO [4,5,6]
string description

# running, paused, finished,...
string state


