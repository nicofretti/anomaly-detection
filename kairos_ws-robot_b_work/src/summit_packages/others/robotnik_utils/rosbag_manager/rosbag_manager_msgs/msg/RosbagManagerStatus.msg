Header header

# Component's state
robotnik_msgs/State state

bool recording
int32 time_recording

# path to save the rosbag
string path
# Bag name
string bag_name
# Is the bag being compressed
bool compression
# Topics are subscribed using regular expresions
bool regex
# Disk space used in the recording (in MB)
float64 stored_size
